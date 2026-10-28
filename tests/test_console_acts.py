"""The console's acts and GO gate (CONTRACT §4.1; review B2, M3, M4): ids, expect, repeated ids, goals that count only
once the node read them back in this cycle, the gate checked before anything reaches the runner, the Studio lease,
E-STOP replies and the console_events.jsonl evidence. A Session without processes: a fake runner and a fake node."""

import json
import os
import queue
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from g1deploy import console, workspace as W
from g1deploy.attempts import Attempts


class FakeProc:
    """A runner / node process: keys sent, printed lines (console.Proc.new_lines reads them)."""

    def __init__(self):
        self.keys, self.lines, self.count, self.lines_lock, self.up = [], [], 0, threading.Lock(), True

    def key(self, s):
        self.keys.append(s)

    def alive(self):
        return self.up

    def stop(self, *a):
        self.up = False

    def print(self, line):
        with self.lines_lock:
            self.lines.append((time.monotonic(), line))
            self.count += 1

    new_lines = console.Proc.new_lines


class FakeNode:
    """The node's control socket: records commands; a goal is echoed in the next status (goal_seq) unless
    read_back is False; clear_map answers with the node's map_cleared event."""

    def __init__(self, sess):
        self.sess, self.cmds, self.read_back, self.answer = sess, [], True, True

    def __call__(self, **msg):
        self.cmds.append(msg)
        if not self.answer:
            return False
        ns = self.sess.node_status
        if msg["cmd"] == "goal" and self.read_back:
            ns["goal_seq"] = msg["seq"]
            ns["goal"] = [msg["x"], msg["y"]] if msg.get("frame") == "odom" else [msg["ahead"], msg["left"]]
        if msg["cmd"] == "clear_map":
            self.sess.node_events.append({"event": "map_cleared", "t": round(time.monotonic(), 3)})
        return True

    def sent(self, cmd):
        return [c for c in self.cmds if c["cmd"] == cmd]


def session(path, mode="robot", **kw):
    path.mkdir(parents=True, exist_ok=True)
    a = SimpleNamespace(mode=mode, replay=None, operator="ana", goal=[1.5, 0.0], odom="leg" if mode == "robot" else "dds",
                        iface="eno1" if mode == "robot" else "lo", remote_start=False, require_sent_goal=False,
                        profile=None, guards="off", goal_lookahead=None, rear_blind_deg=0.0, rear_blind_range_m=0.0,
                        rules=None, bundle=None, robot=None)
    a.__dict__.update(kw)
    s = console.Session.__new__(console.Session)
    s.args, s.dir, s.t0 = a, path, time.monotonic() - 100.0
    s.events, s.procs, s.node_status, s.inbox = [], {}, {}, queue.Queue()
    s.runner_state = {"init": False, "control": False, "iface": "GAMEPAD", "stream": False, "estop": False}
    s.attempts = Attempts(path / "attempts.jsonl", "deploy")
    s.cleared_t = s.control_t = None
    s.rec_stats, s.remote, s.temps, s.map_cache, s.identity = {}, {"connected": True, "samples": 3, "last": None}, None, (0.0, None), None
    s.map_lock = threading.Lock()
    s.init_acts(a)
    s.node_cmd = FakeNode(s)
    return s


def arm(s):
    """Runner in CONTROL for 5 s, node up and waiting."""
    s.procs["runner"], s.procs["node"] = FakeProc(), FakeProc()
    s.runner_state.update(init=True, control=True)
    s.control_t = time.monotonic() - s.t0 - 5.0
    s.node_status.update(phase="WAIT", goal=[1.5, 0.0], goal_seq=None, xy=[0.0, 0.0],
                         odom={"mode": s.args.odom, "lio_age_s": None, "doubt": False, "lost": None, "watchdog": None})
    return s


def goal_odom(s, seq, source="studio", **kw):
    rec = s.run_act(s.make_act("goal_odom", source, x=2.0, y=0.5, seq=seq, check_id=f"chk-{seq}", **kw))
    s.check_pending()
    return rec


@pytest.fixture(autouse=True)
def no_rehearsal(monkeypatch):
    monkeypatch.setattr(console, "rehearsal_running", lambda: None)


def main_loop(s):
    """The console's main loop, as far as acts go."""
    stop = threading.Event()

    def run():
        while not stop.is_set():
            try:
                kind, rec, _ = s.inbox.get(timeout=0.02)
                if kind == "act":
                    s.run_act(rec)
            except queue.Empty:
                pass
            s.check_pending()
    threading.Thread(target=run, daemon=True).start()
    return stop


# ---------------------------------------------------------------------------------------------------- GO gate
def test_a_refused_go_never_reaches_the_runner(tmp_path):
    """M3: GO used to switch the runner to the ZMQ stream before it refused for the node's warm-up."""
    s = arm(session(tmp_path))
    s.node_status["phase"] = "BOOT"
    rec = s.act("g", source="studio")
    assert rec["result"] == "refused" and "still starting" in rec["reason"]
    assert s.procs["runner"].keys == [] and not s.node_cmd.sent("go")
    s.runner_state["control"] = False
    assert s.go_locks()[0].startswith("Start SONIC first")
    s.runner_state.update(control=True, estop=True)
    assert "E-STOP" in s.act("g")["reason"] and s.procs["runner"].keys == []


def test_robot_go_needs_a_goal_read_back_in_this_cycle(tmp_path):
    """B2: a robot console's launch goal never counts; goal_odom counts once the node echoes its seq, for one attempt;
    the attempt row keeps what was in force at GO."""
    s = arm(session(tmp_path))
    rec = s.act("g", source="studio")
    assert rec["result"] == "refused" and "launch goal never counts" in rec["reason"]
    assert s.procs["runner"].keys == []
    check = {"theta": 0.35, "t": [0.2, -0.1], "overlap": 0.8, "P_c": [0.1, 0.0, 1.2]}
    g = s.run_act(s.make_act("goal_odom", "studio", x=2.0, y=0.5, seq=7, check_id="chk-7", start_check=check,
                             rehearsal={"names": ["exact", "drift", "offset"], "verdict": "robust"}))
    assert g["result"] is None and "read back" in "; ".join(s.go_locks())  # pending: nothing counts yet
    s.check_pending()
    assert g["result"] == "done" and s.goal_seq == 7 and s.node_cmd.sent("goal")[0] == {
        "cmd": "goal", "seq": 7, "frame": "odom", "x": 2.0, "y": 0.5}
    assert s.go_locks() == []
    rec = s.act("g", source="studio")
    assert rec["result"] == "done", rec["reason"]
    assert s.procs["runner"].keys == ["#", "\n"] and len(s.node_cmd.sent("go")) == 1
    row = s.attempts.rows[-1]
    assert row["check_id"] == "chk-7" and row["start_check"] == check and row["rehearsal"]["verdict"] == "robust"
    assert row["odom"] == "leg" and row["goal_seq"] == 7 and row["goal_odom"] == [2.0, 0.5] and row["source"] == "studio"
    assert row["profile"] == "deploy" and row["operator"] == "ana"
    s.node_status["phase"] = "RUN"
    assert "already walking" in s.act("g", source="studio")["reason"]
    s.node_status["phase"] = "ARRIVED"
    s.attempts.update(time.monotonic() - s.t0, s.node_status)  # the walk ended: a new cycle
    rec = s.act("g", source="studio")
    assert rec["result"] == "refused" and "since attempt 1 ended" in rec["reason"]
    assert goal_odom(s, 8)["result"] == "done" and s.go_locks() == []
    assert goal_odom(s, 8)["reason"] == "goal seq 8 was already used in this session"


def test_a_goal_the_node_does_not_read_back_is_refused(tmp_path):
    s = arm(session(tmp_path))
    assert goal_odom(s, 1)["result"] == "done" and s.go_locks() == []
    s.node_cmd.read_back = False
    g = s.act("goal", source="panel", ahead=1.0, left=0.0)
    s.check_pending()
    assert g["result"] is None
    time.sleep(console.GOAL_READBACK_S + 0.05)
    s.check_pending()
    assert g["result"] == "refused" and g["reason"] == "goal not confirmed by the node"
    assert any("no goal" in x for x in s.go_locks())  # the earlier goal does not count any more either
    s.node_cmd.answer = False
    assert s.act("goal", ahead=1.0, left=0.0)["reason"] == "the node did not answer"


def test_the_nodes_goal_changed_outside_the_console(tmp_path):
    s = arm(session(tmp_path))
    goal_odom(s, 3)
    s.node_status["goal"] = [3.0, 0.0]  # e.g. a goal sent straight to the node's control socket
    assert any("changed outside" in x for x in s.go_locks())


def test_clear_map_starts_a_cycle_and_go_waits_for_its_scan(tmp_path):
    s = arm(session(tmp_path))
    goal_odom(s, 1)
    time.sleep(0.01)  # the node's event times are rounded to the ms
    c = s.act("c", source="panel")
    assert c["result"] is None
    s.check_pending()
    assert c["result"] == "done" and s.cleared_t is not None
    locks = "; ".join(s.go_locks())
    assert "no goal sent since Clear map" in locks and "still scanning (10 s left)" in locks
    s.cleared_t -= console.CLEAR_MAP_SCAN_S + 0.5
    assert goal_odom(s, 2)["result"] == "done" and s.go_locks() == []
    s.node_status["phase"] = "RUN"
    assert "stand first" in s.act("c")["reason"]


def test_a_sim_console_counts_its_launch_goal_once_unless_practice(tmp_path):
    s = arm(session(tmp_path / "sim", mode="sim"))
    assert s.go_locks() == []
    assert s.act("g", source="script")["result"] == "done"
    s.node_status["phase"] = "ARRIVED"
    s.attempts.update(time.monotonic() - s.t0, s.node_status)
    assert any("since attempt 1 ended" in x for x in s.go_locks())
    script_goal = s.act("goal", source="script", ahead=1.0, left=0.5)  # a mission's next leg
    s.check_pending()
    assert script_goal["result"] == "done" and s.go_locks() == []
    p = arm(session(tmp_path / "practice", mode="sim", require_sent_goal=True))
    assert any("no goal" in x for x in p.go_locks())


def test_studio_lease_refuses_only_the_phone_panels_go(tmp_path):
    s = arm(session(tmp_path))
    goal_odom(s, 1)
    assert s.run_act(s.make_act("lease", "panel", holder="x", until=time.time() + 5))["result"] == "refused"
    assert s.run_act(s.make_act("lease", "studio", holder="ana (studio)", until=time.time() + 3))["result"] == "done"
    assert s.go_locks() == [] and "operator lease (ana (studio))" in s.go_locks("panel")[0]
    rec = s.act("g", source="panel")
    assert rec["result"] == "refused" and "Studio holds the operator lease" in rec["reason"]
    assert s.procs["runner"].keys == []
    s.run_act(s.make_act("lease", "hub", holder="b", until=time.monotonic() - 1.0))  # monotonic and expired
    assert s.go_locks("panel") == []
    s.run_act(s.make_act("lease", "studio", holder="c", ttl_s=3.0))
    assert s.lease_active()["holder"] == "c"
    s.run_act(s.make_act("lease", "studio", holder=None))
    assert s.lease is None and s.act("g", source="panel")["result"] == "done"


def test_odometry_lost_locks_go(tmp_path):
    s = arm(session(tmp_path, odom="fused"))
    goal_odom(s, 1)
    s.node_status["odom"] = dict(s.node_status["odom"], lost="LIO silent")
    assert any("odometry lost (LIO silent)" in x for x in s.go_locks())


# ---------------------------------------------------------------------------------------------------- the act API
def test_http_acts_expect_ids_and_replies(tmp_path):
    s = arm(session(tmp_path))
    stop = main_loop(s)
    try:
        msg = {"act": "goal", "id": "u1", "source": "studio", "operator": "ana", "ahead": 1.0, "left": 0.2,
               "expect": {"runner": "CONTROL", "session": tmp_path.name, "attempts": 0}}
        code, r = s.submit(msg)
        assert code == 200 and r == {"ok": True, "id": "u1", "act": "goal", "result": "done",
                                     "reason": f"the node read back goal {s.goal_seq}", "repeat": False}
        code, r2 = s.submit(dict(msg))  # a retry: its first result, never a second goal
        assert r2["result"] == "done" and r2["repeat"] and len(s.node_cmd.sent("goal")) == 1
        code, r3 = s.submit({"act": "g", "id": "u2", "source": "studio", "expect": {"goal_seq": s.goal_seq + 1}})
        assert code == 200 and r3["result"] == "refused" and "the state changed: goal_seq" in r3["reason"]
        assert s.procs["runner"].keys == []
        code, r4 = s.submit({"act": "g", "id": "u3", "source": "studio", "expect": {"goal_seq": s.goal_seq}})
        assert r4["result"] == "done" and r4["ok"]
        for bad in ({"act": "g"}, {"act": "fly", "id": "x"}, {"act": "goal", "id": "x", "ahead": "far", "left": 0},
                    {"act": "goal_odom", "id": "x", "x": 1.0, "y": 2.0, "seq": 3},
                    {"act": "g", "id": "x", "source": "browser"}, {"act": "g", "id": "x", "expect": [1]}):
            code, r = s.submit(bad)
            assert code == 400 and r["result"] == "refused" and r["reason"], bad
        code, r = s.submit({"act": "tag", "id": "t1", "n": 1, "tag": "clean", "note": "door frame"})
        assert r["result"] == "done" and s.attempts.rows[0]["tag"] == "clean"
        code, r = s.submit({"act": "tag", "id": "t2", "n": 5, "tag": "clean"})
        assert r["result"] == "refused"
    finally:
        stop.set()
    rows = [json.loads(line) for line in (tmp_path / "console_events.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["u1", "u2", "u3", "t1", "t2"]  # every act once, with its result
    assert rows[0]["source"] == "studio" and rows[0]["operator"] == "ana" and rows[0]["args"]["ahead"] == 1.0
    assert set(rows[0]) >= {"t_mono", "wall", "id", "source", "operator", "act", "args", "result", "reason"}
    assert [a["id"] for a in s.panel_state()["acts"]] == ["u1", "u2", "u3", "t1", "t2"]


def test_an_act_reply_waits_at_most_a_second(tmp_path):
    s = arm(session(tmp_path))  # no main loop running
    t0 = time.monotonic()
    code, r = s.submit({"act": "h", "id": "q1", "source": "panel"})
    assert code == 200 and r["result"] == "queued" and r["ok"] and 0.9 < time.monotonic() - t0 < 1.6
    s.run_act(s.inbox.get_nowait()[1])  # the main loop catches up: the result is logged
    assert json.loads((tmp_path / "console_events.jsonl").read_text().splitlines()[-1])["result"] == "done"


def test_estop_reply_from_the_http_thread(tmp_path):
    s = arm(session(tmp_path / "a"))
    runner = s.procs["runner"]
    threading.Timer(0.02, lambda: runner.print("[Control] EMERGENCY STOP: damping")).start()
    code, r = s.submit({"act": "estop", "source": "studio", "operator": "ana"})  # no id needed
    assert code == 200 and r["ok"] and r["key_sent"] and r["runner_alive"] and r["confirmed"]
    assert r["runner"] == "E-STOP" and runner.keys == ["o"] and isinstance(r["t_mono"], float)
    assert s.inbox.get_nowait()[0] == "node_quit"
    row = json.loads((tmp_path / "a" / "console_events.jsonl").read_text().splitlines()[-1])
    assert row["act"] == "estop" and row["source"] == "studio" and row["operator"] == "ana" and row["result"] == "done"
    idle = session(tmp_path / "b")  # before Arm: acknowledged, nothing to send
    code, r = idle.submit({"act": "estop"})
    assert r["ok"] and not r["key_sent"] and not r["runner_alive"] and not r["confirmed"] and r["runner"] == "not started"


def test_remote_presses_are_acts_with_source_remote(tmp_path):
    s = arm(session(tmp_path))
    s.runner_state.update(iface="ZMQ", stream=True)
    console.Session.on_remote(s, json.dumps({"remote": "press", "button": "Left"}))
    kind, val, src = s.inbox.get_nowait()
    s.act(val, source="remote", detail=src)  # what the main loop does with it
    console.Session.on_remote(s, json.dumps({"remote": "press", "button": "Select"}))
    rows = [json.loads(line) for line in (tmp_path / "console_events.jsonl").read_text().splitlines()]
    assert [(r["act"], r["source"]) for r in rows] == [("h", "remote"), ("estop", "remote")]
    assert rows[0]["args"]["detail"] == "remote D-left" and s.node_cmd.sent("stand")


def test_state_carries_the_gate_and_is_strict_json(tmp_path):
    s = arm(session(tmp_path))
    s.node_status["tilt_rad"] = float("nan")
    s.run_act(s.make_act("lease", "studio", holder="ana", until=time.time() + 3))
    st = json.loads(W.dumps(s.panel_state()))
    assert st["tilt_rad"] is None and st["session"] == tmp_path.name and st["console_t_mono"] > 0
    assert st["go_locks"] and "lease" in st["go_locks_panel"][-1] and st["lease"]["holder"] == "ana"
    assert st["odom"]["mode"] == "leg" and st["driving_source"] == "sonic_planner" and st["motion_set"] is None
    assert 4.5 < st["standing_s"] < 6.0 and st["remote"]["connected"] and st["goal"]["ok"] is False
    goal_odom(s, 4)
    st = s.panel_state()
    assert st["go_locks"] == [] and st["goal"]["seq"] == 4 and st["goal"]["check_id"] == "chk-4"
    assert next(x for x in st["ladder"] if x["key"] == "goal")["state"] == "done"
    s.runner_state.update(iface="GAMEPAD", set="Left")
    assert s.driving() == ("SONIC planner (BOXING set)", "sonic_planner", "BOXING")
    s.runner_state.update(iface="ZMQ", stream=True)
    assert s.driving()[1:] == ("student", None)


def test_new_lines_survive_the_buffer_trim(tmp_path):
    p = console.Proc("t", [sys.executable, "-c", "for i in range(5000): print(i)"], tmp_path / "t.log", dict(os.environ))
    p.p.wait(10)
    t0 = time.monotonic()
    while p.count < 5000 and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    lines, n = p.new_lines(4990)
    assert n == 5000 and [line for _, line in lines] == [str(i) for i in range(4990, 5000)]
    assert p.new_lines(5000) == ([], 5000) and len(p.new_lines(0)[0]) == len(p.lines)


# ---------------------------------------------------------------------------------------------------- evidence
def test_session_json_keeps_confirmations_and_the_gate(tmp_path, monkeypatch):
    class NoProc:
        def __init__(self, name, cmd, log, env, stdin=False, on_line=None):
            self.name, self.cmd = name, cmd

    monkeypatch.setattr(console, "Proc", NoProc)
    monkeypatch.setattr(console, "_env", lambda: dict(os.environ))
    a = console.parse_args(["sim", "--layout", "open-2m", "--arm", "--record", str(tmp_path / "s"), "--port-base", "5756",
                            "--operator", "ana", "--confirm", '{"gantry": true, "estop": true}', "--require-sent-goal"])
    a.machine = None
    s = console.Session(a)
    s.start()
    meta = json.loads((tmp_path / "s" / "session.json").read_text())
    assert meta["operator"] == "ana" and meta["confirm"] == {"gantry": True, "estop": True}
    assert meta["go_gate"]["start_goal_counts"] is False and "runner" not in s.procs
    rec_cmd = s.procs["rec"].cmd
    assert "cloud=cloud=tcp://127.0.0.1:5761" in rec_cmd and "odom=odom=tcp://127.0.0.1:5761" in rec_cmd
    s.events_log.close()


def test_recorder_keeps_the_lio_cloud_and_studio_workers_are_heavy():
    import re

    from g1deploy import machine, recorder

    assert "cloud=cloud=tcp://127.0.0.1:5561" in recorder.DEFAULT_ZMQ
    for cmd in ("/x/.venv/bin/python -u /home/u/g1-deploy-studio/studio/nav_worker.py --request r.json",
                "/x/python -u /home/u/g1-deploy-studio/studio/flow_policy_worker.py", "/x/python -u studio/policy_worker.py"):
        assert any(re.search(p, cmd) for p in machine.heavy_patterns()), cmd
