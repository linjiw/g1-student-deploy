"""Robot-day workspace: attempt log, the panel's checks and ladder, the node's map snapshot, the lab hub and the panel
routes (g1deploy/attempts.py, workspace.py, node.py map_snapshot, hub.py, panel.py)."""

import json
import math
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g1deploy import attempts as A, hub, workspace as W
from g1deploy.node import StudentNode


# ----------------------------------------------------------------------------------------------- attempts
def test_attempt_life_cycle(tmp_path):
    log = A.Attempts(tmp_path / "attempts.jsonl", "deploy")
    log.start(10.0, {"phase": "ARRIVED", "xy": [0.0, 0.0], "goal": [2.0, 0.0], "plans": 5}, goal_rel=(2.0, 0.0))
    log.update(10.5, {"phase": "ARRIVED", "plans": 5})  # the previous walk's ARRIVED: not this attempt's end
    assert log.open is not None
    for t, tilt, z, lat in ((11.0, 0.1, 0.78, 0.09), (12.0, 0.31, 0.70, 0.21), (13.0, 0.2, 0.74, 0.1)):
        log.update(t, {"phase": "RUN", "tilt_rad": tilt, "pelvis_z_m": z, "latency_s": lat, "plans": 5 + int(t) - 10,
                       "no_safe": t == 12.0, "max_lag_s": 0.2})
    log.update(14.0, {"phase": "ARRIVED", "goal_dist_m": 0.31, "xy": [1.7, 0.1], "plans": 8})
    row = log.rows[0]
    assert row["result"] == "arrived" and row["duration_s"] == 4.0 and row["walked_m"] == pytest.approx(1.7, abs=0.01)
    assert (row["max_tilt"], row["min_pelvis_z"], row["max_latency_s"]) == (0.31, 0.7, 0.21)
    assert row["plans"] == 3 and row["no_safe_plans"] == 1 and row["end_goal_dist_m"] == 0.31
    assert log.tag(1, "touched", "left hand on the door frame") and not log.tag(1, "bogus") and not log.tag(2, "clean")
    saved = A.load(tmp_path / "attempts.jsonl")
    assert saved[0]["tag"] == "touched" and "left_start" not in saved[0]


def test_attempt_closes_on_estop_and_new_go():
    log = A.Attempts(None)
    log.start(1.0, {"phase": "WAIT"})
    log.start(2.0, {"phase": "RUN"})  # GO again while open: the first one is superseded
    log.update(3.0, {"phase": "RUN"}, estop=True)
    assert [r["result"] for r in log.rows] == ["superseded", "E-STOP"] and log.open is None


# ----------------------------------------------------------------------------------------------- near body
def _map(cells, yaw=0.0, half=2.0, res=0.1):
    """A node map with the robot at the origin: `cells` maps (x, y) world points to a code."""
    m = int(2 * half / res)
    rows = [[" "] * m for _ in range(m)]
    for (x, y), ch in cells.items():
        rows[int((x + half) / res)][int((y + half) / res)] = ch
    return {"ok": True, "res": res, "half": half, "x0": -half, "y0": -half, "rows": ["".join(r) for r in rows],
            "robot": [0.0, 0.0, yaw], "rear_blind_deg": 180.0, "rear_blind_range_m": 1.0}


def test_near_body_counts_what_the_planner_sees():
    nb = W.near_body(_map({(0.33, 0.05): "2", (-0.33, 0.05): "3", (-1.25, 0.05): "3", (0.15, 0.25): "s", (1.5, 0.0): "3"}))
    assert nb["close"] == 1 and nb["close_cells"][0][0] == pytest.approx(0.35, abs=0.06)  # ahead; behind is in the blind
    assert nb["behind_tall"] == 1 and nb["self_like"] == 1
    turned = W.near_body(_map({(0.05, 0.33): "2"}, yaw=math.pi / 2))  # facing +y: the cell is ahead
    assert turned["close"] == 1 and turned["close_cells"][0][0] > 0.3
    assert W.near_body({"ok": False}) is None


# ----------------------------------------------------------------------------------------------- checks / ladder
def _ctx(**kw):
    c = {"mode": "robot", "t": 100.0, "runner_state": {"init": False, "control": False, "estop": False},
         "node_status": {}, "node_ready": False, "procs": {"rec": True}, "remote": {"connected": True},
         "rec_stats": {"streams": {"lowstate": {"rate_hz": 990.0, "age_s": 0.01}, "lidar": {"rate_hz": 10.0, "age_s": 0.05}}},
         "temps": {"max_c": 48, "max_joint": "left_knee", "ankle_c": 45, "knee_c": 48}, "machine_busy": [],
         "near_body": None, "attempts": [], "cleared_t": None, "control_t": None, "goal_rel": (1.5, 0.0), "goal_t": 0.0}
    c.update(kw)
    return c


def test_checks_flag_temps_remote_and_skip_near_while_walking():
    lv = {c["key"]: c["level"] for c in W.checks(_ctx())}
    assert lv["temps"] == "ok" and lv["remote"] == "ok" and lv["lowstate"] == "ok" and lv["near"] == "na"
    hot = {c["key"]: c for c in W.checks(_ctx(temps={"max_c": 78, "max_joint": "left_ankle_pitch", "ankle_c": 78, "knee_c": 50},
                                              remote={"connected": False}))}
    assert hot["temps"]["level"] == "bad" and hot["remote"]["level"] == "bad" and "60 C" in hot["temps"]["detail"]
    nb = {"close": 3, "close_cells": [], "behind_tall": 2, "near_m": 0.5, "behind_m": [1.0, 1.6], "self_like": 0}
    standing = {c["key"]: c["level"] for c in W.checks(_ctx(near_body=nb, node_status={"phase": "WAIT"}))}
    assert standing["near"] == "warn" and standing["behind"] == "warn"
    walking = {c["key"] for c in W.checks(_ctx(near_body=nb, node_status={"phase": "RUN"}))}
    assert "near" not in walking and "behind" not in walking
    sim = {c["key"]: c["level"] for c in W.checks(_ctx(mode="sim", temps=None))}
    assert sim["temps"] == "na" and "remote" not in sim


def test_ladder_walks_through_a_robot_day():
    def now(ctx):
        steps = W.ladder(dict(ctx, checks=W.checks(ctx)))
        return next(s["key"] for s in steps if s["state"] in ("now",))
    ctx = _ctx()
    assert now(ctx) == "arm"
    ctx["procs"] = {"rec": True, "runner": True, "node": True}
    assert now(ctx) == "init"
    ctx["runner_state"] = {"init": True, "control": True, "estop": False}
    ctx["control_t"] = 50.0
    assert now(ctx) == "planner"
    ctx["node_ready"] = True
    assert now(ctx) == "map"
    ctx["cleared_t"] = 95.0  # 5 s of scan left
    assert now(ctx) == "map"
    ctx["t"] = 106.0
    assert now(ctx) == "go"  # the launch goal counts for the first attempt
    ctx["attempts"] = [{"n": 1, "result": None, "t_end": None, "tag": None}]
    assert now(ctx) == "walk"
    ctx["attempts"] = [{"n": 1, "result": "arrived", "t_end": 120.0, "tag": None}]
    assert now(ctx) == "tag"
    ctx["attempts"][0]["tag"] = "clean"
    assert now(ctx) == "map"  # next attempt: clear the map again (people moved), then a new goal
    ctx.update(cleared_t=121.0, t=140.0)
    assert now(ctx) == "goal"
    ctx["goal_t"] = 130.0
    assert now(ctx) == "go"
    ctx["runner_state"] = dict(ctx["runner_state"], estop=True)
    steps = W.ladder(dict(ctx, checks=[]))
    assert steps[0]["key"] == "estop" and all(s["state"] in ("done", "blocked") for s in steps[1:])


# ----------------------------------------------------------------------------------------------- node map
def test_node_map_snapshot_codes_and_path():
    n, res = 241, 0.05
    lo = np.array([-6.0, -6.0])
    sup, ceil = np.full((n, n), np.nan), np.full((n, n), np.nan)

    def cell(x, y):
        return int((x - lo[0]) / res), int((y - lo[1]) / res)
    sup[100:140, 100:140] = 0.0  # floor around the origin
    sup[cell(1.0, 0.0)] = 0.6  # an obstacle 1 m ahead
    sup[cell(0.0, -1.5)] = 1.4  # a tall one 1.5 m to the right
    ceil[cell(-0.5, 0.5)] = 1.2  # a beam over floor
    sup[cell(0.2, 0.3)] = 0.9  # a self leak: seen only from up close
    near = np.zeros((n, n), bool)
    near[cell(0.2, 0.3)] = True
    mem = SimpleNamespace(res=res, lo=lo, n=n, support=sup, ceiling=ceil, near_only=lambda: near, stats={"points": 1})
    plan = {"qpos": np.column_stack([np.linspace(0, 1, 20), np.zeros(20)])}
    node = SimpleNamespace(memory=mem, status={"xy": [0.0, 0.0], "yaw_rad": 0.0, "goal": [2.0, 0.0], "phase": "WAIT"},
                           mem_lock=threading.Lock(), active=(plan, 0, None, None), MAP_CODES=StudentNode.MAP_CODES,
                           args=SimpleNamespace(rear_blind_deg=180.0, rear_blind_range_m=1.0))
    m = StudentNode.map_snapshot(node, 2.0, 0.1)
    assert m["ok"] and len(m["rows"]) == 40 and all(len(r) == 40 for r in m["rows"])

    def code(x, y):
        return m["rows"][int((x - m["x0"]) / m["res"])][int((y - m["y0"]) / m["res"])]
    assert code(1.0, 0.0) == "2" and code(0.0, -1.5) == "3" and code(-0.5, 0.5) == "^" and code(0.2, 0.3) == "s"
    assert code(0.5, 0.5) == "." and code(1.9, 1.9) == " "
    assert len(m["path"]) == 4 and m["goal"] == [2.0, 0.0] and m["rear_blind_deg"] == 180.0
    json.dumps(m, default=float)
    node.memory = None
    assert StudentNode.map_snapshot(node)["ok"] is False


# ----------------------------------------------------------------------------------------------- hub
def _hub(tmp_path):
    args = SimpleNamespace(day=str(tmp_path / "lab-day-x"), robot="lab-g1", iface="lo", port=8769)
    h = hub.Hub(args)
    h.studio_link = None
    return h


def test_hub_commands_and_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    cmd, out = h.command({"mode": "sim", "profile": "deploy", "course": "open-2m"})
    assert cmd[:3] == ["nice", "-n", "10"] and cmd[4:7] == ["sim", "--profile", "deploy"] and "--arm" in cmd
    assert cmd[cmd.index("--panel") + 1] == "8771" and cmd[cmd.index("--port-base") + 1] == str(hub.REHEARSAL_PORT_BASE)
    assert cmd[cmd.index("--layout") + 1] == "open-2m" and out.parent == h.day
    cmd, _ = h.command({"mode": "robot", "profile": "deploy", "goal": [2.0, 0.5], "note": "door 1"})
    assert cmd[cmd.index("--robot") + 1] == "lab-g1" and cmd[cmd.index("--goal") + 1:cmd.index("--goal") + 3] == ["2.0", "0.5"]
    assert cmd[cmd.index("--panel") + 1] == "8770" and "--arm" in cmd
    with pytest.raises(ValueError):
        h.command({"mode": "fly", "profile": "deploy"})
    r = h.start({"mode": "robot", "profile": "deploy", "confirm": {"gantry": True}})
    assert not r["ok"] and "checklist" in r["error"]
    r = h.start({"mode": "sim", "profile": "no-such-profile"})
    assert not r["ok"] and "no deploy profile" in r["error"]
    assert h.stop()["ok"] is False  # nothing running
    assert hub.studio_id(h.day, h.day / "robot-120000-deploy") == "robot-lab-day-x-robot-120000-deploy"


def test_hub_lists_sessions_with_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    s = h.day / "sim-120000-deploy-open-2m"
    s.mkdir()
    (s / "session.json").write_text(json.dumps({"mode": "sim", "profile": {"name": "deploy"}, "task": "/x/open-2m/task.json",
                                                "deploy_identity": {"short": "abc"}}))
    (s / "attempts.jsonl").write_text(json.dumps({"n": 1, "result": "arrived", "tag": "clean"}) + "\n")
    (s / "report.md").write_text("# report\n")
    st = h.state()
    x = st["sessions"][0]
    assert (x["profile"], x["identity"], x["task"], x["attempts"][0]["result"]) == ("deploy", "abc", "open-2m", "arrived")
    assert h.report(s.name).startswith("# report") and h.report("../../etc") is None
    assert {p["name"] for p in st["profiles"]} >= {"deploy", "deploy-full-lidar"}


# ----------------------------------------------------------------------------------------------- panel routes
def test_panel_serves_state_map_and_tags():
    from g1deploy.panel import start_panel

    log = A.Attempts(None, "deploy")
    log.start(1.0, {"phase": "WAIT"})
    sess = SimpleNamespace(panel_state=lambda: {"ok": 1}, node_map=lambda: {"ok": True, "rows": []}, attempts=log,
                           inbox=__import__("queue").Queue(), estop=lambda src: None)
    server, token = start_panel(sess, 0, token="tok")
    port = server.server_address[1]
    try:
        base = f"http://127.0.0.1:{port}"
        assert json.loads(urllib.request.urlopen(f"{base}/map?t=tok").read())["ok"] is True
        page = urllib.request.urlopen(f"{base}/?t=tok").read().decode()
        assert "G1 Lab Console" in page and "__TOKEN__" not in page
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/state?t=wrong")
        req = urllib.request.Request(f"{base}/act?t=tok", data=json.dumps({"act": "tag", "n": 1, "tag": "clean"}).encode(),
                                     method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["ok"] and log.rows[0]["tag"] == "clean"
        bad = urllib.request.Request(f"{base}/act?t=tok", data=json.dumps({"act": "tag", "n": "x"}).encode(), method="POST")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
    finally:
        server.shutdown()


# ----------------------------------------------------------------------------------------------- scans and rehearsals
def test_code_grid_matches_the_node_map_codes():
    n, res = 241, 0.05
    lo = np.array([-6.0, -6.0])
    sup, ceil = np.full((n, n), np.nan), np.full((n, n), np.nan)
    sup[100:140, 100:140] = 0.0
    sup[int((1.0 + 6) / res), int(6 / res)] = 1.2
    g = W.code_grid(sup, ceil, lo, res, [0.0, 0.0], 2.0, 0.1)
    assert len(g["rows"]) == 40 and g["res"] == pytest.approx(0.1)
    i, j = int((1.0 - g["x0"]) / g["res"]), int((0.0 - g["y0"]) / g["res"])
    assert g["rows"][i][j] == "3" and g["rows"][20][20] == "."


def _scan(h, name, start=(0.5, -0.2), yaw=0.3):
    d = h.scans_dir / name
    (d / "course").mkdir(parents=True)
    n, res = 241, 0.05
    sup = np.full((n, n), np.nan)
    sup[110:130, 110:130] = 0.0
    np.savez_compressed(d / "memory.npz", support=sup, ceiling=np.full((n, n), np.nan), lo=np.array([-6.0, -6.0]),
                        res=res, start_xy=np.array(start), start_yaw=yaw)
    (d / "course" / "task.json").write_text("{}")
    (d / "scan.json").write_text(json.dumps({"name": name, "verdict": "GO", "course": str(d / "course/task.json")}))
    return d


def test_hub_scan_grid_rehearsal_command_and_trajectory(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    _scan(h, "scan-100000")
    g = h.scan_grid("scan-100000")
    assert g["start"] == pytest.approx([0.5, -0.2, 0.3]) and len(g["rows"]) == 100
    assert h.scan_grid("../x") is None and h.scan_file("scan-100000", "../../etc/passwd") is None
    cmd, out = h.command({"mode": "sim", "profile": "deploy", "course": "scan:scan-100000", "goal": [1.5, 0.2],
                          "script": hub.REHEARSE_SCRIPT, "label": "rehearse"})
    assert cmd[cmd.index("--task") + 1].endswith("scan-100000/course/task.json") and "--layout" not in cmd
    assert "--arm" not in cmd and cmd[cmd.index("--script") + 1] == hub.REHEARSE_SCRIPT
    assert cmd[cmd.index("--goal") + 1:cmd.index("--goal") + 3] == ["1.5", "0.2"] and out.name.startswith("rehearse-")
    with pytest.raises(ValueError):
        h.command({"mode": "sim", "profile": "deploy", "course": "scan:nope"})
    assert h.rehearse({"profile": "deploy", "scan": None, "goal": None})["ok"] is False
    s = h.day / "rehearse-100500-deploy-twin-scan-100000"
    (s / "plans").mkdir(parents=True)
    t = np.linspace(0, 1, 50)
    np.savez_compressed(s / "sim.npz", root_pos=np.column_stack([2 * t, 0 * t, 0.75 + 0 * t]), contact_n=np.zeros(50))
    np.savez_compressed(s / "plans" / "plan_0000_out.npz", qpos=np.zeros((50, 36)))
    (s / "attempts.jsonl").write_text(json.dumps({"n": 1, "result": "arrived"}) + "\n")
    tr = h.trajectory(s.name)
    assert tr["walked"][-1][0] == pytest.approx(2.0) and len(tr["plans"]) == 1 and tr["attempts"][0]["result"] == "arrived"
    assert tr["contact_max_n"] == 0.0 and tr["running"] is True and h.trajectory("../x") is None


def test_robot_start_scans_first_unless_the_scan_is_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    monkeypatch.setattr(h, "robot_link", lambda: {"carrier": True})
    calls = []
    monkeypatch.setattr(h, "scan", lambda **kw: calls.append(kw) or {"ok": True, "name": "scan-x"})
    launched = []
    monkeypatch.setattr(hub.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or SimpleNamespace(pid=1, poll=lambda: None))
    ok = {k: True for k in ("gantry", "estop", "remote", "clear")}
    r = h.start({"mode": "robot", "profile": "deploy", "goal": [2.0, 0.0], "confirm": ok})
    assert r["scanning"] and calls[0]["then"]["goal"] == [2.0, 0.0] and not launched
    _scan(h, "scan-120000")
    r = h.start({"mode": "robot", "profile": "deploy", "goal": [2.0, 0.0], "confirm": ok, "scene_scan": "scan-120000"})
    assert r["ok"] and len(calls) == 1 and launched and "--arm" in launched[0]
    assert (Path(r["dir"]) / "scene_scan.txt").read_text().strip() == "scan-120000"


def test_hub_does_not_count_its_own_session_as_another_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    h.slots["robot"] = {"proc": SimpleNamespace(pid=500, poll=lambda: None), "info": {"pid": 500}}
    snap = {"load": [1, 1, 1], "cpus": 28, "gpu": None, "heavy": [],
            "chains": [{"pid": 500, "cmd": "console.py robot"}, {"pid": 501, "cmd": "node.py"}, {"pid": 900, "cmd": "node.py"}]}
    monkeypatch.setattr(hub.machine, "snapshot", lambda own_pid=None: dict(snap, chains=list(snap["chains"])))
    monkeypatch.setattr(hub.machine, "_ancestors", lambda pid: {pid, 500} if pid in (500, 501) else {pid})
    pc = h.pc()
    assert [c["pid"] for c in pc["chains"]] == [900] and pc["busy"] and "900" in pc["busy"][0]


def test_a_walk_makes_older_scans_stale(tmp_path, monkeypatch):
    import os
    import time

    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    d = _scan(h, "scan-130000")
    old = time.time() - 60
    os.utime(d / "scan.json", (old, old))
    s = h.day / "robot-130500-deploy"
    s.mkdir()
    (s / "session.json").write_text("{}")
    assert h.walked_since(old) == []  # a session that did not walk leaves the scan valid
    (s / "attempts.jsonl").write_text(json.dumps({"n": 1, "result": "arrived", "walked_m": 2.3}) + "\n")
    assert h.walked_since(old)[0]["walked_m"] == 2.3
    assert h.list_scans()[0]["walked_since"][0]["session"] == "robot-130500-deploy"


def test_rehearsal_beside_a_live_robot_session(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    _scan(h, "scan-140000")
    h.slots["robot"] = {"proc": SimpleNamespace(pid=700, poll=lambda: None),
                        "info": {"pid": 700, "mode": "robot", "dir": str(h.day / "robot-x"), "panel_port": 8770}}
    launched = []
    monkeypatch.setattr(hub.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or SimpleNamespace(pid=701, poll=lambda: None))
    req = {"profile": "deploy", "scan": "scan-140000", "goal": [1.5, 0.0]}
    monkeypatch.setattr(hub, "gpu_free_mib", lambda: 12000.0)
    monkeypatch.setattr(h, "panel_state", lambda cur: {"phase": "ARRIVED"})  # an old console: no interlock
    assert "before rehearsals could run beside it" in h.rehearse(req)["error"] and not launched
    monkeypatch.setattr(h, "panel_state", lambda cur: {"phase": "RUN", "interlock": True})
    assert "walking" in h.rehearse(req)["error"]
    monkeypatch.setattr(h, "panel_state", lambda cur: {"phase": "ARRIVED", "interlock": True})
    r = h.rehearse(req)
    assert r["ok"] and launched and "--port-base" in launched[0] and h.live("sim")["pid"] == 701
    assert "robot session is live" in h.start({"mode": "robot", "profile": "deploy"})["error"]
    assert "rehearsal is running" in h.scan(goal=[1, 0])["error"]


def test_console_ports_follow_the_base():
    from g1deploy import console

    assert console.chain_ports() == {"stream": 5556, "debug": 5557, "cloud": 5560, "odom": 5561, "control": 5562}
    a = console.parse_args(["sim", "--layout", "open-2m", "--port-base", "5656"])
    assert a.port_base == 5656 and console.chain_ports(a.port_base)["control"] == 5662
    import re
    from g1deploy.machine import ROBOT_CHAIN
    assert re.search(ROBOT_CHAIN, "x/g1deploy/console.py robot --robot lab-g1")
    assert re.search(ROBOT_CHAIN, "/x/g1_deploy_onnx_ref eno1 /m.onnx")
    assert not re.search(ROBOT_CHAIN, "/x/g1_deploy_onnx_ref lo /m.onnx")
    assert re.search(ROBOT_CHAIN, "python -u g1deploy/node.py --iface eno1 --cloud dds")
    assert not re.search(ROBOT_CHAIN, "python -u g1deploy/node.py --iface lo --cloud dds")
    assert not re.search(ROBOT_CHAIN, "x/g1deploy/console.py sim --layout open-2m")


def test_rehearsal_beside_the_robot_waits_for_gpu_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    _scan(h, "scan-150000")
    h.slots["robot"] = {"proc": SimpleNamespace(pid=700, poll=lambda: None),
                        "info": {"pid": 700, "mode": "robot", "dir": str(h.day / "robot-x"), "panel_port": 8770}}
    monkeypatch.setattr(h, "panel_state", lambda cur: {"phase": "ARRIVED", "interlock": True})
    monkeypatch.setattr(hub, "gpu_free_mib", lambda: 5800.0)
    r = h.rehearse({"profile": "deploy", "scan": "scan-150000", "goal": [1.5, 0.0]})
    assert not r["ok"] and "GB free" in r["error"]


def test_rehearsal_pair_exact_then_drift_and_its_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    _scan(h, "scan-140000")
    launched = []
    monkeypatch.setattr(hub.subprocess, "Popen", lambda cmd, **kw: launched.append((cmd, kw["env"])) or SimpleNamespace(pid=701, poll=lambda: 0))
    monkeypatch.setattr(hub.threading, "Thread", lambda target, args, daemon: SimpleNamespace(start=lambda: target(*args)))
    assert h.rehearse({"profile": "deploy", "scan": "scan-140000", "goal": [1.5, 0.0]})["ok"]
    (exact, env0), (drift, env1) = launched
    assert "--odom" not in exact and "G1_LEG_ODOM" not in env0
    assert drift[drift.index("--odom") + 1] == "leg" and json.loads(env1["G1_LEG_ODOM"]) == hub.DRIFT
    dirs = sorted(h.day.glob("rehearse*"), key=lambda d: "drift" in d.name)
    t = np.linspace(0, 1, 50)
    for d, side in zip(dirs, (0.0, 1.0)):  # the drifted walk goes 1 m to the side: another way
        (d / "session.json").write_text("{}")
        (d / "console_summary.json").write_text("{}")
        (d / "attempts.jsonl").write_text(json.dumps({"n": 1, "result": "arrived"}) + "\n")
        np.savez_compressed(d / "sim.npz", root_pos=np.column_stack([2 * t, side * np.sin(np.pi * t), 0.75 + 0 * t]),
                            contact_n=np.zeros(50))
    pairs = h.rehearsal_pairs([hub.session_summary(d, h.day) for d in dirs])
    assert pairs[dirs[0].name]["verdict"] == "other way" and pairs[dirs[0].name]["apart_m"] > 0.9


def test_robot_start_odometry_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    h = _hub(tmp_path)
    cmd, _ = h.command({"mode": "robot", "profile": "deploy", "goal": [2.0, 0.0], "odom": "fused"})
    assert cmd[cmd.index("--odom") + 1] == "fused"
    with pytest.raises(ValueError):
        h.command({"mode": "robot", "profile": "deploy", "odom": "dds"})
    monkeypatch.setattr(h, "robot_link", lambda: {"carrier": True, "lio_bridge": False})
    ok = {"gantry": True, "estop": True, "remote": True, "clear": True}
    r = h.start({"mode": "robot", "profile": "deploy", "odom": "fused", "confirm": ok})
    assert not r["ok"] and "g1 lio start" in r["error"]
