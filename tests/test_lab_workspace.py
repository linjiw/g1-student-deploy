"""Robot-day workspace: attempt log, the panel's checks and ladder, the node's map snapshot, the lab hub and the panel
routes (g1deploy/attempts.py, workspace.py, node.py map_snapshot, hub.py, panel.py)."""

import json
import math
import threading
import urllib.error
import urllib.request
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
    assert cmd[1:4] == ["sim", "--profile", "deploy"] and "--arm" in cmd and cmd[cmd.index("--panel") + 1] == "8771"
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
