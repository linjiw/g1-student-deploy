"""Start-pose check: the live map registered to the scan's (g1deploy/registration.py)."""
import math

import numpy as np

from g1deploy import registration as R


def coded(points, center, half=4.0, res=0.1):
    """A workspace.code_grid-like dict with '3' at the given world points."""
    n = int(2 * half / res)
    x0, y0 = center[0] - half, center[1] - half
    rows = [[" "] * n for _ in range(n)]
    for x, y in points:
        i, j = int((x - x0) / res), int((y - y0) / res)
        if 0 <= i < n and 0 <= j < n:
            rows[i][j] = "3"
    return {"rows": ["".join(r) for r in rows], "res": res, "x0": x0, "y0": y0}


def scene():
    rng = np.random.default_rng(1)
    pts = []
    for cx, cy, hx, hy in [(1.8, 0.3, 0.12, 0.18), (1.8, 1.3, 0.12, 0.18), (-1.2, 0.2, 0.25, 0.2), (0.5, -2.5, 0.6, 0.5),
                           (-1.5, 2.5, 1.5, 0.1), (3.0, -1.0, 0.1, 1.0)]:
        for _ in range(int(400 * hx * hy / 0.01) + 20):
            pts.append((cx + rng.uniform(-hx, hx), cy + rng.uniform(-hy, hy)))
    return np.array(pts)


def test_recovers_a_turn_and_a_move_and_converts_the_goal():
    world = scene()
    scan_pose = (0.0, 0.0, 0.0)
    now = (0.15, -0.2, math.radians(-17.0))  # the robot now: 0.25 m off, turned 17 deg right
    scan = dict(coded(world, scan_pose[:2], 5.0), start=list(scan_pose))
    live = dict(coded(world, now[:2]), ok=True, robot=list(now))
    c = R.start_check(scan, live, goal_scan=[2.7, 0.4])
    assert c["ok"] and c["overlap"] > 0.8
    assert abs(c["turned_deg"] + 17.0) < 1.0 and abs(c["moved_m"] - 0.25) < 0.06
    # the same spot in the scene, seen from the robot now
    g = R.to_frame(np.array([[2.7, 0.4]]), now)[0]
    assert np.hypot(*(np.array(c["goal_now"]) - g)) < 0.08


def test_a_different_scene_is_not_trusted():
    world = scene()
    other = scene() * [-1.0, 1.0] + [0.3, 0.0]
    scan = dict(coded(world, (0, 0), 5.0), start=[0.0, 0.0, 0.0])
    live = dict(coded(other, (0, 0)), ok=True, robot=[0.0, 0.0, 0.0])
    assert not R.start_check(scan, live, [2.0, 0.0])["ok"]


def test_repose_task_moves_boxes_into_the_robot_frame():
    task = {"scene_boxes_world": [{"center_xy": [1.0, 0.0], "half_xy": [0.1, 0.1], "yaw": 0.0}], "goal_world_xy": [2.0, 0.0]}
    out = R.repose_task(task, math.radians(90.0), [0.5, 0.0])
    assert np.allclose(out["scene_boxes_world"][0]["center_xy"], [0.5, 1.0], atol=1e-9)
    assert np.allclose(out["goal_world_xy"], [0.5, 2.0], atol=1e-9)


def test_hub_sends_the_scan_goal_as_the_same_spot(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from g1deploy import hub

    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    monkeypatch.setenv("G1_PANEL_TOKEN", "test-token")
    h = hub.Hub(SimpleNamespace(day=str(tmp_path / "lab-day-x"), robot="lab-g1", iface="lo", port=8769))
    world, now = scene(), (0.15, -0.2, math.radians(-17.0))
    (h.day / "robot-x").mkdir(parents=True)
    h.slots["robot"] = {"proc": SimpleNamespace(pid=700, poll=lambda: None),
                        "info": {"pid": 700, "mode": "robot", "dir": str(h.day / "robot-x"), "panel_port": 8770}}
    monkeypatch.setattr(h, "walking", lambda: False)
    monkeypatch.setattr(h, "scan_grid", lambda name: dict(coded(world, (0, 0), 5.0), start=[0.0, 0.0, 0.0]))
    sent = []

    def proxy(method, path, query="", body=None, slot="robot"):
        if path == "/map":
            return 200, json.dumps(dict(coded(world, now[:2]), ok=True, robot=list(now))).encode(), "application/json"
        sent.append(json.loads(body))
        return 200, b"{}", "application/json"
    monkeypatch.setattr(h, "panel_proxy", proxy)
    r = h.send_goal([2.7, 0.4], scan="scan-x")
    g = R.to_frame(np.array([[2.7, 0.4]]), now)[0]
    assert r["ok"] and abs(sent[0]["ahead"] - g[0]) < 0.08 and abs(sent[0]["left"] - g[1]) < 0.08
    assert json.loads((h.day / "robot-x" / "start_check.jsonl").read_text())["action"] == "send_goal"
    h.send_goal([1.0, 0.5])  # no scan: relative to the robot now, unchanged
    assert {k: v for k, v in sent[-1].items() if k not in ("id", "source")} == {"act": "goal", "ahead": 1.0, "left": 0.5}
