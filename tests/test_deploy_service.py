"""The hub as a deploy service (API v1, g1deploy/hub.py): scans and their frames, the start check and goal conversion,
the act relay, rehearsal sets and their tracks, the live view poller, readiness, practice, evidence (g1deploy/evidence.py)
and the site check's live output (tools/site_check.py --live-out). Fake consoles and nodes; non-identity frames
(start_S = [0.3, -0.2, 1.2], a start offset of 20 deg / 0.2 m)."""

import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from g1deploy import contract, evidence, frames as F, hub, wire

from .test_registration import scene

START_S = [0.3, -0.2, 1.2]
OFFSET = F.offset_pose(-20.0, 0.2)
P_C = [1.7, -0.6, 0.4]
TOKEN = "test-token"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("G1_PANEL_TOKEN", TOKEN)
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    monkeypatch.setattr(hub, "gpu_free_mib", lambda: 12000.0)


def make_hub(tmp_path, **kw):
    args = SimpleNamespace(day=str(tmp_path / "runs" / "lab-day-x"), robot="lab-g1", iface="g1test0", port=8789,
                           no_studio_link=True, **kw)
    return hub.Hub(args)


def coded_map(points, center, half=4.0, res=0.1, robot=None):
    """A node map (StudentNode.map_snapshot shape): seen floor everywhere, '3' at the points."""
    n = int(2 * half / res)
    x0, y0 = center[0] - half, center[1] - half
    rows = [["."] * n for _ in range(n)]
    for x, y in points:
        i, j = int((x - x0) / res), int((y - y0) / res)
        if 0 <= i < n and 0 <= j < n:
            rows[i][j] = "3"
    return {"ok": True, "rows": ["".join(r) for r in rows], "res": res, "x0": x0, "y0": y0, "half": half,
            "robot": list(robot or [center[0], center[1], 0.0])}


def make_scan(h, name, start=START_S, pts_scene=None, age_s=0.0):
    """A scan in memory_S (the robot at `start`), with its twin (memory_S) and course (scene)."""
    d = h.scans_dir / name
    (d / "course").mkdir(parents=True)
    n, res, lo = 241, 0.05, np.array([start[0] - 6.0, start[1] - 6.0])
    sup = np.full((n, n), np.nan)
    ii, jj = np.mgrid[0:n, 0:n]
    sup[np.hypot(lo[0] + ii * res - start[0], lo[1] + jj * res - start[1]) < 5.0] = 0.0
    pts_scene = scene() if pts_scene is None else pts_scene
    pm = F.apply(start, pts_scene)
    idx = np.round((pm - lo) / res).astype(int)
    ok = ((idx >= 0) & (idx < n)).all(axis=1)
    sup[idx[ok, 0], idx[ok, 1]] = 1.2
    goal_xy = F.apply(start, [2.5, -0.3])
    np.savez_compressed(d / "memory.npz", support=sup, ceiling=np.full((n, n), np.nan), lo=lo, res=res,
                        start_xy=np.array(start[:2]), start_yaw=start[2], goal_xy=goal_xy, near=np.zeros((n, n), bool))
    box_m = {"center_xy": F.apply(start, [1.8, 0.3]).tolist(), "half_xy": [0.12, 0.18], "yaw": start[2] + 0.1,
             "z_bottom": 0.0, "z_top": 1.2}
    (d / "twin.json").write_text(json.dumps({"boxes": [box_m]}))
    task = {"scene_boxes_world": [{"center_xy": [1.8, 0.3], "half_xy": [0.12, 0.18], "yaw": 0.1, "z_bottom": 0.0,
                                   "z_top": 1.2}], "goal_world_xy": [2.5, -0.3], "initial_root_xy": [0.0, 0.0]}
    (d / "course" / "task.json").write_text(json.dumps(task))
    (d / "scan.json").write_text(json.dumps({"name": name, "time": "2026-10-28T10:00:00", "verdict": "NO-GO", "frames": 80,
                                             "goal_ahead_left": [2.5, -0.3],
                                             "reasons": ["60 % of the route crosses never-seen cells",
                                                         "goal inside an obstacle box (centre [1, 1])"]}))
    if age_s:
        t = time.time() - age_s
        os.utime(d / "scan.json", (t, t))
    return d


class FakeConsole:
    """A console panel: /state, /map, /act (token checked), recording every act."""

    def __init__(self, state=None, live_map=None, token=TOKEN, act_results=None):
        self.state = state or {"runner": "CONTROL", "phase": "WAIT", "attempts": [], "checks": [], "ladder": []}
        self.map, self.acts, self.token = live_map, [], token
        self.act_results = act_results  # id -> record: a console with GET /act_result (contract v3 D); None: without
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                u = urlparse(self.path)
                if parse_qs(u.query).get("t", [""])[0] != fake.token:
                    return self._reply(403, {})
                if u.path == "/state":
                    return self._reply(200, fake.state)
                if u.path == "/map":
                    return self._reply(200, fake.map or {"ok": False})
                if u.path == "/act_result" and fake.act_results is not None:
                    i = parse_qs(u.query).get("id", [""])[0]
                    return self._reply(200, dict(fake.act_results.get(i) or {"result": None}, known=i in fake.act_results,
                                                 id=i))
                return self._reply(404, {})

            def do_POST(self):
                u = urlparse(self.path)
                if parse_qs(u.query).get("t", [""])[0] != fake.token:
                    return self._reply(403, {})
                msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.acts.append(msg)
                return self._reply(200, {"ok": True, "id": msg.get("id"), "result": "done", "reason": None})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def go_live(h, slot, d, port, **info):
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps({"mode": "robot" if slot == "robot" else "sim"}))
    h.slots[slot] = {"proc": SimpleNamespace(pid=999999, poll=lambda: None), "info": dict({
        "pid": 999999, "mode": slot, "dir": str(d), "panel_port": port, "started": time.time(),
        "port_base": 47000 if slot == "robot" else 47100, "profile": "deploy"}, **info)}


def rehearsal_run(h, name, kind, set_name, result="arrived", scan="scan-a", offset=None, n=200, walk=None, contact=4.5,
                  identity=None, passage=None, goal=(2.5, -0.3), lag=0.2, started=None, ended=True, profile="deploy"):
    d = h.day / name
    d.mkdir(parents=True)
    ran = identity or {}
    (d / "session.json").write_text(json.dumps({"mode": "sim", "odom": "lio" if kind == "drift" else "dds",
                                                "profile": {"name": profile, "sha256": ran.get("profile_sha")},
                                                "bundle_manifest_sha256": ran.get("bundle_sha"),
                                                "deploy_identity": {"rules_sha256": ran.get("rules_sha")}}))
    if ended:
        (d / "console_summary.json").write_text("{}")
        (d / "report.json").write_text(json.dumps({"node": {"max_lag_s": lag, "overruns": 0}, "machine_busy": []}))
    (d / "attempts.jsonl").write_text(json.dumps({"n": 1, "result": result, "duration_s": 8.5}) + "\n")
    if started is not None:
        (d / "hub_start.json").write_text(json.dumps({"t": started}))
    s_from_c = offset or [0.0, 0.0, 0.0]
    goal = list(goal)
    meta = {"scan": scan, "goal": goal, "goal_scene": goal, "profile": profile, "set": set_name,
            "kind": kind, "drift": {"motion": "lab-g1-leg"} if kind == "drift" else None, "scene_from_course": s_from_c}
    if kind == "offset":
        meta["offset"] = {"pose": s_from_c}
    if identity is not None:
        meta.update(identity=identity, started=started)
    if passage is not None:
        meta["passage"] = passage
    (d / "rehearsal.json").write_text(json.dumps(meta))
    t = np.arange(1, 2 * n + 1) * 0.01  # the 100 Hz log
    end = F.apply(F.inverse(s_from_c), goal)  # a walk that ends at the goal on the simulator's truth
    xy = np.c_[np.linspace(0, end[0], 2 * n), np.linspace(0, end[1], 2 * n)] if walk is None else walk
    q = np.zeros((n, 50))
    q[:, :2] = xy[::2]
    q[:, 2] = 0.78
    q[:, 3] = 1.0
    names = hub.model_joint_names(50)
    for k, jn in enumerate(names):
        q[:, 7 + k] = 0.01 * k + (0.5 if "hand" in jn else 0.0)  # hand joints recognisable
    cn = np.zeros(2 * n)
    cn[100:104] = [3.0, contact * 0.8, contact, 4.0]
    np.savez_compressed(d / "sim.npz", t=t, root_pos=np.c_[xy, 0.78 + 0 * t], root_quat=np.tile([1.0, 0, 0, 0], (2 * n, 1)),
                        contact_n=cn, qpos=q)
    return d


# ------------------------------------------------------------------------------------------------- frames on scans
def test_scan_scene_carries_its_frames(tmp_path):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    sc = h.scan_scene("scan-a")
    assert sc["start_S"] == pytest.approx(START_S) and sc["scene_from_memory"] == pytest.approx(F.inverse(START_S))
    assert sc["goal_scene_at_scan"] == pytest.approx([2.5, -0.3], abs=1e-9)
    # the twin (memory_S) lands on the course box (scene) through scene_from_memory
    tw = sc["twin"][0]
    assert F.apply(sc["scene_from_memory"], tw["center_xy"]) == pytest.approx(sc["course_boxes"][0]["center_xy"])
    assert F.wrap(tw["yaw"] + sc["scene_from_memory"][2]) == pytest.approx(sc["course_boxes"][0]["yaw"])
    sup, ceil, near = wire.decode_memory(sc["memory"])
    with np.load(sc["memory_file"]) as z:
        same = np.isfinite(z["support"]) == np.isfinite(sup)
        assert same.all() and np.allclose(np.nan_to_num(z["support"]), np.nan_to_num(sup), atol=0.011)
    assert sc["memory"]["lo"] == pytest.approx([START_S[0] - 6, START_S[1] - 6]) and near is not None
    assert sc["reasons"] == [] and len(sc["site_reasons"]) == 2  # goal-dependent reasons are the goal check's
    json.loads(wire.dumps(sc))


# ------------------------------------------------------------------------------------------------- start check + goal
def test_start_check_and_send_goal_convert_into_odom(tmp_path):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    odom_from_scene = F.compose(P_C, F.inverse(OFFSET))
    world = scene()
    con = FakeConsole(live_map=coded_map(F.apply(odom_from_scene, world), P_C[:2], robot=P_C))
    try:
        go_live(h, "robot", h.day / "robot-100000-deploy", con.port)
        r = h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3]})
        assert not r["ok"] and "check_id" in r["reason"] and not con.acts  # never without a goal check
        r = h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3], "check_id": "gc-1",
                            "operator": "lw", "id": "u-1", "expect": {"phase": "WAIT"}})
        assert r["ok"], r
        c = r["check"]
        assert c["at"] > 1e9 and len(c["t"]) == 2 and c["P_c"] == pytest.approx(P_C)
        assert c["turned_deg"] == pytest.approx(-20.0, abs=1.0) and c["moved_m"] == pytest.approx(0.2, abs=0.06)
        want = F.apply(odom_from_scene, [2.5, -0.3])
        g = r["goal_odom"]
        assert math.hypot(g["x"] - want[0], g["y"] - want[1]) < 0.08 and g["check_id"] == "gc-1"
        act = con.acts[-1]
        assert act["act"] == "goal_odom" and act["seq"] == g["seq"] and act["check_id"] == "gc-1"
        assert act["id"] == "u-1" and act["source"] == "studio" and act["operator"] == "lw" and act["expect"] == {"phase": "WAIT"}
        audit = [json.loads(x) for x in (h.day / "hub_actions.jsonl").read_text().splitlines()]
        assert audit[-1]["act"] == "goal_odom" and audit[-1]["code"] == 200 and audit[-1]["session"] == "robot-100000-deploy"
        rows = [json.loads(x) for x in (h.day / "robot-100000-deploy" / "start_check.jsonl").read_text().splitlines()]
        assert [x["action"] for x in rows] == ["start_check", "send_goal"] and rows[1]["check_id"] == "gc-1"
        # Studio's operator lease goes through like any act; its 2 s renewals are audited only when they change
        for until in (10, 12, 14):
            assert h.act("robot", {"act": "lease", "holder": "lw", "until": until, "id": f"l{until}"})[0] == 200
        h.act("robot", {"act": "lease", "holder": None, "until": 0, "id": "l0"})
        assert [a["until"] for a in con.acts if a["act"] == "lease"] == [10, 12, 14, 0]
        audit = [json.loads(x) for x in (h.day / "hub_actions.jsonl").read_text().splitlines()]
        assert [a["args"].get("holder") for a in audit if a["act"] == "lease"] == ["lw", None]
        # a goal on an obstacle of the live map is refused before anything is sent
        n = len(con.acts)
        r = h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [1.8, 0.3], "check_id": "gc-2"})
        assert not r["ok"] and "obstacle" in r["reason"] and len(con.acts) == n
        # the live view's align is that start check
        v = h.align_of("robot")
        assert v["source"] == "start_check" and v["scene_from_odom"] == pytest.approx(c["scene_from_odom"])
        # the robot is walking: no goal
        con.state = dict(con.state, phase="RUN")
        assert "walking" in h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3],
                                            "check_id": "gc-3"})["reason"]
    finally:
        con.close()


def test_start_check_envelope_needs_the_goals_own_set(tmp_path):
    """The rehearsed start offsets come from the set in force for the goal (tests/test_deploy_service_v3.py); a set from
    before identities, or a start check without a goal, has none."""
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    for kind in ("exact", "drift", "offset"):
        rehearsal_run(h, f"rehearse-1000{RUNS.index(kind)}-{kind}", kind, "set-100000",
                      offset=OFFSET if kind == "offset" else None)
    assert h.offset_envelope("scan-a") is None and h.offset_envelope("scan-a", [2.5, -0.3]) is None
    odom_from_scene = F.compose(P_C, F.inverse(OFFSET))
    con = FakeConsole(live_map=coded_map(F.apply(odom_from_scene, scene()), P_C[:2], robot=P_C))
    try:
        go_live(h, "robot", h.day / "robot-110000-deploy", con.port)
        c = h.start_check_v1("scan-a", "robot")
        assert c["ok"] and c["within_offset_envelope"] is None and c["offset_envelope"] is None
        assert h.start_check_v1("scan-a", "robot", [2.5, -0.3])["within_offset_envelope"] is None
        assert "two numbers" in h.start_check_v1("scan-a", "robot", ["x", 1])["error"]
    finally:
        con.close()


RUNS = ("exact", "drift", "offset")


# ------------------------------------------------------------------------------------------------- rehearsal sets
def test_rehearsal_set_runs_exact_drift_and_offset(tmp_path, monkeypatch):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    launched = []
    monkeypatch.setattr(hub.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or SimpleNamespace(pid=701, poll=lambda: 0))
    monkeypatch.setattr(hub.threading, "Thread", lambda target, args, daemon: SimpleNamespace(start=lambda: target(*args)))
    r = h.rehearse_set({"scan": "scan-a", "goal_scene": [2.5, -0.3], "profile": "deploy", "odom": "leg"})
    assert r["ok"] and len(launched) == 3
    exact, drift, offset = launched
    assert "--odom" not in exact and exact[exact.index("--goal") + 1:exact.index("--goal") + 3] == ["2.5", "-0.3"]
    assert drift[drift.index("--odom") + 1] == "lio" and json.loads(drift[drift.index("--lio-noise") + 1])["motion"] == "lab-g1-leg"
    task = Path(offset[offset.index("--task") + 1])
    assert task.parent.name == "reposed" and "--port-base" in offset and "--arm" not in offset
    g = [float(v) for v in offset[offset.index("--goal") + 1:offset.index("--goal") + 3]]
    assert np.allclose(F.apply(OFFSET, g), [2.5, -0.3], atol=1e-6)  # the same spot in the scene
    rt = json.loads(task.read_text())
    assert np.allclose(F.apply(OFFSET, rt["scene_boxes_world"][0]["center_xy"]), [1.8, 0.3])
    metas = {json.loads((d / "rehearsal.json").read_text())["kind"]: json.loads((d / "rehearsal.json").read_text())
             for d in h.day.glob("rehearse*")}
    assert set(metas) == set(RUNS) and len({m["set"] for m in metas.values()}) == 1
    assert metas["offset"]["scene_from_course"] == pytest.approx(OFFSET) and metas["exact"]["scene_from_course"] == [0, 0, 0]
    assert not h.set_job["running"] and h.set_job["error"] is None
    assert not h.rehearse_set({"scan": "scan-a", "goal_scene": ["x", 1]})["ok"]


def test_rehearsal_sets_verdicts_and_tracks(tmp_path):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    rehearsal_run(h, "rehearse-100000-deploy-twin-scan-a", "exact", "set-100000")
    rehearsal_run(h, "rehearse-drift-100040-deploy-twin-scan-a", "drift", "set-100000")
    rehearsal_run(h, "rehearse-offset-100120-deploy-twin-scan-a", "offset", "set-100000", offset=OFFSET,
                  result="stopped", contact=9.0)
    st = h.refresh()
    s = next(x for x in st["rehearsals"] if x["name"] == "set-100000")
    assert s["verdict"] == "robust" and s["apart_m"] == 0.0 and s["offset_ok"] is False and s["runs"]["drift"]
    assert st["scans"][0]["rehearsals"] == ["set-100000"]
    code, tr = h.track("rehearse-offset-100120-deploy-twin-scan-a")
    assert code == 200 and not tr["running"] and tr["frame"] == "scene" and tr["joint_names"] == list(contract.MJ_JOINTS)
    q = np.array(tr["qpos"])
    assert q.shape[1] == 36 and tr["hz"] == pytest.approx(25.0) and len(q) == 100
    names = hub.model_joint_names(50)
    rsp = contract.MJ_JOINTS.index("right_shoulder_pitch_joint")
    assert q[0, 7 + rsp] == pytest.approx(0.01 * names.index("right_shoulder_pitch_joint"))  # by name, not [:36]
    assert not np.any(q[:, 7:] >= 0.5)  # no hand joint leaked into the 29
    assert q[0, :2] == pytest.approx(OFFSET[:2], abs=1e-4)  # the course origin is the offset in the scene
    assert F.yaw_of_quat(q[0, 3:7]) == pytest.approx(OFFSET[2], abs=1e-3)
    assert [c["n"] for c in tr["contacts"]] == [9.0] and tr["contact_max_n"] == 9.0 and tr["arrived"] is False
    assert tr["final_dist_m"] == pytest.approx(0.0, abs=0.02)  # the fixture walk ends at the goal on the truth
    code, ex = h.track("rehearse-100000-deploy-twin-scan-a")
    assert ex["arrived"] and np.array(ex["qpos"])[-1, 0] == pytest.approx(2.5, abs=0.03)
    assert h.track("../etc")[0] == 404
    json.loads(wire.dumps(tr))


# ------------------------------------------------------------------------------------------------- live view
class FakeNode:
    """A node control socket (REP) that answers `view` with new parts by seq, or an older node that queues it."""

    def __init__(self, port, old=False):
        import zmq

        self.sock = zmq.Context.instance().socket(zmq.REP)
        self.sock.bind(f"tcp://127.0.0.1:{port}")
        self.old, self.reqs, self.stop = old, [], threading.Event()
        self.cloud_seq, self.mem_seq = 3, 1
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while not self.stop.is_set():
            if not self.sock.poll(50):
                continue
            msg = json.loads(self.sock.recv())
            self.reqs.append(msg)
            if self.old:
                self.sock.send_string(json.dumps({"ok": True, "queued": msg.get("cmd")}))
                continue
            out = {"ok": True, "t_mono": time.monotonic() - 0.05, "phase": "WAIT", "frame": "odom",
                   "joint_names": list(contract.MJ_JOINTS), "qpos": [0.0] * 36, "tilt_rad": float("nan")}
            if msg["cloud_after"] < self.cloud_seq:
                out["cloud"] = dict(wire.encode_cloud(np.zeros((5, 3)) + 1.0), seq=self.cloud_seq)
            if msg["mem_after"] < self.mem_seq:
                out["memory"] = dict(wire.encode_memory(np.zeros((4, 4)), np.full((4, 4), np.nan), [0, 0], 0.05),
                                     seq=self.mem_seq, epoch=0)
            self.sock.send_string(json.dumps(out, default=float).replace("NaN", "null"))

    def close(self):
        self.stop.set()
        time.sleep(0.1)
        self.sock.close(0)


def test_view_poller_serves_deltas_and_align(tmp_path):
    h = make_hub(tmp_path)
    port = 47162
    node = FakeNode(port)
    try:
        go_live(h, "sim", h.day / "rehearse-120000-deploy-twin-scan-a", 1, port_base=port - 6, rehearsal=True)
        (h.day / "rehearse-120000-deploy-twin-scan-a" / "rehearsal.json").write_text(json.dumps(
            {"scan": "scan-a", "set": "set-1", "kind": "offset", "scene_from_course": OFFSET}))
        h.ensure_views({"sim": dict(h.live("sim"), running=True), "robot": None})
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3 and h.views["sim"][1].base is None:
            time.sleep(0.05)
        code, v = h.live_view("sim")
        assert code == 200 and v["ok"] and v["cloud"]["seq"] == 3 and v["memory"]["seq"] == 1
        assert 0.0 <= v["served_age_s"] < 1.0 and v["tilt_rad"] is None
        assert v["align"]["source"] == "course" and v["align"]["scene_from_odom"] == pytest.approx(OFFSET)
        code, v2 = h.live_view("sim", cloud_after=3, mem_after=1)
        assert v2["cloud"] is None and v2["memory"] is None and v2["qpos"] is not None
        time.sleep(0.5)
        later = node.reqs[-1]
        assert later["cloud_after"] == 3 and later["mem_after"] == 1  # the poller only asks for new parts
        rate = len(node.reqs) / (time.monotonic() - t0)
        assert rate < 7.0  # one 5 Hz poller, whatever the clients
        json.loads(wire.dumps(v))
    finally:
        for _, p in h.views.values():
            p.stop()
        node.close()


def test_view_poller_backs_off_an_older_node(tmp_path):
    port = 47262
    node = FakeNode(port, old=True)
    p = hub.ViewPoller(port)
    try:
        time.sleep(0.8)
        assert p.status()["supported"] is False and len(node.reqs) == 1 and p.view() is None
    finally:
        p.stop()
        node.close()


# ------------------------------------------------------------------------------------------------- start / stop / practice
def test_v1_robot_start_refuses_a_stale_scan_and_passes_the_operator(tmp_path, monkeypatch):
    h = make_hub(tmp_path)
    monkeypatch.setattr(h, "robot_link", lambda: {"carrier": True, "lio_bridge": False})
    launched = []
    monkeypatch.setattr(hub.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or SimpleNamespace(pid=1, poll=lambda: None))
    ok = {k: True for k in hub.CONFIRM_ROBOT}
    make_scan(h, "scan-old", age_s=hub.SCAN_FRESH_S + 60)
    r = h.start_v1({"mode": "robot", "profile": "deploy", "scan": "scan-old", "confirm": ok, "operator": "lw"})
    assert not r["ok"] and r["error"].startswith("scan stale") and not launched
    make_scan(h, "scan-new")
    r = h.start_v1({"mode": "robot", "profile": "deploy", "scan": "scan-new", "confirm": dict(ok, rope=False)})
    assert not r["ok"] and "rope" in r["error"]
    monkeypatch.setattr(hub, "console_supports", lambda flag: True)
    r = h.start_v1({"mode": "robot", "profile": "deploy", "scan": "scan-new", "confirm": ok, "operator": "lw"})
    assert r["ok"] and launched
    cmd = launched[0]
    assert cmd[cmd.index("--operator") + 1] == "lw" and json.loads(cmd[cmd.index("--confirm") + 1])["rope"] is True
    hs = json.loads((Path(r["dir"]) / "hub_start.json").read_text())
    assert hs["confirm"]["gantry"] and hs["operator"] == "lw"
    assert (Path(r["dir"]) / "scene_scan.txt").read_text().strip() == "scan-new"
    # practice never beside a robot session
    r = h.start_v1({"mode": "sim", "practice": True, "scan": "scan-new", "profile": "deploy"})
    assert not r["ok"] and "never run together" in r["error"]


def test_v1_practice_start_reposes_the_course(tmp_path, monkeypatch):
    h = make_hub(tmp_path)
    make_scan(h, "practice-090000")
    launched = []
    monkeypatch.setattr(hub.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or SimpleNamespace(pid=1, poll=lambda: None))
    r = h.start_v1({"mode": "sim", "practice": True, "scan": "practice-090000", "profile": "deploy", "odom": "fused",
                    "operator": "lw"})
    assert r["ok"], r
    cmd = launched[0]
    assert cmd[cmd.index("--odom") + 1] == "fused" and "--arm" in cmd and cmd[cmd.index("--panel") + 1] == "8771"
    task = json.loads(Path(cmd[cmd.index("--task") + 1]).read_text())
    assert np.allclose(F.apply(OFFSET, task["scene_boxes_world"][0]["center_xy"]), [1.8, 0.3])
    meta = json.loads((Path(r["dir"]) / "practice.json").read_text())
    assert meta["scene_from_course"] == pytest.approx(OFFSET) and Path(r["dir"]).name.startswith("practice-")
    assert h.live("sim")["practice"]
    # a robot session is refused while practice runs
    monkeypatch.setattr(h, "robot_link", lambda: {"carrier": True, "lio_bridge": False})
    assert "practice session is live" in h.start({"mode": "robot", "profile": "deploy",
                                                  "confirm": {k: True for k in hub.CONFIRM_ROBOT}}, v1=True)["error"]
    assert "a session is live" in h.practice_scan({"course": "open-2m"})["error"]


def test_v1_stop_never_while_walking(tmp_path, monkeypatch):
    h = make_hub(tmp_path)
    killed = []
    monkeypatch.setattr(hub.os, "killpg", lambda pid, sig: killed.append(pid))
    con = FakeConsole(state={"runner": "CONTROL", "phase": "RUN", "attempts": [{"n": 1, "result": None}]})
    try:
        go_live(h, "robot", h.day / "robot-130000-deploy", con.port)
        assert "walking" in h.stop_v1("robot")["error"] and not killed
        con.state = {"runner": "CONTROL", "phase": "WAIT", "attempts": []}
        assert "before Arm" in h.stop_v1("robot")["error"]
        con.state = {"runner": "CONTROL", "phase": "ARRIVED", "attempts": [{"n": 1, "result": "arrived"}]}
        assert h.stop_v1("robot")["ok"] and killed == [999999]
        con.state = {"runner": "E-STOP", "phase": "STOPPED", "attempts": []}
        assert h.stop_v1("robot")["ok"]
    finally:
        con.close()


def test_adopt_finds_consoles_of_any_lab_day_with_their_token(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    old = runs / "lab-day-20261027" / "robot-235500-deploy"
    old.mkdir(parents=True)
    (old / "panel_url.txt").write_text("http://127.0.0.1:8770/?t=yesterdays-token\n")
    elsewhere = tmp_path / "other" / "lab-day-20261027" / "sim-1"
    elsewhere.mkdir(parents=True)
    procs = [(4242, f"python g1deploy/console.py robot --profile deploy --arm --panel 8770 --record {old} --odom fused"),
             (4343, f"python g1deploy/console.py sim --profile deploy --record {elsewhere} --port-base 5656")]
    monkeypatch.setattr(hub.machine, "_procs", lambda skip=(): procs)
    monkeypatch.setattr(hub.Adopted, "poll", lambda self: None)
    h = make_hub(tmp_path)
    info = h.live("robot")
    assert info and info["dir"] == str(old) and info["odom"] == "fused" and info["port_base"] == 5556
    assert h._token_for(info) == "yesterdays-token" and h.live("sim") is None  # not a lab-day beside this hub's
    lv = h.live_v1("robot", dict(info, running=True))
    assert lv["panel_token_file"] == str(old / "panel_url.txt") and lv["control_port"] == 5562
    assert "yesterdays-token" not in wire.dumps(lv)


# ------------------------------------------------------------------------------------------------- state / readiness / http
def test_v1_state_lists_robot_sessions_uncapped_and_is_strict(tmp_path):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    for i in range(45):
        d = h.day / f"robot-1{i:05d}-deploy"
        d.mkdir()
        (d / "session.json").write_text(json.dumps({"mode": "robot", "profile": {"name": "deploy"}}))
    for i in range(50):
        d = h.day / f"sim-2{i:05d}-deploy-open-2m"
        d.mkdir()
        (d / "session.json").write_text(json.dumps({"mode": "sim"}))
    st = h.state_v1()
    json.loads(wire.dumps(st))
    assert st["api"] == 1 and "send_goal" in st["capabilities"] and set(st["version"]) == {"kit_commit", "kit_dirty", "private_commit"}
    assert len(st["sessions"]["robot"]) == 45 and len(st["sessions"]["other"]) == hub.SESSIONS_PAGE
    assert st["sessions"]["other_total"] == 50 and h.sessions_v1(offset=40)["other"][-1]["mode"] == "sim"
    assert st["live"] == {"robot": None, "sim": None} and st["phone_panel"] is None
    sc = st["scans"][0]
    assert sc["name"] == "scan-a" and sc["verdict"] == "GO" and sc["site_verdict"] == "NO-GO" and sc["reasons"] == []
    assert {"name", "label", "featured", "note"} <= set(st["profiles"][0])
    assert {"iface", "carrier", "livox_driver", "lidar_hz", "lio_bridge", "lio_pose_age_s", "robot"} <= set(st["robot_link"])
    assert set(st["doctor"]) >= {"t", "exit", "text"} and set(st["gpu"]) == {"free_mib", "rehearsal_mib"}


def test_readiness_before_a_session(tmp_path):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    rd = h.readiness("scan-a", "robot")
    ids = {c["id"]: c for c in rd["checks"]}
    for c in rd["checks"]:
        assert c["status"] in ("pass", "warn", "block", "unknown") and c["group"] in ("link", "robot", "perception", "scene", "operator")
        assert c["evidence"] in ("MEASURED", "CONFIGURED", "OPERATOR", "DERIVED")
        assert c["measurable_after"] in ("connect", "scan", "session", "arm", "node", "goal", "rehearsal")
    assert ids["link.carrier"]["status"] == "block"  # no such interface: no carrier
    assert ids["scene.scan"]["status"] == "pass" and ids["scene.scan"]["anchor"]["kind"] == "pose"
    assert ids["scene.start_check"]["status"] == "unknown" and ids["scene.start_check"]["measurable_after"] == "node"
    assert ids["robot.runner"]["measurable_after"] == "arm" and ids["perception.rear_blind"]["anchor"]["kind"] == "wedge"
    json.loads(wire.dumps(rd))


def test_lidar_and_lio_stats_from_their_logs(tmp_path):
    a = tmp_path / "adapter.log"
    a.write_text("[livox_to_utlidar] 100 packets/s -> 10.0 frames/s, 14000 points/frame, tag-dropped 0, imu 200 Hz\n"
                 "noise\n[livox_to_utlidar] 0 packets/s -> 0.0 frames/s, 0 points/frame, tag-dropped 0, imu 7 Hz\n")
    s = hub.lidar_stats(a)
    assert s["hz"] == 0.0 and s["at"] == pytest.approx(a.stat().st_mtime)
    b = tmp_path / "bridge.log"
    b.write_text("[dds_lio] LIO odom in 5, clouds 2, odom out 55, pelvis [0, 0, 0], LIO pose age 42 ms\n")
    assert hub.lio_stats(b)["pose_age_s"] == pytest.approx(0.042)
    assert hub.lidar_stats(tmp_path / "none.log")["hz"] is None


def test_http_v1_routes(tmp_path):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    server = hub.serve(h, 0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/api/v1/state?t=wrong")
        st = json.loads(urllib.request.urlopen(f"{base}/api/v1/state?t={TOKEN}").read())
        assert st["api"] == 1
        assert "Deploy Studio" in urllib.request.urlopen(f"{base}/?t={TOKEN}").read().decode()
        assert "__TOKEN__" not in urllib.request.urlopen(f"{base}/legacy?t={TOKEN}").read().decode()
        sc = json.loads(urllib.request.urlopen(f"{base}/api/v1/scan/scan-a/scene?t={TOKEN}").read())
        assert sc["start_S"] == pytest.approx(START_S)
        live = json.loads(urllib.request.urlopen(f"{base}/api/v1/scan/live?t={TOKEN}").read())
        assert live["running"] is False and live["joint_names"] == list(contract.MJ_JOINTS)
        for path in ("/api/v1/live/robot/view", "/api/v1/panel/robot/state", "/api/v1/session/nope/manifest",
                     "/api/v1/rehearsal/nope/track", "/api/v1/scan/nope/scene"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(f"{base}{path}?t={TOKEN}")
            assert e.value.code == 404 and "error" in json.loads(e.value.read())

        def post(path, body):
            req = urllib.request.Request(f"{base}{path}?t={TOKEN}", data=json.dumps(body).encode(), method="POST")
            try:
                return json.loads(urllib.request.urlopen(req).read())
            except urllib.error.HTTPError as e:
                return json.loads(e.read())
        r = post("/api/v1/send_goal", {"slot": "robot", "scan": "scan-a", "goal_scene": [1, 0]})
        assert not r["ok"] and r["reason"]
        assert not post("/api/v1/stop", {"slot": "robot"})["ok"]
        assert not post("/api/v1/act", {"slot": "robot", "act": "g", "id": "x"})["ok"]
        assert json.loads((h.day / "hub_actions.jsonl").read_text().splitlines()[-1])["code"] == 404
        rd = json.loads(urllib.request.urlopen(f"{base}/api/v1/readiness?scan=scan-a&t={TOKEN}").read())
        assert rd["checks"]
        d = h.day / "sim-140000-deploy-open-2m"
        d.mkdir()
        (d / "session.json").write_text(json.dumps({"mode": "sim", "odom": "dds"}))
        m = json.loads(urllib.request.urlopen(f"{base}/api/v1/session/{d.name}/manifest?t={TOKEN}").read())
        assert [i["id"] for i in m["items"]] == list(evidence.ITEMS) and (d / "manifest.json").exists()
    finally:
        server.shutdown()


# ------------------------------------------------------------------------------------------------- evidence
def test_manifest_present_missing_and_na(tmp_path):
    d = tmp_path / "robot-150000-deploy"
    (d / "rec" / "lowstate").mkdir(parents=True)
    np.savez(d / "rec" / "lowstate" / "000000.npz", t=np.linspace(0.0, 2.0, 5))  # a real chunk (v2 reads its span)
    (d / "rec" / "meta.json").write_text(json.dumps({"lidar": False, "zmq": ["odom=odom=tcp://127.0.0.1:5561"]}))
    (d / "session.json").write_text(json.dumps({"mode": "robot", "odom": "fused", "deploy_identity": {"short": "a"},
                                                "confirm": {"gantry": True}}))
    (d / "runner.log").write_text("x")
    (d / "node.log").write_text("x")
    (d / "attempts.jsonl").write_text(json.dumps({"n": 1, "result": "arrived", "rehearsal": {"set": "set-1"}}) + "\n")
    (d / "scene_scan.txt").write_text("scan-a\n")
    (tmp_path / "scans" / "scan-a").mkdir(parents=True)
    (tmp_path / "scans" / "scan-a" / "scan.json").write_text("{}")
    m = evidence.manifest(d)
    st = {i["id"]: i for i in m["items"]}
    assert [i["id"] for i in m["items"]] == list(evidence.ITEMS)
    assert st["lowstate"]["status"] == "present" and st["lowstate"]["bytes"] == (d / "rec/lowstate/000000.npz").stat().st_size
    assert st["lowstate"]["span_s"] == 2.0 and st["lowstate"]["required"]
    # raw lidar is a required stream on the robot (2026-10-28): recorded without it, it is missing
    assert st["lidar_raw"]["status"] == "missing" and st["lidar_raw"]["required"] and st["lio_cloud"]["status"] == "missing"
    assert "did not record the LIO cloud" in st["lio_cloud"]["reason"]
    assert st["scan"]["status"] == "present" and st["rehearsal"]["status"] == "present"
    assert st["operator_confirmations"]["status"] == "present" and st["start_checks"]["status"] == "missing"
    assert st["video"]["status"] == "n/a" and m["complete"] is False
    sim = tmp_path / "sim-1"
    sim.mkdir()
    (sim / "session.json").write_text(json.dumps({"mode": "sim", "odom": "dds"}))
    s2 = {i["id"]: i["status"] for i in evidence.manifest(sim)["items"]}
    assert s2["node_events"] == "n/a" and s2["operator_confirmations"] == "n/a" and s2["lio_odom"] == "n/a"


# ------------------------------------------------------------------------------------------------- site check live
def test_site_check_live_out(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import site_check

    q = np.zeros(36)
    q[:3] = [0.3, -0.2, 0.78]
    q[3:7] = [math.cos(0.6), 0, 0, math.sin(0.6)]  # yaw 1.2
    q[7:] = 0.1
    mem = SimpleNamespace(support=np.zeros((5, 5)), ceiling=np.full((5, 5), np.nan), lo=np.array([-1.0, -1.0]),
                          res=0.05, near_only=lambda: np.zeros((5, 5), bool))

    class Conv:  # identity mount and torso at the pelvis; the self filter drops x < 0.2
        front_half_deg, rear_range = 90.0, 1.0
        mount = SimpleNamespace(rot_torso=np.eye(3), pos_torso=np.zeros(3))
        bf = SimpleNamespace(mask=lambda pw, q, origins=None: np.hypot(*(pw[:, :2] - [0.3, -0.2]).T) > 0.2)

        @staticmethod
        def fk(q):
            c, s = math.cos(1.2), math.sin(1.2)
            return np.array([0.3, -0.2, 1.0]), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raw = np.array([[2.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.5, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    out = tmp_path / "live.json"
    site_check.write_live(out, mem, Conv(), raw, q, 12, time.monotonic())
    live = json.loads(out.read_text())
    assert live["running"] and live["n_frames"] == 12 and live["start_S"] == pytest.approx(START_S)
    assert live["qpos"][:2] == pytest.approx([0.0, 0.0], abs=1e-9) and F.yaw_of_quat(live["qpos"][3:7]) == pytest.approx(0.0, abs=1e-9)
    pts, cls = wire.decode_cloud(live["cloud"])
    assert list(cls) == [0, 1, 2, 0]  # fused ahead, self-filtered at the body, rear blind close behind, far rear kept
    assert pts[0, :2] == pytest.approx([2.0, 0.0], abs=0.01)  # scene: ahead of the robot
    assert live["memory"]["n"] == [5, 5] and live["joint_names"] == list(contract.MJ_JOINTS)
    assert not (tmp_path / "live.json.tmp").exists()
