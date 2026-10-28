"""The node's live view (CONTRACT §2; review M2, M8, M9): strict JSON, the plan's frames from the current tick, the full
memory grid with its seq and epoch, the last lidar frame by class (fused, self-filtered, rear-blind dropped), odometry
health per mode. A node without planner or sensors (StudentNode.__new__); tests/test_node_supervisor.py polls a real one
over its control socket."""

import base64
import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g1deploy import contract as C, node as N
from g1deploy.node import StudentNode


def strict_loads(text):
    def bad(c):
        raise AssertionError(f"non-strict JSON constant {c}")
    return json.loads(text, parse_constant=bad)


def fake_node():
    n = StudentNode.__new__(StudentNode)
    n.args = SimpleNamespace(odom="leg", rear_blind_deg=180.0, rear_blind_range_m=1.0, mount_calib=None)
    now = time.monotonic()
    q = np.zeros(36)
    q[3], q[0], q[7] = 1.0, float("nan"), 0.25  # a NaN pose: exactly when odometry fails
    n.view_state = (now - 0.01, q, 0.004)
    n.pose_hist = tuple((round(now - 0.05 * k, 3), 0.1 * k, 0.0, 0.0) for k in range(9, -1, -1))
    n.status = {"phase": "RUN", "goal": [2.0, 0.0], "goal_seq": 5, "goal_dist_m": float("nan"), "standing": False,
                "tilt_rad": 0.05, "latency_s": 0.08}
    n.tick, n.plan_seq = 100, 3
    plan = np.zeros((60, 36))
    plan[:, 0] = np.arange(60)  # row index in x: which rows the view picked
    n.active = ({"qpos": plan}, 90, None, None)  # anchored at tick 90: row 9 plays at tick 100
    n.route = SimpleNamespace(path=np.column_stack([np.linspace(0, 2, 500), np.zeros(500)]))
    n.mem_lock = threading.Lock()
    res, lo = 0.05, np.array([-6.0, -6.0])
    sup, ceil = np.full((241, 241), np.nan), np.full((241, 241), np.nan)
    sup[100:140, 100:140] = 0.0
    sup[130, 120] = 0.62
    sup[131, 120] = 9.0  # clipped to 5.08 m
    ceil[110, 120] = 1.2
    near = np.zeros((241, 241), bool)
    near[121, 121] = True
    n.memory = SimpleNamespace(support=sup, ceiling=ceil, lo=lo, res=res, n=241, near_only=lambda: near)
    n.memory_epoch = 0
    return n


def decode_cloud(c):
    xyz = np.frombuffer(base64.b64decode(c["points_b64"]), dtype="<i2").reshape(-1, 3) / 100.0 + c["origin"]
    return xyz, np.frombuffer(base64.b64decode(c["class_b64"]), dtype=np.uint8)


def test_view_reply_is_strict_and_complete():
    n = fake_node()
    n.view_memory(time.monotonic())
    fused = np.column_stack([np.linspace(-3, 3, 9000), np.linspace(1, 2, 9000), np.zeros(9000)])
    fused[5] = np.nan  # dropped, never cast to int16
    selfp = np.tile([[0.1, 0.0, 1.0]], (50, 1))
    rear = np.column_stack([np.linspace(-0.9, -0.3, 2000), np.zeros(2000), np.full(2000, 0.5)])
    n.view_frame(123.4, {0: fused, 1: selfp, 2: rear}, np.array([0.0, 0.0, 1.25]))
    v = strict_loads(n.view_reply({"cmd": "view", "cloud_after": 0, "mem_after": 0, "max_points": 4000}))
    assert v["ok"] and v["frame"] == "odom" and v["phase"] == "RUN" and v["joint_names"] == list(C.MJ_JOINTS)
    assert np.asarray(v["mount"]["rot_torso"]).shape == (3, 3) and len(v["mount"]["pos_torso"]) == 3
    assert len(v["qpos"]) == 36 and v["qpos"][0] is None and v["qpos"][7] == 0.25 and v["goal_dist_m"] is None
    assert 0.01 < v["state_age_s"] < 1.0 and abs(v["t_mono"] - time.monotonic()) < 1.0
    assert len(v["pose_hist"]) == 10 and v["goal"] == [2.0, 0.0] and v["goal_seq"] == 5
    p = v["plan"]  # rows from the current tick: row 9 now, +0.4 s = row 29, +0.9 s = row 54
    assert p["seq"] == 3 and p["frames_t"] == [0.4, 0.9] and [f[0] for f in p["frames"]] == [29.0, 54.0]
    assert p["path"][0] == [9.0, 0.0] and len(p["path"]) == 11
    assert len(v["route"]) <= 200 and v["route"][0] == [0.0, 0.0]
    assert v["odom"]["mode"] == "leg" and v["odom"]["lio_age_s"] is None and v["rear_blind_deg"] == 180.0
    c = v["cloud"]
    xyz, cls = decode_cloud(c)
    assert c["seq"] == 1 and c["n"] == len(xyz) == len(cls) <= 4000 and c["total"] == [8999, 50, 2000]
    assert (cls == 1).sum() == 50 and (cls == 2).sum() == 600 and (cls == 0).sum() == 4000 - 650  # per-class caps
    assert np.isfinite(xyz).all() and np.abs(xyz[cls == 2][:, 0]).max() <= 0.9 + 0.006
    m = v["memory"]
    assert m["n"] == [241, 241] and m["lo"] == [-6.0, -6.0] and m["res"] == 0.05 and m["epoch"] == 0
    assert m["order"] == "C, first index x"
    sup = np.frombuffer(base64.b64decode(m["support_b64"]), dtype=np.uint8).reshape(241, 241)
    ceil = np.frombuffer(base64.b64decode(m["ceiling_b64"]), dtype=np.uint8).reshape(241, 241)
    near = np.frombuffer(base64.b64decode(m["near_b64"]), dtype=np.uint8).reshape(241, 241)
    assert sup[0, 0] == 255 and sup[100, 100] == 0 and sup[130, 120] == 31 and sup[131, 120] == 254
    assert ceil[110, 120] == 60 and ceil[0, 0] == 255 and near[121, 121] == 1 and near.sum() == 1
    # cell (i, j) is centred at lo + (i, j) res, first index x: the 0.62 m cell is at x = 0.5, y = 0.0
    assert np.allclose(np.array(m["lo"]) + np.array([130, 120]) * m["res"], [0.5, 0.0])
    again = strict_loads(n.view_reply({"cmd": "view", "cloud_after": 1, "mem_after": m["seq"]}))
    assert again["cloud"] is None and again["memory"] is None
    few = strict_loads(n.view_reply({"cmd": "view", "max_points": 500}))
    assert few["cloud"]["n"] == 500 and few["memory"]["seq"] == m["seq"]


def test_memory_part_changes_only_when_the_memory_does():
    n = fake_node()
    n.view_memory(100.0)
    seq = n.view_mem["seq"]
    n.view_memory(100.5)  # within VIEW_MEM_PERIOD_S: not even looked at
    n.memory.support[50, 50] = 0.3
    assert n.view_mem["seq"] == seq
    n.view_memory(101.2)
    assert n.view_mem["seq"] == seq + 1
    n.view_memory(102.4)  # unchanged: same seq
    assert n.view_mem["seq"] == seq + 1
    n.memory_epoch += 1  # clear_map: a new grid
    n.view_memory(103.6)
    v = strict_loads(n.view_reply({"cmd": "view", "mem_after": seq}))
    assert v["memory"]["seq"] == seq + 2 and v["memory"]["epoch"] == 1


def test_view_without_a_plan_or_state_yet():
    n = fake_node()
    n.active, n.view_state, n.status, n.pose_hist = None, None, {"phase": "BOOT"}, ()
    del n.route
    v = strict_loads(n.view_reply({"cmd": "view"}))
    assert v["plan"] is None and v["qpos"] is None and v["route"] is None and v["cloud"] is None
    assert v["odom"]["mode"] == "leg" and v["pose_hist"] == []
    assert strict_loads(n.view_reply({"cmd": "view", "max_points": "many"}))["ok"] is False


def test_odometry_health_per_mode():
    n = fake_node()
    now = time.monotonic()
    n.fused = SimpleNamespace(filter=SimpleNamespace(bad_since=now - 0.6, discontinuity=None, lio_lost=None),
                              speed_alarm=None, lost=None, t_msg=now - 0.05)
    n.args.odom = "fused"
    o = n.odom_state(now)
    assert o["doubt"] and o["lost"] is None and o["lio_age_s"] == pytest.approx(0.05) and o["watchdog"] == "ok"
    n.fused.lost, n.fused.filter.lio_lost = now, now
    assert n.odom_state(now)["lost"] == "LIO refused by the leg / IMU filter"
    n.fused.filter.discontinuity = now
    assert n.odom_state(now)["lost"] == "LIO jump"
    del n.fused
    n.args.odom = "zmq"
    n.watchdog = N.OdomWatchdog()
    o = n.odom_state(now, age_odo=0.03)
    assert o["lio_age_s"] == 0.03 and o["watchdog"] == "ok" and not o["doubt"]
    n.watchdog.bad_since = now - 0.2
    o = n.odom_state(now, age_odo=0.03)
    assert o["doubt"] and o["watchdog"].startswith("LIO and leg speeds disagree")
    del n.watchdog
    assert n.odom_state(now)["watchdog"] == "off"
    n.args.odom = "dds"
    assert n.odom_state(now) == {"mode": "dds", "lio_age_s": None, "doubt": False, "lost": None, "watchdog": None}


def test_cloud_part_clips_and_keeps_classes_apart():
    far = np.array([[400.0, 0.0, 0.0], [1.0, 2.0, 3.0], [np.inf, 0.0, 0.0]])
    c = N.pack_cloud(7, 1.0, {0: far}, origin=np.zeros(3))
    xyz, cls = decode_cloud(c)
    assert c["n"] == 2 and xyz[0, 0] == pytest.approx(327.67) and np.allclose(xyz[1], [1, 2, 3]) and (cls == 0).all()
    empty = N.pack_cloud(8, 1.0, {}, origin=None)
    assert empty["n"] == 0 and empty["origin"] == [0.0, 0.0, 0.0]


def test_sensor_frame_keeps_what_it_dropped_for_display():
    """M9: the rear-blind and self-filtered returns stay out of the memory but reach the view (class 1 and 2), and the
    fused result is the same with or without keep_dropped."""
    from g1deploy.perception.cloud_memory import BodyFilter
    from g1deploy.perception.unitree_lidar import CloudFromSensorFrame, Mount

    mount = Mount.load(Path(__file__).resolve().parents[1] / "robots/lab-g1/mount.json")
    p_t = np.array([[2.0, 0.0, -0.5], [-0.5, 0.0, -0.1], [-0.6, 0.3, -0.2], [-2.0, 0.0, -0.5], [0.08, 0.0, -0.25],
                    [0.3, 1.5, -0.5]])  # ahead, rear near x2, rear far, on the body, to the side
    p_s = (p_t - mount.pos_torso) @ mount.rot_torso
    q = np.zeros(36)
    q[2], q[3] = 0.78, 1.0
    plain = CloudFromSensorFrame(BodyFilter(margin_m=0.05), mount, rear_blind_deg=180.0, rear_blind_range_m=1.0)
    pw0, _ = plain(p_s, np.zeros(len(p_s)), 100.0, lambda t: q)
    conv = CloudFromSensorFrame(BodyFilter(margin_m=0.05), mount, rear_blind_deg=180.0, rear_blind_range_m=1.0)
    conv.keep_dropped = True
    pw, org = conv(p_s, np.zeros(len(p_s)), 100.0, lambda t: q)
    selfp, rear = conv.last_dropped
    assert np.allclose(pw, pw0) and len(pw) == 3 and len(rear) == 2 and len(selfp) == conv.stats["self_removed"] == 1
    assert len(pw) + len(selfp) + len(rear) == len(p_t)
    # registered with the same pose as the fused points: torso frame (-0.5, 0, -0.1) is 0.5 m behind the pelvis area
    assert np.all(rear[:, 0] < 0) and conv.last_origin is not None and math.isfinite(float(conv.last_origin[2]))
