"""The frames of a deployment (g1deploy/frames.py): scan memory, scene, a console's odometry, a reposed course.

Non-identity on purpose: the scan start start_S = [0.3, -0.2, 1.2] (IMU yaw, as on lab-g1) and a start offset of
20 deg / 0.2 m between the scan and the start check."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from g1deploy import frames as F, registration as R, wire

from .test_registration import coded, scene

START_S = [0.3, -0.2, 1.2]
OFFSET = F.offset_pose(-20.0, 0.2)  # the robot at the start check, in scene
P_C = [1.7, -0.6, 0.4]  # where the node's odometry has it then (any odometry origin)


def test_pose_algebra():
    a, b = [0.3, -0.2, 1.2], [1.0, 0.5, -0.4]
    ab = F.compose(a, b)
    assert np.allclose(F.compose(F.inverse(a), ab), b)
    assert np.allclose(F.compose(a, F.inverse(a)), [0, 0, 0], atol=1e-12)
    p = np.array([[1.0, 2.0], [-0.5, 0.3]])
    assert np.allclose(F.apply(ab, p), F.apply(a, F.apply(b, p)))
    assert np.allclose(F.apply(F.inverse(a), F.apply(a, p)), p)
    assert F.wrap(3 * math.pi) == pytest.approx(math.pi) and F.offset_pose()[2] == pytest.approx(math.radians(-20))


def test_scene_from_memory_puts_the_scan_start_at_the_origin():
    s_from_m = F.scene_from_memory(START_S)
    assert np.allclose(F.compose(s_from_m, START_S), [0, 0, 0], atol=1e-12)
    goal_scene = np.array([2.5, -0.3])  # ahead / left of the robot at the scan
    goal_memory = F.apply(START_S, goal_scene)  # what site_check writes as goal_xy
    assert np.allclose(F.apply(s_from_m, goal_memory), goal_scene)
    assert not np.allclose(goal_memory, goal_scene)


def test_start_check_scene_from_odom_round_trips_goals():
    world_scene = scene()  # obstacle points in scene
    # the scan: obstacles in memory_S, the robot standing at START_S
    mem_pts = F.apply(START_S, world_scene)
    scan = dict(coded(mem_pts, START_S[:2], 5.0), start=list(START_S))
    # the live map: the same obstacles in odom; the robot (at OFFSET in scene) is at P_C in odom
    odom_from_scene = F.compose(P_C, F.inverse(OFFSET))
    live = dict(coded(F.apply(odom_from_scene, world_scene), P_C[:2]), ok=True, robot=list(P_C))
    c = R.start_check(scan, live, goal_scan=[2.5, -0.3])
    assert c["ok"] and c["P_c"] == pytest.approx(P_C)
    assert c["turned_deg"] == pytest.approx(-20.0, abs=1.0) and c["moved_m"] == pytest.approx(0.2, abs=0.06)
    s_from_o = c["scene_from_odom"]
    # the node's pose at the check lands on the offset in scene
    robot_scene = F.compose(s_from_o, P_C)
    assert np.allclose(robot_scene[:2], OFFSET[:2], atol=0.06) and robot_scene[2] == pytest.approx(OFFSET[2], abs=0.02)
    # the contract's formula, written out
    th, t = c["theta"], np.array(c["t"])
    xc, yc, yc_ = P_C
    p_odom = np.array([2.0, 1.0])
    rot = lambda a, v: np.array([math.cos(a) * v[0] - math.sin(a) * v[1], math.sin(a) * v[0] + math.cos(a) * v[1]])  # noqa: E731
    by_hand = rot(-th, rot(-yc_, p_odom - [xc, yc]) - t)
    assert np.allclose(F.apply(s_from_o, p_odom), by_hand)
    assert F.compose(s_from_o, [0, 0, 0.7])[2] == pytest.approx(F.wrap(0.7 - yc_ - th))
    # a goal picked in scene, sent in odom, is the same spot in the room
    goal_scene = np.array([2.5, -0.3])
    goal_odom = F.apply(F.inverse(s_from_o), goal_scene)
    assert np.allclose(goal_odom, F.apply(odom_from_scene, goal_scene), atol=0.08)
    # and the start check's own goal_now (robot frame now) agrees
    assert np.allclose(F.apply(F.inverse(P_C), goal_odom), c["goal_now"], atol=0.02)


def test_reposed_course_back_to_scene():
    task = {"scene_boxes_world": [{"center_xy": [1.5, 0.4], "half_xy": [0.2, 0.1], "yaw": 0.3}], "goal_world_xy": [2.5, -0.3]}
    rfs = F.inverse(OFFSET)  # robot (course origin) from scene
    course = R.repose_task(task, rfs[2], rfs[:2])
    s_from_c = OFFSET
    assert np.allclose(F.apply(s_from_c, course["scene_boxes_world"][0]["center_xy"]), [1.5, 0.4])
    assert np.allclose(F.apply(s_from_c, course["goal_world_xy"]), [2.5, -0.3])
    assert F.wrap(course["scene_boxes_world"][0]["yaw"] + s_from_c[2]) == pytest.approx(0.3)


def test_qpos_to_turns_the_pelvis_and_keeps_the_joints():
    q = np.zeros(36)
    q[:3] = [1.0, 0.0, 0.78]
    q[3:7] = [math.cos(0.1), 0, 0, math.sin(0.1)]  # yaw 0.2
    q[7:] = np.arange(29) * 0.01
    out = F.qpos_to(START_S, q)
    assert np.allclose(out[:2], F.apply(START_S, [1.0, 0.0])) and out[2] == 0.78
    assert F.yaw_of_quat(out[3:7]) == pytest.approx(F.wrap(0.2 + 1.2)) and np.allclose(out[7:], q[7:])


def test_wire_is_strict_and_round_trips():
    s = wire.dumps({"a": float("nan"), "b": [np.float32(1.5), np.inf], "c": np.arange(3), "d": Path("/x")})
    assert json.loads(s) == {"a": None, "b": [1.5, None], "c": [0, 1, 2], "d": "/x"}
    sup = np.full((5, 4), np.nan)
    sup[1, 2], sup[3, 0] = 0.0, 7.0
    ceil = np.full((5, 4), np.nan)
    ceil[2, 2] = 1.226
    m = wire.encode_memory(sup, ceil, [-1.0, -2.0], 0.05, near=np.eye(5, 4, dtype=bool), seq=3)
    assert m["n"] == [5, 4] and m["seq"] == 3 and m["order"] == "C, first index x"
    s2, c2, n2 = wire.decode_memory(m)
    assert s2[1, 2] == 0.0 and s2[3, 0] == pytest.approx(5.08) and np.isnan(s2[0, 0]) and c2[2, 2] == pytest.approx(1.22)
    assert n2[1, 1] and not n2[1, 2]
    pts = np.array([[1.0, 2.0, 0.5], [np.nan, 0, 0], [1.2, 2.1, 0.4], [500.0, 0, 0]])
    cl = wire.encode_cloud(pts, [0, 0, 2, 1], origin=[1.0, 2.0, 0.0])
    p, k = wire.decode_cloud(cl)
    assert cl["n"] == 3 and np.allclose(p[:2], [[1.0, 2.0, 0.5], [1.2, 2.1, 0.4]]) and list(k) == [0, 2, 1]
    assert p[2, 0] == pytest.approx(1.0 + 327.0)  # clipped, never wrapped
    many = wire.encode_cloud(np.random.default_rng(0).normal(size=(10000, 3)), np.r_[np.zeros(9000), np.ones(1000)],
                             max_points=1000)
    _, k = wire.decode_cloud(many)
    assert many["n"] == 1000 and (k == 1).sum() == 100
