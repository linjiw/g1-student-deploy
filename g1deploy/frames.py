"""The frames of a deployment and the poses between them (the hub computes every transform; clients only apply them).

Poses are [x, y, yaw] (m, rad); R(a) rotates counter-clockwise; "A_from_B" is the pose of B's origin in A, so a point
maps as p_A = R(yaw) p_B + [x, y] and poses chain as A_from_C = compose(A_from_B, B_from_C).

- memory_S   a scan's memory grid frame (memory.npz lo/res). The robot stood at start_S = [start_xy, start_yaw] (leg
             odometry: IMU yaw). twin.json and studio_twin.json are in memory_S.
- scene      the scan robot frame: start_S at [0, 0, 0], x ahead. course/task.json is in scene.
             scene_from_memory = inverse(start_S).
- odom_K     the node odometry frame of console K (leg: node-start origin with IMU yaw; zmq | fused: the LIO world;
             dds: the sim course frame).
- A start check (registration.start_check) returns theta, t with p_robot_now = R(theta) p_scene + t, measured with the
  node at P_c = [xc, yc, yawc] (the live map's robot pose in odom_K). Then scene_from_odom = inverse(P_c o [t, theta]):
  p_scene = R(-theta) (R(-yawc) (p_odom - [xc, yc]) - t); yaw_scene = yaw_odom - yawc - theta.
- A course reposed by a start offset (registration.repose_task with robot_from_scene = inverse(offset)): the robot
  starts at the course origin, which sits at `offset` in scene, so scene_from_course = offset.
"""

from __future__ import annotations

import math

import numpy as np

# The start offset every 2026-10-27 walk showed between its scan and GO (16-31 deg right, 0.16-0.26 m): the robot
# stands 0.2 m along a heading 20 deg right of the scan's, turned 20 deg right. The rehearsal set's third run and
# practice sessions start there.
DEFAULT_OFFSET_DEG, DEFAULT_OFFSET_M = -20.0, 0.2


def wrap(a):
    """An angle in (-pi, pi]."""
    return float(math.atan2(math.sin(a), math.cos(a)))


def pose(p):
    p = [float(v) for v in p]
    if len(p) != 3:
        raise ValueError("a pose is [x, y, yaw]")
    return [p[0], p[1], wrap(p[2])]


def compose(a_from_b, b_from_c):
    """A_from_C."""
    ax, ay, at = pose(a_from_b)
    bx, by, bt = pose(b_from_c)
    c, s = math.cos(at), math.sin(at)
    return [ax + c * bx - s * by, ay + s * bx + c * by, wrap(at + bt)]


def inverse(a_from_b):
    """B_from_A."""
    x, y, t = pose(a_from_b)
    c, s = math.cos(t), math.sin(t)
    return [-(c * x + s * y), -(-s * x + c * y), wrap(-t)]


def apply(a_from_b, pts):
    """Points [N, 2] (or one [2]) in B -> in A."""
    x, y, t = pose(a_from_b)
    p = np.asarray(pts, dtype=float)
    one = p.ndim == 1
    p = p.reshape(-1, 2)
    c, s = math.cos(t), math.sin(t)
    out = np.c_[c * p[:, 0] - s * p[:, 1] + x, s * p[:, 0] + c * p[:, 1] + y]
    return out[0] if one else out


def apply_pose(a_from_b, p_b):
    """A pose [x, y, yaw] in B -> in A (the same as compose)."""
    return compose(a_from_b, p_b)


def scene_from_memory(start_s):
    return inverse(start_s)


def robot_from_scene(theta, t):
    """The start check's transform as a pose: p_robot_now = R(theta) p_scene + t."""
    return pose([t[0], t[1], theta])


def scene_from_odom(theta, t, p_c):
    """The scene in a console's odometry frame, from a start check (theta, t) taken with the node at P_c (odom)."""
    return inverse(compose(pose(p_c), robot_from_scene(theta, t)))


def offset_pose(deg=DEFAULT_OFFSET_DEG, m=DEFAULT_OFFSET_M):
    """The start offset as the robot's pose in scene: `m` metres along a heading `deg` from the scan's, turned `deg`."""
    a = math.radians(deg)
    return pose([m * math.cos(a), m * math.sin(a), a])


def offset_envelope(offset):
    """How far the robot may stand from the scan pose and stay inside what a start-offset rehearsal covered:
    (moved_m, turned_deg)."""
    x, y, t = pose(offset)
    return math.hypot(x, y), abs(math.degrees(t))


def crosses(path, a, b):
    """Does the polyline `path` [N, 2] cross the segment a-b (a passage gate, e.g. between two towers' inner faces)?
    A path that reaches the gate's line inside the segment counts; one that only runs along that line does not."""
    p = np.asarray(path, dtype=float).reshape(-1, 2)
    a, b = np.asarray(a, dtype=float).reshape(2), np.asarray(b, dtype=float).reshape(2)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 2 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return False
    u, v = p[:-1], p[1:]

    def side(o, d, w):  # the sign of (d - o) x (w - o)
        return np.sign((d[..., 0] - o[..., 0]) * (w[..., 1] - o[..., 1]) - (d[..., 1] - o[..., 1]) * (w[..., 0] - o[..., 0]))
    s1, s2 = side(a, b, u), side(a, b, v)
    s3, s4 = side(u, v, a[None]), side(u, v, b[None])
    return bool(np.any((s1 * s2 <= 0) & (s1 != s2) & (s3 * s4 <= 0) & (s3 != s4)))


def yaw_of_quat(q_wxyz):
    w, x, y, z = (float(v) for v in q_wxyz)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotate_quat_z(q_wxyz, a):
    """A quaternion (w, x, y, z) turned by `a` about the world z axis (R_z(a) * q)."""
    w, x, y, z = (float(v) for v in q_wxyz)
    c, s = math.cos(a / 2), math.sin(a / 2)
    return [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]


def qpos_to(a_from_b, qpos):
    """36-D qpos rows (pelvis xyz, quat wxyz, joints) in B -> in A (pelvis position and heading; joints unchanged)."""
    q = np.array(qpos, dtype=float, copy=True)
    one = q.ndim == 1
    q = q.reshape(-1, q.shape[-1])
    _, _, t = pose(a_from_b)
    q[:, :2] = apply(a_from_b, q[:, :2])
    for i in range(len(q)):
        q[i, 3:7] = rotate_quat_z(q[i, 3:7], t)
    return q[0] if one else q
