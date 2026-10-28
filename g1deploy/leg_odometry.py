"""Contact-aided leg odometry for the G1: pelvis position from joint encoders + pelvis IMU attitude.

Needs nothing beyond rt/lowstate (no lidar odometry, no ROS 2), so it is the dependency-free odometry
for first hardware runs and the fallback when LIO diverges. Per sample:
  1. FK of both feet relative to the pelvis, in world-aligned axes (IMU attitude, MuJoCo model);
  2. stance foot = the lower one (hysteresis `switch_m`), its sole contact point is fixed in the world;
  3. pelvis = stance anchor - foot offset; on a stance switch the new foot is anchored where it is now;
  4. the floor is z = 0 under the stance sole, which also fixes the pelvis height the planner encodes.
Yaw comes from the IMU (it drifts on the robot; `yaw_drift_rad_s` simulates that here). Slip, double
support and uneven floors are not modelled; measured accuracy is in docs/RUNBOOK.md. Contact threshold 1 cm (was
3 cm): travel / FAST-LIO travel on three lab-g1 walks (2026-10-27) 0.85-0.99 (mean 0.93) vs 0.62-0.95 (0.81), on sim
walks / truth 0.82-0.84 vs 0.72-0.77; no extra drift standing.
"""

from __future__ import annotations

import numpy as np


def _quat_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class LegOdometry:
    def __init__(self, xml=None, contact_m=0.01, scale=1.0, rotate_deg=0.0):
        import mujoco

        from sim.scene import G1_XML

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml or G1_XML))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.foot_geoms, self.foot_bodies = [], []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            self.foot_bodies.append(b)
            self.foot_geoms.append([g for g in range(m.ngeom) if m.geom_bodyid[g] == b
                                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE])
        self.radius = float(m.geom_size[self.foot_geoms[0][0]][0])
        self.contact = contact_m
        # rehearsal error injection: each step's displacement scaled and turned (2026-10-27 robot: ~0.8 and -5..-11 deg)
        a = np.radians(rotate_deg)
        self.err = scale * np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        self.stance, self.anchor, self.p = None, None, None

    def feet(self, quat_wxyz, q_mj):
        """Per foot, relative to the pelvis in world-aligned axes: (ankle body origin, lowest sole point).

        The ankle origin is the anchor (it stays put on a planted foot while the foot rolls heel-to-toe; the
        lowest sole point jumps between the heel and toe spheres); the sole point decides stance and floor height."""
        d = self.data
        d.qpos[:] = 0
        d.qpos[3:7] = quat_wxyz
        d.qpos[7:] = q_mj
        self.mujoco.mj_kinematics(self.model, d)
        out = []
        for body, geoms in zip(self.foot_bodies, self.foot_geoms):
            pts = d.geom_xpos[geoms].copy()
            pts[:, 2] -= self.radius
            out.append((d.xpos[body].copy(), pts[np.argmin(pts[:, 2])]))
        return out

    def reset(self, pelvis_xy, quat_wxyz, q_mj):
        f = self.feet(quat_wxyz, q_mj)
        self.stance = int(f[1][1][2] < f[0][1][2])
        self.p = np.array([pelvis_xy[0], pelvis_xy[1], -f[self.stance][1][2]])
        self.prev = f
        self.dp = np.zeros(3)
        return self.p.copy()

    def update(self, quat_wxyz, q_mj):
        """Velocity-consistent stance: each step the planted foot is the one (within `contact_m` of the floor)
        whose implied pelvis displacement best matches the previous step's; integrate that displacement.
        (Lowest-foot switching with hysteresis kept the anchor on a lifting toe-off foot: 18% under-estimate.)"""
        f = self.feet(quat_wxyz, q_mj)
        lowest = min(f[0][1][2], f[1][1][2])
        best, best_err = None, np.inf
        for i in (0, 1):
            if f[i][1][2] > lowest + self.contact:
                continue
            dp = -(f[i][0] - self.prev[i][0])
            err = np.linalg.norm((dp - self.dp)[:2])
            if err < best_err:
                best, best_err = i, err
        dp = -(f[best][0] - self.prev[best][0])
        self.stance, self.dp, self.prev = best, dp, f
        self.p = self.p + self.err @ dp
        self.p[2] = -f[self.stance][1][2]  # the planted sole is on the floor (z = 0)
        return self.p.copy()


class LegOdomEstimate:
    """Drop-in for flow's `StateEstimate` in the lock-step evaluator: the planner's pose comes from leg
    odometry (+ an IMU yaw drift), joints and roll/pitch stay exact; `.err` = (dx, dy, dyaw) for the memory."""

    def __init__(self, cfg, rng=None):
        self.cfg = dict(cfg) or {"leg": True}
        self.odo = LegOdometry()
        self.err = np.zeros(3)
        self.yaw_err = 0.0
        self.rng = rng or np.random.default_rng(0)

    def __call__(self, q, dt):
        q = np.asarray(q, dtype=np.float64)
        self.yaw_err += float(self.cfg.get("yaw_drift_rad_s", 0.0)) * dt
        h = 0.5 * self.yaw_err
        dq = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
        w1, x1, y1, z1 = dq
        w2, x2, y2, z2 = q[3:7]
        quat = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
        if self.odo.p is None:
            p = self.odo.reset(q[:2], quat, q[7:])
        else:
            p = self.odo.update(quat, q[7:])
        e = q.copy()
        e[:3] = p
        e[3:7] = quat
        self.err = np.array([p[0] - q[0], p[1] - q[1], self.yaw_err])
        return e.astype(np.float32)
