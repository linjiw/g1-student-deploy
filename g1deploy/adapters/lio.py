"""LIO odometry (FAST-LIO2 / Point-LIO on the head MID-360) -> pelvis pose in a floor-aligned world.

The planner needs the pelvis pose at 50 Hz in a gravity-aligned frame whose floor is z = 0 (its
history encodes root height, its memory classifies returns by height). A LIO filter on the
MID-360 reports the pose of the sensor's IMU ("body") in its own start frame ("camera_init"),
which on the G1 is upside down and tilted, and the sensor rides on the torso, i.e. behind the
three waist joints the student bends while ducking.

Frames (4x4 homogeneous, T_a_b maps b-coordinates into a):
  world      gravity aligned, floor z = 0, origin under the start pelvis, x = start heading
  pelvis     G1 pelvis (floating base)
  lidar      MID-360 sensor frame, T_pelvis_lidar(q_waist) = FK(pelvis -> torso_link) · T_torso_lidar
  body       LIO body frame = sensor IMU, T_lidar_body (Livox: IMU at (0.011, 0.02329, -0.04412) in the
             lidar frame, axes aligned; FAST-LIO's mid360 config uses extrinsic_T = [-0.011, -0.02329, 0.04412])
  caminit    LIO world = body frame at LIO start

At start (robot standing still, gravity initialised):
  T_world_pelvis0 = (x=y=0, z = standing pelvis height from FK with the feet on the floor,
                     roll/pitch from the pelvis IMU, yaw = 0)
  T_world_caminit = T_world_pelvis0 · T_pelvis_lidar(q_waist0) · T_lidar_body
Then each LIO update T_caminit_body(t):
  T_world_pelvis(t) = T_world_caminit · T_caminit_body(t) · inv(T_lidar_body) · inv(T_pelvis_lidar(q_waist(t)))
and registered points p_caminit -> T_world_caminit · p.

The mount extrinsic (T_torso_lidar) is the largest uncertainty on hardware: the URDFs disagree
(docs/HARDWARE.md) — calibrate it, e.g. by levelling the floor plane in the first cloud.
"""

from __future__ import annotations

import numpy as np

from g1deploy import contract as C
from g1deploy.perception import mid360

# Livox MID-360 internal IMU position in the lidar frame (Livox spec; FAST-LIO mid360.yaml uses the negation
# as the lidar-in-IMU extrinsic). Axes aligned.
LIVOX_IMU_IN_LIDAR = np.array([0.011, 0.02329, -0.04412])


def hom(r=np.eye(3), p=np.zeros(3)):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p
    return t


def inv(t):
    r, p = t[:3, :3], t[:3, 3]
    return hom(r.T, -r.T @ p)


def quat_to_rot(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(r):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]


class G1Kinematics:
    """pelvis -> torso_link and standing pelvis height from the deploy MJCF (MuJoCo FK)."""

    def __init__(self, xml=None):
        import mujoco

        from sim.scene import G1_XML

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml or G1_XML))
        self.data = mujoco.MjData(self.model)
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                     for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

    def _fk(self, q_mj, base=None):
        d = self.data
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        if base is not None:
            d.qpos[:7] = base
        d.qpos[7:] = q_mj
        self.mujoco.mj_kinematics(self.model, d)
        return d

    def pelvis_torso(self, q_mj):
        d = self._fk(q_mj)
        return hom(d.xmat[self.torso].reshape(3, 3).copy(), d.xpos[self.torso].copy())

    def standing_pelvis_height(self, q_mj, pelvis_rot):
        """Pelvis height with the lowest foot contact sphere on the floor (sphere radius included)."""
        d = self._fk(q_mj, np.r_[0, 0, 0, rot_to_quat(pelvis_rot)])
        m = self.model
        z = [d.geom_xpos[g][2] - m.geom_size[g][0] for g in range(m.ngeom)
             if m.geom_bodyid[g] in self.feet and m.geom_type[g] == self.mujoco.mjtGeom.mjGEOM_SPHERE]
        return -min(z)


class LioFrames:
    def __init__(self, kin: G1Kinematics | None = None, t_torso_lidar=None, t_lidar_body=None):
        self.kin = kin or G1Kinematics()
        self.t_torso_lidar = hom(mid360.ROT_PELVIS, mid360.POS_TORSO) if t_torso_lidar is None else t_torso_lidar
        self.t_lidar_body = hom(np.eye(3), LIVOX_IMU_IN_LIDAR) if t_lidar_body is None else t_lidar_body
        self.t_world_caminit = None

    def t_pelvis_lidar(self, q_mj):
        return self.kin.pelvis_torso(q_mj) @ self.t_torso_lidar

    def initialise(self, pelvis_imu_quat_wxyz, q_mj, t_caminit_body0=None):
        """Call once, standing still, when the LIO filter has initialised gravity."""
        r = quat_to_rot(pelvis_imu_quat_wxyz)
        yaw = np.arctan2(r[1, 0], r[0, 0])
        cz, sz = np.cos(-yaw), np.sin(-yaw)
        r_level = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ r  # keep roll/pitch, zero the yaw
        t_world_pelvis0 = hom(r_level, np.array([0.0, 0.0, self.kin.standing_pelvis_height(q_mj, r_level)]))
        t0 = np.eye(4) if t_caminit_body0 is None else t_caminit_body0
        self.t_world_caminit = t_world_pelvis0 @ self.t_pelvis_lidar(q_mj) @ self.t_lidar_body @ inv(t0)
        return t_world_pelvis0

    def pelvis_pose(self, t_caminit_body, q_mj):
        t = self.t_world_caminit @ t_caminit_body @ inv(self.t_lidar_body) @ inv(self.t_pelvis_lidar(q_mj))
        return t[:3, 3].copy(), rot_to_quat(t[:3, :3])

    def points_world(self, points_caminit):
        p = np.asarray(points_caminit, dtype=float)
        return p @ self.t_world_caminit[:3, :3].T + self.t_world_caminit[:3, 3]

    def lidar_origin_world(self, t_caminit_body):
        return (self.t_world_caminit @ t_caminit_body @ inv(self.t_lidar_body))[:3, 3].copy()


def default_standing_q():
    return C.DEFAULT_MJ.copy()


def fix_time(stamp_s, now_mono=None, now_unix=None, max_age_s=0.3):
    """The monotonic time a LIO pose is valid at, from its header stamp (FAST-LIO: the scan's end, on the lidar clock,
    which the MID-360 keeps within a few ms of this PC's clock), so PoseFilter predicts it forward to now instead of
    treating a pose a scan + processing old as current (at 0.5 m/s, 50-100 ms of age put every walking frame 2.5-5 cm
    behind the robot). Returns (t_mono, age_s); an age outside [0, max_age_s] (clocks apart) counts as 0."""
    import time

    now_mono = time.monotonic() if now_mono is None else now_mono
    now_unix = time.time() if now_unix is None else now_unix
    age = now_unix - float(stamp_s)
    if not 0.0 <= age <= max_age_s:
        return now_mono, None
    return now_mono - age, age


class PoseFilter:
    """50 Hz pelvis pose from 10 Hz LIO fixes + high-rate pelvis IMU attitude.

    Position: alpha-beta filter on LIO pelvis positions (constant-velocity prediction between fixes),
    so the planner's finite-difference velocities stay smooth. Attitude: the pelvis IMU quaternion,
    re-yawed onto the LIO yaw (the IMU yaw drifts; LIO yaw does not); the yaw offset is blended at
    `yaw_gain` per fix.
    """

    def __init__(self, alpha=0.5, beta=0.1, yaw_gain=0.3):
        self.alpha, self.beta, self.yaw_gain = alpha, beta, yaw_gain
        self.p = self.v = None
        self.t = None
        self.yaw_offset = None

    @staticmethod
    def _yaw(r):
        return np.arctan2(r[1, 0], r[0, 0])

    def fix(self, t, pos, quat_wxyz, imu_quat_wxyz):
        pos = np.asarray(pos, dtype=float)
        if self.p is None:
            self.p, self.v, self.t = pos.copy(), np.zeros(3), t
        else:
            dt = max(t - self.t, 1e-3)
            pred = self.p + self.v * dt
            r = pos - pred
            self.p = pred + self.alpha * r
            self.v = self.v + self.beta * r / dt
            self.t = t
        off = self._yaw(quat_to_rot(quat_wxyz)) - self._yaw(quat_to_rot(imu_quat_wxyz))
        off = np.arctan2(np.sin(off), np.cos(off))
        if self.yaw_offset is None:
            self.yaw_offset = off
        else:
            d = np.arctan2(np.sin(off - self.yaw_offset), np.cos(off - self.yaw_offset))
            self.yaw_offset += self.yaw_gain * d

    def pose(self, t, imu_quat_wxyz):
        if self.p is None:
            return None
        p = self.p + self.v * (t - self.t)
        c, s = np.cos(self.yaw_offset), np.sin(self.yaw_offset)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return p, rot_to_quat(rz @ quat_to_rot(imu_quat_wxyz)), self.v.copy()
