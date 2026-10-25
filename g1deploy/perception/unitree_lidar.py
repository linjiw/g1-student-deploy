"""The G1's own MID-360 stream: Unitree DDS `rt/utlidar/cloud_livox_mid360` (+ `rt/utlidar/imu_livox_mid360`).

What the robot publishes (Unitree lidar-service doc; layout measured on a real G1, see docs/HARDWARE.md):
  type   sensor_msgs::msg::dds_::PointCloud2_, 10 Hz, frame_id "livox_frame", published by PC1's lidar_driver
  points RAW sensor frame of the upside-down MID-360, NOT deskewed; ~20k points per frame, ~55% (0,0,0) placeholders
  layout point_step 22: x y z intensity float32 @0/4/8/12, ring uint16 @16, time float32 @18 = ns from frame start
  stamp  header.stamp = frame start, PC1's clock (can be minutes off the Orin without chrony)
Fields are always read by name from `msg.fields`; the layout above is only what the encoder writes.

Node side (`CloudFromSensorFrame`): drop placeholders and the 0.1 m blind zone, time every point on the node's
clock (receive time - transport - (t_last - t)), and re-express each 5 ms slice with the node's own estimate
of the sensor pose at that time: odometry pelvis pose (DdsState ring) -> MuJoCo FK through the waist ->
torso_link -> the calibrated mount. That is deskew + registration without any external LIO process, and the
self filter runs per slice with the joint state of that instant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import mid360

TOPIC_CLOUD = "rt/utlidar/cloud_livox_mid360"
TOPIC_IMU = "rt/utlidar/imu_livox_mid360"
_PF = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}
LAYOUT = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7), ("ring", 16, 4), ("time", 18, 7)]
POINT_STEP = 22


def imu_type():
    """sensor_msgs::msg::dds_::Imu_ (absent from unitree_sdk2py; ROS 2 sensor_msgs/Imu layout)."""
    import cyclonedds.idl as idl
    import cyclonedds.idl.annotations as annotate
    import cyclonedds.idl.types as types
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Quaternion_, Vector3_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_

    @dataclass
    @annotate.final
    @annotate.autoid("sequential")
    class Imu_(idl.IdlStruct, typename="sensor_msgs.msg.dds_.Imu_"):
        header: Header_
        orientation: Quaternion_
        orientation_covariance: types.array[types.float64, 9]
        angular_velocity: Vector3_
        angular_velocity_covariance: types.array[types.float64, 9]
        linear_acceleration: Vector3_
        linear_acceleration_covariance: types.array[types.float64, 9]

    return Imu_


# ------------------------------------------------------------------------------------------- codec
def encode_cloud(points_sensor, t_ns, stamp_s, frame_id="livox_frame", placeholder_frac=0.55, rng=None):
    """PointCloud2_ in the real G1 layout (points in the raw sensor frame)."""
    from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_

    rng = rng or np.random.default_rng()
    p = np.asarray(points_sensor, dtype=np.float32).reshape(-1, 3)
    t = np.asarray(t_ns, dtype=np.float32)
    n_pad = int(round(len(p) * placeholder_frac / max(1e-9, 1 - placeholder_frac)))
    if n_pad:  # placeholders interleaved at random positions, as the driver emits them
        pos = rng.integers(0, len(p) + 1, n_pad)
        p = np.insert(p, pos, 0.0, axis=0)
        t = np.insert(t, pos, 0.0)
    n = len(p)
    rec = np.zeros(n, dtype=np.dtype({"names": ["x", "y", "z", "intensity", "ring", "time"],
                                      "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f4"],
                                      "offsets": [0, 4, 8, 12, 16, 18], "itemsize": POINT_STEP}))
    rec["x"], rec["y"], rec["z"] = p[:, 0], p[:, 1], p[:, 2]
    rec["intensity"] = np.where(np.linalg.norm(p, axis=1) > 0, 100.0, 0.0)
    rec["ring"] = (np.arange(n) % 4).astype(np.uint16)
    rec["time"] = t
    sec = int(np.floor(stamp_s))
    header = Header_(Time_(sec, int((stamp_s - sec) * 1e9)), frame_id)
    fields = [PointField_(name, off, dt, 1) for name, off, dt in LAYOUT]
    return PointCloud2_(header, 1, n, fields, False, POINT_STEP, POINT_STEP * n, rec.tobytes(), True)


def decode_cloud(msg, min_range_m=mid360.MIN_RANGE_M):
    """PointCloud2_ -> (points (N, 3) sensor frame, t_s (N,) seconds from frame start, stamp_s). Placeholders dropped."""
    raw = bytes(msg.data)
    step, n = int(msg.point_step), int(msg.width) * int(msg.height)
    buf = np.frombuffer(raw, dtype=np.uint8)[: step * n].reshape(n, step)
    fields = {f.name: (int(f.offset), int(f.datatype)) for f in msg.fields}

    def col(name):
        off, dt = fields[name]
        typ = np.dtype(_PF[dt]).newbyteorder(">" if msg.is_bigendian else "<")
        return buf[:, off:off + typ.itemsize].copy().view(typ).reshape(-1).astype(np.float64)

    pts = np.stack([col("x"), col("y"), col("z")], 1)
    tname = next((k for k in ("time", "timestamp", "offset_time", "t") if k in fields), None)
    t = col(tname) if tname else np.zeros(n)
    # time unit: ns (G1 driver) unless the span says otherwise (FAST-LIO style seconds / livox CustomMsg us)
    span = float(t.max() - t.min()) if n else 0.0
    t_s = t * (1e-9 if span > 1e3 else 1e-6 if span > 1.0 else 1.0)
    ok = np.isfinite(pts).all(1) & (np.linalg.norm(pts, axis=1) > min_range_m)
    stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
    return pts[ok], t_s[ok], stamp


# ------------------------------------------------------------------------------------- mount / FK
@dataclass
class Mount:
    """MID-360 pose on torso_link (sensor -> torso). Default = unitree_ros g1_29dof_rev_1_0 (inverted, 2.93 deg)."""
    pos_torso: np.ndarray
    rot_torso: np.ndarray

    @classmethod
    def nominal(cls):
        return cls(mid360.POS_TORSO.copy(), mid360.ROT_PELVIS.copy())

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(np.asarray(d["pos_torso"], dtype=float), mid360.rpy_matrix(tuple(d["rpy"])))

    def save(self, path, note=""):
        from scipy.spatial.transform import Rotation

        rpy = Rotation.from_matrix(self.rot_torso).as_euler("xyz")  # URDF fixed-axis roll, pitch, yaw
        Path(path).write_text(json.dumps({"pos_torso": self.pos_torso.tolist(), "rpy": rpy.tolist(), "note": note},
                                         indent=1) + "\n")


class TorsoFK:
    """torso_link world pose from a 36-D qpos (pelvis pose + 29 joints) via MuJoCo kinematics."""

    def __init__(self, body_filter):
        import mujoco

        self.bf, self.mujoco = body_filter, mujoco
        self.torso = mujoco.mj_name2id(body_filter.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    def __call__(self, qpos36):
        self.bf.capsules(qpos36)  # runs mj_kinematics on the filter's model with this qpos
        d = self.bf.data
        return d.xpos[self.torso].copy(), d.xmat[self.torso].reshape(3, 3).copy()


class CloudFromSensorFrame:
    """Raw sensor-frame frames -> world points + per-point origins, deskewed with the node's own odometry."""

    def __init__(self, body_filter, mount: Mount | None = None, transport_s=0.01, slice_s=0.005, rear_blind_deg=0.0,
                 rear_blind_range_m=0.0):
        self.bf, self.fk = body_filter, TorsoFK(body_filter)
        self.mount = mount or Mount.nominal()
        self.transport, self.slice = transport_s, slice_s
        # Rear blind sector: returns from the back `rear_blind_deg` of torso-frame azimuth are dropped before
        # anything is fused. The robot walks forward; behind it are the spotter, the safety rope and the gantry,
        # which the memory would otherwise keep as obstacles touching the robot (lab-g1 door runs, 2026-10-24).
        self.front_half_deg = 180.0 - 0.5 * float(rear_blind_deg)
        # > 0: only rear returns closer than this (horizontally, from the torso) are dropped: the rope and the spotter
        # stay out of the memory while the far rear stays observed (the frozen planner was trained seeing all around)
        self.rear_range = float(rear_blind_range_m or 0.0)
        self.stats = {"frames": 0, "points": 0, "self_removed": 0, "rear_blind": 0}

    def __call__(self, pts_s, t_s, recv_wall, qpos_at):
        """qpos_at(t_wall) -> 36-D qpos (odometry pelvis pose + joints) or None."""
        if len(pts_s) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        if self.front_half_deg < 180.0:
            p_t = pts_s @ self.mount.rot_torso.T + self.mount.pos_torso
            front = np.abs(np.degrees(np.arctan2(p_t[:, 1], p_t[:, 0]))) <= self.front_half_deg
            if self.rear_range > 0:
                front |= np.hypot(p_t[:, 0], p_t[:, 1]) >= self.rear_range
            self.stats["rear_blind"] += int((~front).sum())
            pts_s, t_s = pts_s[front], t_s[front]
            if len(pts_s) == 0:
                self.stats["frames"] += 1
                return np.zeros((0, 3)), np.zeros((0, 3))
        t_wall = recv_wall - self.transport - (t_s.max() - t_s)
        bins = np.floor((t_wall - t_wall.min()) / self.slice).astype(int)
        out_p, out_o = [], []
        for b in np.unique(bins):
            sel = bins == b
            q = qpos_at(float(t_wall[sel].mean()))
            if q is None:
                continue
            p_t, r_t = self.fk(q)
            r_s, o = r_t @ self.mount.rot_torso, p_t + r_t @ self.mount.pos_torso
            pw = pts_s[sel] @ r_s.T + o
            keep = self.bf.mask(pw, q, origins=o)
            self.stats["self_removed"] += int((~keep).sum())
            out_p.append(pw[keep])
            out_o.append(np.repeat(o[None], int(keep.sum()), 0))
        self.stats["frames"] += 1
        self.stats["points"] += len(pts_s)
        if not out_p:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.concatenate(out_p), np.concatenate(out_o)


# ------------------------------------------------------------------------------------- calibration
def calibrate_mount(frames, qpos, mount: Mount, body_filter, floor_band_m=0.15, iters=3):
    """Correct the mount so the floor seen standing still is level at z = 0.

    frames: list of raw sensor-frame point arrays; qpos: the standing 36-D qpos (pelvis pose from IMU attitude +
    FK standing height, joints). Fits the floor plane in the world (points near z = 0, robot removed), then
    rotates/shifts the mount about the sensor so that plane becomes z = 0. Yaw and xy are unobservable from the
    floor and are left as they are. Returns (new mount, report)."""
    from scipy.spatial.transform import Rotation

    fk = TorsoFK(body_filter)
    p_t, r_t = fk(qpos)
    pts_s = np.concatenate(frames)
    m = Mount(mount.pos_torso.copy(), mount.rot_torso.copy())
    report = []
    for _ in range(iters):
        r_s, o = r_t @ m.rot_torso, p_t + r_t @ m.pos_torso
        pw = pts_s @ r_s.T + o
        pw = pw[body_filter.mask(pw, qpos)]
        fl = pw[np.abs(pw[:, 2]) < floor_band_m]
        if len(fl) < 100:
            raise ValueError("too few floor returns: is the robot standing on open floor?")
        a = np.c_[fl[:, :2] - o[:2], np.ones(len(fl))]
        (gx, gy, c), *_ = np.linalg.lstsq(a, fl[:, 2], rcond=None)  # z = gx dx + gy dy + c (about the sensor xy)
        n = np.array([-gx, -gy, 1.0])
        n /= np.linalg.norm(n)
        report.append({"floor_z_at_sensor_m": float(c), "tilt_deg": float(np.degrees(np.arccos(n[2])))})
        # rotate the world so the fitted normal becomes +z, about the sensor origin; then lift by -c
        axis = np.cross(n, [0, 0, 1.0])
        ang = np.arcsin(min(1.0, np.linalg.norm(axis)))
        r_fix = Rotation.from_rotvec(axis / max(np.linalg.norm(axis), 1e-12) * ang).as_matrix() if ang > 1e-9 else np.eye(3)
        # new sensor->world rotation r_fix @ r_s, and the sensor origin moves down by c (floor up to 0)
        r_s_new = r_fix @ r_s
        o_new = o - np.array([0.0, 0.0, c])
        m = Mount(r_t.T @ (o_new - p_t), r_t.T @ r_s_new)
    return m, report
