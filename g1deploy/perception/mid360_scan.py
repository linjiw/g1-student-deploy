"""Realistic Livox MID-360 for MuJoCo: real scan pattern, per-point timing, mount error, deskew modes.

The student was evaluated with an idealised sensor: a 2 deg x 2 deg grid (5,400 rays) cast at 10 Hz
from the pelvis-rigid mount, all rays at one instant. The physical MID-360 differs in ways that can
matter for the memory the planner reads:

- pattern   non-repetitive rosette (Livox), ~200k points/s, ~20k per 0.1 s frame; the table is the
            MuJoCo-LiDAR `mid360.npy` (800k (azimuth, elevation) rows, sensor frame, temporal order),
            consumed cyclically at `points_per_s`;
- timing    every point fires at its own time; the rays of each 5 ms physics step are cast from the
            sensor pose at that step, so walking motion, head sway and arm motion inside a frame are real;
- mount     the sensor rides on torso_link (behind the three waist joints the student bends). The rays
            are cast from the TRUE mount (`mount_error` = pitch/roll/yaw deg + xyz m on top of nominal);
            the robot converts returns with the ASSUMED nominal mount, so calibration error shows up as
            displaced points, exactly as on the robot;
- deskew    "ideal": each point re-expressed with the believed pose at its firing time (LIO deskew);
            "none": the whole frame re-expressed with the frame-end pose (raw driver cloud + one pose);
- noise     range noise, random dropout (dark/specular surfaces), 0.1 m blind zone, `range_m` cap.

Use: `scanner.substep(t)` after every physics step; `scanner.pop_frame()` returns a finished frame
(points in the believed world frame, per-point sensor origins, frame end time) for CloudMemory.add_cloud.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import mid360

TABLE = Path(__file__).resolve().parents[2] / "bundle/newton/allterrain/perception/data/mid360.npy"


def _rpy(r, p, y):
    return mid360.rpy_matrix((r, p, y))


class Mid360Scanner:
    def __init__(self, model, data, *, points_per_s=200_000, frame_hz=10.0, range_m=4.0, range_noise_m=0.02,
                 dropout=0.0, mount_error=None, deskew="ideal", rng=None, table=TABLE, geomgroup=(1, 0, 0, 0, 0, 0),
                 phase=0, body_filter=None, occlusion=None):
        import mujoco

        self.mujoco, self.model, self.data = mujoco, model, data
        tab = np.load(table).astype(np.float64)
        az, el = tab[:, 0], tab[:, 1]
        self.dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
        self.cursor = int(phase) % len(self.dirs)
        # Head-shell occlusion (one G1 unit measured a front blind sector of +-45 deg azimuth): rays in the
        # sensor-frame azimuth sector |az| <= az_deg and elevation band [el_lo_deg, el_hi_deg] never return.
        self.occluded = np.zeros(len(self.dirs), dtype=bool)
        if occlusion:
            a = np.degrees(np.arctan2(np.sin(az), np.cos(az)))
            e = np.degrees(el)
            self.occluded = (np.abs(a) <= float(occlusion.get("az_deg", 45.0))) \
                & (e >= float(occlusion.get("el_lo_deg", -90.0))) & (e <= float(occlusion.get("el_hi_deg", 90.0)))
        self.rate, self.frame_dt = float(points_per_s), 1.0 / frame_hz
        self.range_m, self.noise, self.dropout = range_m, range_noise_m, dropout
        self.deskew = deskew
        self.rng = rng or np.random.default_rng(0)
        self.group = np.asarray(geomgroup, dtype=np.uint8)
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        # nominal mount on torso_link (what the robot assumes) and the true one (what the rays use)
        self.r_assumed, self.p_assumed = mid360.ROT_PELVIS.copy(), mid360.POS_TORSO.copy()
        e = dict(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, x_m=0.0, y_m=0.0, z_m=0.0)
        e.update(mount_error or {})
        self.r_true = self.r_assumed @ _rpy(*np.radians([e["roll_deg"], e["pitch_deg"], e["yaw_deg"]]))
        self.p_true = self.p_assumed + np.array([e["x_m"], e["y_m"], e["z_m"]])
        self.t_last = None
        self.frame_end = None
        self.buf = []  # (points_believed, origins_believed, t)
        self.frames = []
        self.frame_times = []  # per-point firing times of each finished frame (same order as frames)
        self.believed_torso = None  # optional callable t -> (pos, rot) of torso_link as the robot believes it
        # Self filter per 5 ms slice with the joint state of that slice (Livox points carry their own time;
        # one pose per 0.1 s frame leaks swinging hands/shoulders into the memory as obstacles and ceilings).
        self.body_filter = body_filter
        self.self_removed = 0
        self.keep_raw = False  # also keep unfiltered sensor-frame frames (Unitree DDS format, see unitree_lidar.py)
        self.raw, self.raw_t, self.raw_frames = [], [], []

    def _torso_pose(self):
        d = self.data
        return d.xpos[self.torso].copy(), d.xmat[self.torso].reshape(3, 3).copy()

    def substep(self, t):
        """Cast the rays scheduled for (t_last, t] from the current state."""
        if self.t_last is None:
            self.t_last, self.frame_end = t, t + self.frame_dt
            return
        n = int(round((t - self.t_last) * self.rate))
        self.t_last = t
        if n <= 0:
            return
        idx = (self.cursor + np.arange(n)) % len(self.dirs)
        self.cursor = int((self.cursor + n) % len(self.dirs))
        p_torso, r_torso = self._torso_pose()
        o_true = p_torso + r_torso @ self.p_true
        d_world = self.dirs[idx] @ (r_torso @ self.r_true).T
        geom = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n)
        self.mujoco.mj_multiRay(self.model, self.data, o_true, d_world.reshape(-1), self.group, 1, -1, geom, dist,
                                None, n, float(self.range_m))
        hit = (geom >= 0) & (dist > mid360.MIN_RANGE_M) & (dist <= self.range_m) & ~self.occluded[idx]
        if self.dropout > 0:
            hit &= self.rng.random(n) >= self.dropout
        tt = dist[hit] + self.rng.normal(0.0, self.noise, int(hit.sum()))
        p_sensor = (self.dirs[idx][hit] * tt[:, None])  # sensor-frame returns
        if self.keep_raw:
            self.raw.append(p_sensor.copy())
            self.raw_t.append(t)
        if self.believed_torso is not None:
            p_torso, r_torso = self.believed_torso(t)
        if self.body_filter is not None and len(p_sensor):
            pw = p_sensor @ (r_torso @ self.r_assumed).T + (p_torso + r_torso @ self.p_assumed)
            keep = self.body_filter.mask(pw, self.data.qpos)
            self.self_removed += int((~keep).sum())
            p_sensor = p_sensor[keep]
        self.buf.append((p_sensor, p_torso.copy(), r_torso.copy(), t))
        if t + 1e-9 >= self.frame_end:
            self._close_frame(t)

    def _close_frame(self, t):
        if self.deskew == "none":  # every return re-expressed with the frame-end pose
            _, p_end, r_end, _ = self.buf[-1]
            pts = [ps @ (r_end @ self.r_assumed).T + (p_end + r_end @ self.p_assumed) for ps, *_ in self.buf]
            org = [np.repeat((p_end + r_end @ self.p_assumed)[None], len(ps), 0) for ps, *_ in self.buf]
        else:  # ideal deskew: each return with the pose at its own firing time
            pts = [ps @ (r @ self.r_assumed).T + (p + r @ self.p_assumed) for ps, p, r, _ in self.buf]
            org = [np.repeat((p + r @ self.p_assumed)[None], len(ps), 0) for ps, p, r, _ in self.buf]
        times = [np.full(len(ps), ts) for ps, _, _, ts in self.buf]
        if self.keep_raw:  # what the G1 driver publishes: sensor frame (true mount), time from frame start
            t0 = t - self.frame_dt
            self.raw_frames.append((np.concatenate(self.raw) if self.raw else np.zeros((0, 3)),
                                    np.concatenate([np.full(len(r), ts - t0) for r, ts in zip(self.raw, self.raw_t)])
                                    if self.raw else np.zeros(0), t0))
            self.raw, self.raw_t = [], []
        self.frames.append((np.concatenate(pts) if pts else np.zeros((0, 3)),
                            np.concatenate(org) if org else np.zeros((0, 3)), t))
        self.frame_times.append(np.concatenate(times) if times else np.zeros(0))
        self.buf = []
        self.frame_end += self.frame_dt

    def pop_raw_frames(self):
        out, self.raw_frames = self.raw_frames, []
        return out

    def pop_frames(self, with_times=False):
        out, times = self.frames, self.frame_times
        self.frames, self.frame_times = [], []
        return list(zip(out, times)) if with_times else out


def raw_frame_sensor(scanner_frame_points, origin_pose):
    """Helper for DDS publishing: world points -> sensor frame of `origin_pose` = (pos, rot)."""
    p, r = origin_pose
    return (np.asarray(scanner_frame_points) - p) @ r


def mount_error_from_str(s):
    """'pitch_deg=1.5,z_m=0.01' -> dict."""
    if not s:
        return None
    return {k: float(v) for k, v in (kv.split("=") for kv in s.split(","))}


def nominal_mount():
    return mid360.ROT_PELVIS, mid360.POS_TORSO, math.degrees(mid360.RPY[1])
