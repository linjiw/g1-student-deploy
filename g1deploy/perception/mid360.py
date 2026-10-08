"""Livox MID-360 on the G1 head: mount pose and the scan pattern the student was evaluated with.

Mirrors `allterrain/perception/visibility.py` (MID360_INVERTED, pelvis frame with the waist at
zero) without its torch dependency, so the simulator process can cast rays on its own. The
Isaac student memory used a 2 deg x 2 deg grid over 360 deg azimuth and the sensor-frame
elevation band [-7, 52] deg, range capped at 4 m (planner config `sensor.range_m`).

The real robot's mount is upside down; the URDFs disagree on its pitch/offset (see
docs/HARDWARE.md) — calibrate `pos_pelvis` / `rot_pelvis` on the robot before trusting maps.
"""

from __future__ import annotations

import math

import numpy as np

TORSO_IN_PELVIS = np.array([-0.0039635, 0.0, 0.044])
POS_TORSO = np.array([0.0002835, 0.00003, 0.428434])
RPY = (math.pi, 0.0511, 0.0)
ELEVATION_DEG = (-7.0, 52.0)
MIN_RANGE_M = 0.1


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


POS_PELVIS = TORSO_IN_PELVIS + POS_TORSO
ROT_PELVIS = rpy_matrix(RPY)


def ray_grid(az_step_deg=2.0, el_step_deg=2.0, elevation_deg=ELEVATION_DEG):
    """Unit ray directions in the sensor frame, as flow's RayCastMemory builds them."""
    az = np.deg2rad(np.arange(0, 360, az_step_deg))
    el = np.deg2rad(np.arange(elevation_deg[0], elevation_deg[1] + 1e-9, el_step_deg))
    a, e = np.meshgrid(az, el, indexing="ij")
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1).reshape(-1, 3)


def sensor_pose_world(pelvis_pos, pelvis_rot):
    """World origin and sensor->world rotation from the pelvis pose."""
    return np.asarray(pelvis_pos) + pelvis_rot @ POS_PELVIS, pelvis_rot @ ROT_PELVIS


def mj_cast(model, data, pelvis_pos, pelvis_rot, rays_sensor, range_m=4.0, range_noise_m=0.02, rng=None,
            geomgroup=(1, 0, 0, 0, 0, 0)):
    """Cast the pattern with mj_multiRay; returns world points (N, 3), origin, hit geom ids."""
    import mujoco

    origin, r_sensor = sensor_pose_world(pelvis_pos, pelvis_rot)
    dirs = rays_sensor @ r_sensor.T
    n = len(dirs)
    geom = np.zeros(n, dtype=np.int32)
    dist = np.zeros(n)
    mujoco.mj_multiRay(model, data, origin, dirs.reshape(-1).astype(np.float64), np.asarray(geomgroup, dtype=np.uint8),
                       1, -1, geom, dist, None, n, float(range_m))
    hit = (geom >= 0) & (dist > MIN_RANGE_M) & (dist <= range_m)
    rng = rng or np.random.default_rng()
    tt = dist[hit] + rng.normal(0.0, range_noise_m, int(hit.sum()))
    return origin + dirs[hit] * tt[:, None], origin, geom[hit]
