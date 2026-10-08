#!/usr/bin/env python3
"""Hardware ladder H2: calibrate the head MID-360 mount from a standing capture, and map its coverage.

The robot stands still on open, flat floor (gantry slack or in its fixed stand). Reads the G1's own lidar
topic (rt/utlidar/cloud_livox_mid360) and rt/lowstate for `--seconds`, then:
  1. coverage: histogram of returns over sensor azimuth x elevation; bins far below the median density are
     reported as blocked (head shell, cables) - the simulator can then be given that exact occlusion mask;
  2. mount: fits the floor plane through the nominal mount + FK (pelvis attitude from the IMU, pelvis height
     from FK with the feet on the floor) and corrects pitch/roll/height so the floor is level at z = 0;
     writes the result for `node.py --mount-calib`.
Works the same against the simulator (`--iface lo`, sim started with `--lidar-io dds`).

usage: calibrate_mount.py --iface <nic> --out mount.json [--seconds 3] [--png coverage.png]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1deploy.adapters.lio import G1Kinematics, quat_to_rot  # noqa: E402
from g1deploy.node import DdsCloudInput, DdsState  # noqa: E402
from g1deploy.perception.cloud_memory import BodyFilter  # noqa: E402
from g1deploy.perception.unitree_lidar import Mount, calibrate_mount  # noqa: E402


def coverage(points_sensor, az_bins=72, el_bins=20):
    p = np.asarray(points_sensor)
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    el = np.degrees(np.arctan2(p[:, 2], np.linalg.norm(p[:, :2], axis=1)))
    h, ae, ee = np.histogram2d(az, el, bins=[az_bins, el_bins], range=[[-180, 180], [-8, 53]])
    med = np.median(h[h > 0]) if (h > 0).any() else 0
    blocked = [(float(ae[i]), float(ae[i + 1]), float(ee[j]), float(ee[j + 1]))
               for i in range(az_bins) for j in range(el_bins) if h[i, j] < 0.1 * med]
    az_cols = [(float(ae[i]), float(ae[i + 1])) for i in range(az_bins) if h[i].sum() < 0.3 * np.median(h.sum(1))]
    return h, ae, ee, blocked, az_cols


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--start-mount", default=None, help="mount JSON to start from (default: URDF nominal)")
    ap.add_argument("--png", default=None)
    args = ap.parse_args(argv)
    state, cloud = DdsState(args.iface, odom="none"), DdsCloudInput()
    frames, t0 = [], time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        frames += [c["points"] for c in cloud.poll()]
        time.sleep(0.01)
    with state.lock:
        low = state.low
    if low is None or not frames:
        print(json.dumps({"error": "no lowstate or no lidar frames", "frames": len(frames)}))
        return 1
    q_mj, _, imu, _ = low
    r = quat_to_rot(imu)
    yaw = np.arctan2(r[1, 0], r[0, 0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    r_level = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ r
    from g1deploy.adapters.lio import rot_to_quat

    z = G1Kinematics().standing_pelvis_height(q_mj, r_level)
    qpos = np.r_[0.0, 0.0, z, rot_to_quat(r_level), q_mj]
    h, ae, ee, blocked, az_cols = coverage(np.concatenate(frames))
    start = Mount.load(args.start_mount) if args.start_mount else Mount.nominal()
    mount, report = calibrate_mount(frames, qpos, start, BodyFilter())
    mount.save(args.out, note=f"floor fit over {len(frames)} frames; before {report[0]}, after {report[-1]}")
    out = {"frames": len(frames), "points_per_frame": int(np.mean([len(f) for f in frames])),
           "standing_pelvis_z_m": float(z), "floor_before": report[0], "floor_after": report[-1],
           "blocked_azimuth_columns_deg": az_cols, "blocked_bins": len(blocked), "mount": json.loads(Path(args.out).read_text())}
    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.imshow(h.T, origin="lower", aspect="auto", extent=[-180, 180, ee[0], ee[-1]], cmap="viridis")
        ax.set_xlabel("sensor azimuth (deg, 0 = forward)")
        ax.set_ylabel("sensor elevation (deg)")
        fig.tight_layout()
        fig.savefig(args.png, dpi=120)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
