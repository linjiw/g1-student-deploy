#!/usr/bin/env python3
"""Accuracy of leg odometry on recorded closed-loop trajectories (true joints + true attitude in, pelvis out).

usage: leg_odom_check.py runs/g3-base [--yaw-drift 0.0]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1deploy.leg_odometry import LegOdomEstimate  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--yaw-drift", type=float, default=0.0)
    args = ap.parse_args(argv)
    rows = []
    for f in sorted(glob.glob(f"{args.run_dir}/*/closed_loop.npz")):
        q = np.load(f)["realized_qpos"]
        if len(q) < 20:
            continue
        est = LegOdomEstimate({"yaw_drift_rad_s": args.yaw_drift})
        e = np.stack([est(x, 0.02) for x in q])
        dist = float(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1).sum())
        err = np.linalg.norm(e[:, :2] - q[:, :2], axis=1)
        vel_t = np.diff(q[:, :2], axis=0) / 0.02
        vel_e = np.diff(e[:, :2], axis=0) / 0.02
        rows.append(dict(ep=Path(f).parent.name, dist_m=dist, final_err_m=float(err[-1]), max_err_m=float(err.max()),
                         z_err_m=float(np.abs(e[:, 2] - q[:, 2]).mean()),
                         vel_err_rms=float(np.sqrt(((vel_e - vel_t) ** 2).sum(1).mean()))))
    a = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "ep"}
    rel = a["final_err_m"] / np.maximum(a["dist_m"], 0.5)
    print(json.dumps({"episodes": len(rows), "dist_m_median": float(np.median(a["dist_m"])),
                      "final_err_m_median": float(np.median(a["final_err_m"])), "final_err_m_p90": float(np.quantile(a["final_err_m"], .9)),
                      "drift_pct_median": float(100 * np.median(rel)), "drift_pct_p90": float(100 * np.quantile(rel, .9)),
                      "pelvis_z_err_m_median": float(np.median(a["z_err_m"])),
                      "xy_velocity_err_rms_median": float(np.median(a["vel_err_rms"]))}, indent=1))


if __name__ == "__main__":
    main()
