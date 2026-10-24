#!/usr/bin/env python3
"""Does the guards' self mask ever act on real lidar? Rebuild the node's memory from a recorded robot session and run
g1deploy.guards.self_mask on it, at the deployed self-filter margin and at the leaky 3 cm one.

The node path, offline: rec/lowstate -> leg odometry (g1deploy/leg_odometry.py, origin under the first pelvis
sample, as `--odom leg`) -> the DdsState ring (tools/debug_plan.py RecordedState) -> every recorded raw MID-360 frame
through CloudFromSensorFrame (deskew, mount calibration, mesh self filter) into CloudMemory with the bundle's sensor
config -> guards.memory_view(near_only=True) -> guards.self_mask at the last measured pose.

Per margin it reports: occupied cells, near-only cells (evidence only ever seen from closer than `near_m`), the
cells inside the mask band around the planner's capsules, the mask's candidates (band & near-only & occupied), what
it hides or refuses, and where the candidates are relative to the pelvis.

usage: guard_mask_real.py SESSION [SESSION ...] [--margins 0.05 0.03] [--mount-calib robots/lab-g1/mount.json]
                          [--out runs/profile/guard_mask_real.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import bundle  # noqa: E402
from g1deploy.recording import open_session  # noqa: E402


def leg_state(rec, every=5):
    """RecordedState with the pelvis from leg odometry (every `every`-th lowstate sample, interpolated between)."""
    from g1deploy.leg_odometry import LegOdometry
    from tools.debug_plan import RecordedState

    low = rec.stream("lowstate", keys=["q", "dq", "imu_quat"])
    t, q, imu = low["t"], low["q"][:, :29].astype(np.float64), low["imu_quat"].astype(np.float64)
    odo, idx, pos = LegOdometry(), np.arange(0, len(t), every), []
    for i in idx:
        pos.append(odo.reset((0.0, 0.0), imu[i], q[i]) if odo.p is None else odo.update(imu[i], q[i]))
    pos = np.array(pos)
    p = np.stack([np.interp(t, t[idx], pos[:, k]) for k in range(3)], 1)
    return RecordedState(t, np.c_[p, imu, q], low["dq"][:, :29].astype(np.float64)), p


def rebuild(rec, flow, pdict, margin, mount):
    from g1deploy.perception.cloud_memory import CloudMemory
    from g1deploy.perception.unitree_lidar import CloudFromSensorFrame

    state, p = leg_state(rec)
    sensor = dict(flow.planner.PlannerConfig(**pdict).sensor, self_margin_m=margin)
    memory = CloudMemory(np.asarray(p[0, :2]), rng=np.random.default_rng(0), **sensor)
    memory.latency = 0.0
    conv = CloudFromSensorFrame(memory.body_filter, mount, transport_s=0.01)
    used, t_last = 0, None
    for t_f, pts, t_s in rec.lidar_frames():
        state.until(t_f + 0.05)
        if state.qpos_at(t_f) is None:
            continue
        p_w, org = conv(pts, t_s, t_f, state.qpos_at)
        memory.add_cloud(0.0, p_w, org, None)
        memory.step(t_f)
        used, t_last = used + 1, t_f
    return memory, state, used, t_last, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--margins", type=float, nargs="+", default=[0.05, 0.03])
    ap.add_argument("--mount-calib", default=str(ROOT / "robots/lab-g1/mount.json"))
    ap.add_argument("--body", default="dex3")
    ap.add_argument("--out", default=str(ROOT / "runs/profile/guard_mask_real.json"))
    args = ap.parse_args(argv)
    os.environ.setdefault("G1_BODY_MODEL", args.body)
    import torch
    from scipy.ndimage import label

    from g1deploy import guards as G
    from g1deploy.fastplan import make_planner
    from g1deploy.perception.unitree_lidar import Mount

    flow = bundle.load_flow()
    pdict = dict(bundle.student_config()["planner"], device="cuda" if torch.cuda.is_available() else "cpu")
    planner = make_planner(flow.planner.PlannerConfig(**pdict), fast=False)  # capsules + FK only
    cfg = G.preset("deploy")
    mount = Mount.load(args.mount_calib) if args.mount_calib else Mount.nominal()
    results = []
    for s in args.sessions:
        rec = open_session(s)
        for margin in args.margins:
            memory, state, used, t_last, p = rebuild(rec, flow, pdict, margin, mount)
            view = G.memory_view(memory, near_only=True)
            q = state.qpos_at(t_last)
            occupied = (np.nan_to_num(view.support, nan=0.0) > 0.05) | (np.isfinite(view.ceiling) & (view.ceiling < 1e2))
            band = G.body_cells(planner, q, view, cfg.self_mask_band_m)
            cand = band & view.near_only & occupied
            lab, _ = label(occupied, structure=np.ones((3, 3), dtype=bool))
            _, info = G.self_mask(view, planner, q, cfg)
            ij = np.argwhere(cand)
            rel = (view.lo + ij * view.res) - q[:2]
            row = {"session": str(s), "self_margin_m": margin, "frames": used,
                   "pelvis_xyz": np.round(q[:3], 3).tolist(),
                   "leg_odom_drift_m": round(float(np.linalg.norm(p[-1, :2] - p[0, :2])), 3),
                   "memory": {k: int(v) for k, v in memory.stats.items()} if hasattr(memory, "stats") else None,
                   "occupied_cells": int(occupied.sum()), "near_only_cells": int((view.near_only & occupied).sum()),
                   "band_cells_occupied": int((band & occupied).sum()), "mask_candidates": int(cand.sum()),
                   "candidate_components": int(len(np.unique(lab[cand]))) if cand.any() else 0,
                   "self_masked_cells": int(info.get("self_masked_cells", 0)),
                   "self_mask_refused": info.get("self_mask_refused"),
                   "candidates_rel_xy_m": np.round(rel[:12], 2).tolist(),
                   "candidate_support_m": np.round(np.nan_to_num(view.support[cand], nan=-1)[:12], 2).tolist(),
                   "candidate_ceiling_m": np.round(np.where(np.isfinite(view.ceiling[cand]), view.ceiling[cand], -1)[:12],
                                                   2).tolist()}
            results.append(row)
            print(json.dumps({k: row[k] for k in ("session", "self_margin_m", "frames", "occupied_cells",
                                                   "near_only_cells", "band_cells_occupied", "mask_candidates",
                                                   "self_masked_cells", "self_mask_refused", "candidates_rel_xy_m")}),
                  flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"guard_config": cfg.describe() if hasattr(cfg, "describe") else str(cfg),
                               "band_m": cfg.self_mask_band_m, "max_cells": cfg.self_mask_max_cells,
                               "mount": args.mount_calib, "results": results}, indent=1, default=str) + "\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
