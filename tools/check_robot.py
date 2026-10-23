#!/usr/bin/env python3
"""Read-only robot / perception check (hardware ladder H1-H2). Sends no commands.

  H1  check_robot.py --iface <nic> [--seconds 5]
      rt/lowstate rate, mode_machine, pelvis IMU tilt, joints vs the model contract limits.
  H2  check_robot.py --iface <nic> --perception [--cloud tcp://<bridge>:5561] [--png out.png]
      + LIO bridge clouds -> CloudMemory: floor plane height/tilt from ground returns, returns left near
      the robot after the self filter, obstacle/ceiling cell counts, an image of both layers.
Also works against the simulator (`--iface lo`, cloud tcp://localhost:5560).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1deploy import bundle, contract as C  # noqa: E402
from g1deploy.node import CloudInput, DdsCloudInput, DdsState, LegOdomSource, OdomInput, quat_tilt  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", required=True)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--perception", action="store_true")
    ap.add_argument("--cloud", default="dds", help="'dds' = the G1's rt/utlidar topic; or a ZMQ endpoint")
    ap.add_argument("--odom", choices=("dds", "zmq", "leg"), default="dds")
    ap.add_argument("--mount-calib", default=None)
    ap.add_argument("--odom-connect", default="tcp://localhost:5561")
    ap.add_argument("--png", default=None)
    args = ap.parse_args(argv)
    state = DdsState(args.iface, args.domain, args.odom)
    odom = OdomInput(state, args.odom_connect) if args.odom == "zmq" else None
    if args.odom == "leg":
        LegOdomSource(state)
    contract = json.loads((bundle.ROOT / "config/sonic_model_contract.json").read_text())
    limits = np.asarray([j["range_rad"] for j in contract["joints"] if j["type"] != "free"])
    cloud = (DdsCloudInput() if args.cloud == "dds" else CloudInput(args.cloud)) if args.perception else None
    converter = None
    memory = None
    n_low, t0, last, tilts, out_of_range = 0, time.monotonic(), None, [], set()
    near_after = clouds = 0
    while time.monotonic() - t0 < args.seconds:
        if odom is not None:
            odom.poll()
        with state.lock:
            low, t_low = state.low, state.t_low
        if low is not None and t_low != last:
            n_low, last = n_low + 1, t_low
            q = low[0]
            tilts.append(quat_tilt(low[2]))
            bad = np.flatnonzero((q < limits[:, 0] - 0.05) | (q > limits[:, 1] + 0.05))
            out_of_range.update(C.MJ_JOINTS[i] for i in bad)
        if cloud is not None:
            snap = state.snapshot()
            for c in cloud.poll():
                if snap is None:
                    continue
                if memory is None:
                    from g1deploy.perception.cloud_memory import CloudMemory
                    from g1deploy.perception.unitree_lidar import CloudFromSensorFrame, Mount

                    memory = CloudMemory(snap["qpos"][:2], mounts=("mid360_inverted",), latency_s=0.0, range_m=4.0)
                    converter = CloudFromSensorFrame(memory.body_filter,
                                                     Mount.load(args.mount_calib) if args.mount_calib else Mount.nominal())
                if c.get("raw"):  # sensor frame -> world with our own pose (deskew + registration)
                    pts_all = c["points"]
                    t_wall = c["wall"] - converter.transport - (c["t_s"].max() - c["t_s"])
                    q_now = state.qpos_at(float(t_wall.mean())) if len(pts_all) else snap["qpos"]
                    p_t, r_t = converter.fk(q_now)
                    pts = pts_all @ (r_t @ converter.mount.rot_torso).T + (p_t + r_t @ converter.mount.pos_torso)
                    origin = p_t + r_t @ converter.mount.pos_torso
                else:
                    pts, origin = c["points"], c["origin"]
                memory.add_cloud(0.0, pts, origin, snap["qpos"])
                memory.step(time.monotonic() - t0)
                clouds += 1
                ground = pts[pts[:, 2] < 0.15]
                if len(ground) > 50:
                    A = np.c_[ground[:, :2], np.ones(len(ground))]
                    coef = np.linalg.lstsq(A, ground[:, 2], rcond=None)[0]
                    memory.stats["floor_fit"] = (float(coef[2]), float(np.degrees(np.arctan(np.hypot(*coef[:2])))))
                keep = memory.body_filter.mask(pts, snap["qpos"]) & (pts[:, 2] > 0.1)  # floor returns are legitimate
                rel = pts[keep, :2] - snap["qpos"][:2]
                near_after += int((np.linalg.norm(rel, axis=1) < 0.3).sum())
        time.sleep(0.0002)
    rate = n_low / args.seconds
    report = {"lowstate_hz": round(rate, 1), "tilt_max_rad": round(max(tilts), 3) if tilts else None,
              "joints_outside_limits": sorted(out_of_range),
              "mode_machine": None}
    if args.perception:
        report.update(clouds=clouds, returns_within_0p3m_after_filter=near_after,
                      floor_z_m_and_tilt_deg=memory.stats.get("floor_fit") if memory else None,
                      obstacle_cells=int((np.nan_to_num(memory.support) > 0.05).sum()) if memory else None,
                      ceiling_cells=int(np.isfinite(memory.ceiling).sum()) if memory else None)
        if args.png and memory is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(np.nan_to_num(memory.support, nan=-0.1).T, origin="lower", cmap="viridis")
            ax[0].set_title("support (m)")
            ax[1].imshow(np.nan_to_num(memory.ceiling, nan=0, posinf=0).T, origin="lower", cmap="magma")
            ax[1].set_title("ceiling (m)")
            fig.savefig(args.png, dpi=120)
    print(json.dumps(report, indent=1))
    ok = rate > 400 and not out_of_range
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # skip interpreter teardown: the leg-odometry thread still reading DDS segfaulted it (exit 245 after the report)
    os._exit(code)
