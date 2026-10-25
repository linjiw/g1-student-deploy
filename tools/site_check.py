#!/usr/bin/env python3
"""Pre-walk site check: will the student see a walkable world from here to the goal? Read-only.

With the robot standing at the start, fills the node's own lidar memory (CloudMemory with the bundle's sensor config,
the calibrated mount, the self filter, the robot's lidar topic) for --seconds, then runs the frozen planner's own
geometry on it and reports what the planner will act on:

  boxes     Flow's memory_boxes (the planner's guide boxes): connected occupied cells become ONE box. Walls or furniture
            that join around the robot become a single box that contains it, and the planner then finds no safe
            candidate (simulation 2026-10-23: a closed room -> one 5.4 x 6.1 m box -> every plan crouched, 452 dropped).
  ceilings  overhead cells below 1.35 m within 0.8 m of the start / goal: node.py defers its stand there (the gantry
            frame, table tops 0.8-1.35 m, phantom self-returns)
  route     Flow's SensorRoute from start to goal on this memory: found or not, length, share through unseen cells

Verdict: NO-GO (a box contains start or goal, or no route), CHECK (merged giant box, ceiling near start/goal, mostly
unseen route), GO. Writes a PNG of support, boxes, route, start and goal.

usage: site_check.py (--robot NAME | --iface NIC) --goal AHEAD LEFT [--seconds 4] [--odom leg|dds] [--png out.png]
       (simulation: --iface lo --odom dds while `g1 sim` holds the robot standing)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import bundle  # noqa: E402
from g1deploy.node import DdsCloudInput, DdsState, LegOdomSource  # noqa: E402

ROBOT_RADIUS_M = 0.30
CEILING_BELOW_M, CEILING_NEAR_M = 1.35, 0.8   # node.py ceiling_near()
GIANT_BOX_M = 1.5                              # half extent: a merged structure, not one object


def inside_box(b, xy, margin=0.0):
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    d = np.asarray(xy, float) - np.asarray(b["center_xy"], float)
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    return bool(np.all(np.abs(local) <= np.asarray(b["half_xy"], float) + margin))


def low_ceiling_near(memory, xy, radius=CEILING_NEAR_M, below=CEILING_BELOW_M):
    c = memory.ceiling
    ij = np.argwhere(np.isfinite(c) & (c < below))
    if not len(ij):
        return []
    pts = memory.lo + ij * memory.res
    d = np.linalg.norm(pts - np.asarray(xy), axis=1)
    sel = d < radius
    return [(round(float(dd), 2), round(float(c[i, j]), 2)) for (i, j), dd in zip(ij[sel], d[sel])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", default=None)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--goal", type=float, nargs=2, required=True, metavar=("AHEAD", "LEFT"))
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--odom", choices=("leg", "dds"), default=None, help="leg (robot) | dds (simulator truth)")
    ap.add_argument("--mount-calib", default=None)
    ap.add_argument("--body", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--rear-blind-deg", type=float, default=None,
                    help="ignore this rear lidar sector, as the node does (robot profile key rear_blind_deg)")
    args = ap.parse_args(argv)
    prof = {}
    if args.robot:
        prof = json.loads((ROOT / "robots" / f"{args.robot}.json").read_text())
    iface = args.iface or prof.get("iface")
    if not iface:
        raise SystemExit("give --robot NAME or --iface NIC")
    odom = args.odom or ("dds" if iface == "lo" else "leg")
    mount_path = args.mount_calib or prof.get("mount_calib")
    if mount_path and not Path(mount_path).is_absolute():
        mount_path = str(ROOT / mount_path)
    os.environ.setdefault("G1_BODY_MODEL", args.body or prof.get("body", "dex3"))
    rear_blind = args.rear_blind_deg if args.rear_blind_deg is not None else float(prof.get("rear_blind_deg") or 0.0)

    flow = bundle.load_flow()
    cfg = bundle.student_config()
    pcfg = flow.planner.PlannerConfig(**dict(cfg["planner"], device="cpu"))
    route_cfg = cfg["route_variants"]["sensor"].get("route", {})
    from g1deploy.perception.cloud_memory import CloudMemory
    from g1deploy.perception.unitree_lidar import CloudFromSensorFrame, Mount

    state = DdsState(iface, 0, odom)
    if odom == "leg":
        LegOdomSource(state)
    cloud = DdsCloudInput()
    mount = Mount.load(mount_path) if mount_path else Mount.nominal()
    memory = conv = None
    t0, frames = time.monotonic(), 0
    while time.monotonic() - t0 < args.seconds or memory is None and time.monotonic() - t0 < args.seconds + 5:
        snap = state.snapshot()
        for c in cloud.poll():
            if snap is None:
                continue
            if memory is None:
                memory = CloudMemory(snap["qpos"][:2], rng=np.random.default_rng(0), **dict(pcfg.sensor))
                memory.latency = 0.0
                conv = CloudFromSensorFrame(memory.body_filter, mount, rear_blind_deg=rear_blind)
            pts, org = conv(c["points"], c["t_s"], c["wall"], state.qpos_at)
            memory.add_cloud(0.0, pts, org, None)
            memory.step(time.monotonic() - t0)
            frames += 1
        time.sleep(0.005)
    if memory is None:
        print(json.dumps({"verdict": "NO-GO", "reasons": ["no lidar frames on rt/utlidar/cloud_livox_mid360 "
                                                           "(g1 lidar start / lidar service)"]}, indent=1))
        sys.stdout.flush()
        os._exit(2)
    q = state.snapshot()["qpos"]
    start = np.asarray(q[:2], float)
    w, x, y, z = q[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    a, l = args.goal
    goal = start + np.array([math.cos(yaw) * a - math.sin(yaw) * l, math.sin(yaw) * a + math.cos(yaw) * l])

    boxes = flow.sensor_map.memory_boxes(memory, grow_overhead_m=pcfg.guide_grow_overhead_m)
    route = flow.route.SensorRoute(start, goal, memory, **route_cfg)
    route.refresh(start, 0.0, force=True)
    path = np.asarray(route.path) if getattr(route, "path", None) is not None else np.zeros((0, 2))
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else None
    unseen = None
    if len(path) > 1:
        ij = np.clip(np.round((path - memory.lo) / memory.res).astype(int), 0, memory.n - 1)
        unseen = round(float(np.mean(~np.isfinite(memory.support[ij[:, 0], ij[:, 1]]))), 2)

    reasons_nogo, reasons_check = [], []
    ground = [b for b in boxes if b["z_bottom"] < 0.3]
    for b in ground:
        if inside_box(b, start, ROBOT_RADIUS_M):
            reasons_nogo.append(f"start inside a {2 * b['half_xy'][0]:.1f} x {2 * b['half_xy'][1]:.1f} m obstacle box "
                                f"(centre {np.round(b['center_xy'], 2).tolist()}): connected walls/furniture around the robot")
        if inside_box(b, goal, ROBOT_RADIUS_M):
            reasons_nogo.append(f"goal inside an obstacle box (centre {np.round(b['center_xy'], 2).tolist()})")
        if max(b["half_xy"]) > GIANT_BOX_M and not inside_box(b, start, ROBOT_RADIUS_M):
            reasons_check.append(f"merged box {2 * b['half_xy'][0]:.1f} x {2 * b['half_xy'][1]:.1f} m at "
                                 f"{np.round(b['center_xy'], 2).tolist()}: the planner treats it as one solid block")
    if not getattr(route, "found", False):
        reasons_nogo.append("no route from start to goal on the robot's map")
    elif unseen is not None and unseen > 0.3:
        reasons_check.append(f"{int(100 * unseen)} % of the route crosses never-seen cells")
    for name, xy in (("start", start), ("goal", goal)):
        near = low_ceiling_near(memory, xy)
        if near:
            reasons_check.append(f"{len(near)} ceiling cells below {CEILING_BELOW_M} m within {CEILING_NEAR_M} m of the "
                                 f"{name} (closest {min(near)[0]} m at {min(near)[1]} m high): the node defers its stand there")
    verdict = "NO-GO" if reasons_nogo else "CHECK" if reasons_check else "GO"
    report = {"verdict": verdict, "reasons": reasons_nogo + reasons_check, "frames": frames,
              "start_xy": np.round(start, 3).tolist(), "start_yaw": round(yaw, 4), "goal_xy": np.round(goal, 3).tolist(),
              "goal_ahead_left": [a, l],
              "route": {"found": bool(getattr(route, "found", False)), "length_m": None if length is None else round(length, 2),
                        "unseen_fraction": unseen},
              "boxes": [{k: (np.round(v, 2).tolist() if isinstance(v, (list, np.ndarray)) else round(float(v), 2))
                         for k, v in b.items()} for b in boxes],
              "mount": mount_path or "nominal", "odom": odom}
    print(json.dumps(report, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        ext = [memory.lo[0], memory.lo[0] + memory.n * memory.res, memory.lo[1], memory.lo[1] + memory.n * memory.res]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(np.nan_to_num(memory.support, nan=-0.2).T, origin="lower", extent=ext, cmap="viridis", vmin=-0.2, vmax=1.5)
        cy, cx = np.nonzero(np.isfinite(memory.ceiling.T) & (memory.ceiling.T < CEILING_BELOW_M))
        ax.scatter(memory.lo[0] + cx * memory.res, memory.lo[1] + cy * memory.res, s=4, c="magenta", label="ceiling < 1.35 m")
        for b in boxes:
            c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
            hx, hy = b["half_xy"]
            corners = [np.asarray(b["center_xy"]) + np.array([c * u - s * v, s * u + c * v])
                       for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
            ax.add_patch(Polygon(corners, fill=False, ec="red" if b["z_bottom"] < 0.3 else "orange", lw=1.5))
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], "w-", lw=2, label="route")
        ax.plot(*start, "co", ms=10, label="start")
        ax.plot(*goal, "y*", ms=16, label="goal")
        ax.arrow(start[0], start[1], 0.5 * math.cos(yaw), 0.5 * math.sin(yaw), color="c", width=0.03)
        ax.set_xlim(start[0] - 5, start[0] + 5)
        ax.set_ylim(start[1] - 5, start[1] + 5)
        ax.set_title(f"site check: {verdict}")
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(args.png, dpi=110)
    sys.stdout.flush()
    os._exit({"GO": 0, "CHECK": 1, "NO-GO": 2}[verdict])  # no DDS teardown (the leg-odometry thread can crash it)


if __name__ == "__main__":
    main()
