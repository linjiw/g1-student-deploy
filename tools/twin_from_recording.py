#!/usr/bin/env python3
"""A simulation twin of what the robot saw while standing still, from a recorded robot session: the node's lidar
memory re-fused from the last `--seconds` of raw MID-360 frames (leg odometry, the session's mount calibration), and
the scene built from the memory's own cells (guards.split_boxes: rectangles of each component's cells, so a thin
diagonal beam stays thin instead of becoming the frozen memory_boxes' bounding slab).

The course is in the frame of the robot at the end of the window (origin under the pelvis, facing +x), like every
course; boxes touching the robot's standing volume (harness, rope, gantry hook) are left out, as site_to_course does.

usage: twin_from_recording.py SESSION --goal AHEAD LEFT --name NAME [--seconds 20] [--until-s T]
       -> sim/courses/NAME/task.json + a top-view PNG next to it
"""

from __future__ import annotations

import argparse
import dataclasses
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--goal", type=float, nargs=2, required=True, metavar=("AHEAD", "LEFT"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=20.0, help="fuse the last S seconds of lidar (robot standing)")
    ap.add_argument("--until-s", type=float, default=None, help="end of the window, s from the recording start")
    ap.add_argument("--deadline", type=float, default=30.0)
    ap.add_argument("--body", default="dex3")
    args = ap.parse_args(argv)
    os.environ.setdefault("G1_BODY_MODEL", args.body)
    from g1deploy import guards as G
    from g1deploy.perception.cloud_memory import CloudMemory
    from g1deploy.perception.unitree_lidar import CloudFromSensorFrame, Mount
    from tools.guard_mask_real import leg_state

    flow = bundle.load_flow()
    pdict = bundle.student_config()["planner"]
    rec = open_session(args.session)
    if not rec.meta.get("lidar_fields"):  # still recording: the recorder writes the layout at the end
        from g1deploy.adapters.livox_to_utlidar import POINT_DT

        rec.meta["lidar_fields"] = [{"name": n, "offset": POINT_DT.fields[n][1], "datatype": 4 if n == "ring" else 7,
                                     "count": 1} for n in POINT_DT.names]
    session = json.loads((rec.root / "session.json").read_text())
    mc = session.get("mount_calib")
    mount = Mount.load(mc["path"]) if mc else Mount.nominal()
    state, p = leg_state(rec)
    frames = list(rec.lidar_frames())
    t_end = frames[-1][0] if args.until_s is None else frames[0][0] + args.until_s
    window = [f for f in frames if t_end - args.seconds <= f[0] <= t_end]
    sensor = dict(flow.planner.PlannerConfig(**dict(pdict, device="cpu")).sensor)
    state.until(window[0][0])
    q0 = state.qpos_at(window[0][0])
    memory = CloudMemory(np.asarray(q0[:2]), rng=np.random.default_rng(0), **sensor)
    memory.latency = 0.0
    conv = CloudFromSensorFrame(memory.body_filter, mount, transport_s=0.01)
    for t_f, pts, t_s in window:
        state.until(t_f + 0.05)
        p_w, org = conv(pts, t_s, t_f, state.qpos_at)
        memory.add_cloud(0.0, p_w, org, None)
        memory.step(t_f)
    q = state.qpos_at(window[-1][0])
    sx, sy = q[:2]
    w, x, y, z = q[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    view = G.memory_view(memory, near_only=False)
    cfg = dataclasses.replace(G.preset("deploy"), max_guide_boxes=10_000, guide_range_m=1e9)
    boxes = G.split_boxes(view, (sx, sy), cfg)
    c, s = math.cos(-yaw), math.sin(-yaw)
    scene, attached = [], []
    for b in boxes:
        dx, dy = b["center_xy"][0] - sx, b["center_xy"][1] - sy
        cx, cy = c * dx - s * dy, s * dx + c * dy
        nb = {"center_xy": [round(cx, 3), round(cy, 3)], "half_xy": [round(v, 3) for v in b["half_xy"]],
              "yaw": round(-yaw, 4), "z_bottom": round(b["z_bottom"], 3), "z_top": round(b["z_top"], 3)}
        gap = max(math.hypot(cx, cy) - math.hypot(*b["half_xy"]), 0.0)
        (attached if gap < 0.3 and b["z_bottom"] < 1.35 and math.hypot(cx, cy) < 0.6 else scene).append(nb)
    base = json.loads((ROOT / "sim/courses/lab-site-2m/task.json").read_text())
    out = ROOT / "sim/courses" / args.name
    out.mkdir(parents=True, exist_ok=True)
    task = dict(base, task_id=f"twin-{args.name}", scene_boxes_world=scene, goal_world_xy=list(args.goal),
                deadline_s=args.deadline,
                description=f"twin of {args.session} (last {args.seconds:.0f} s of lidar, split boxes); "
                            f"{len(attached)} boxes attached to the robot left out")
    (out / "task.json").write_text(json.dumps(task, indent=1) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(7, 7))
    for b in scene:
        cy_, sy_ = math.cos(b["yaw"]), math.sin(b["yaw"])
        hx, hy = b["half_xy"]
        pts = [np.array(b["center_xy"]) + np.array([cy_ * u - sy_ * v, sy_ * u + cy_ * v])
               for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
        over = b["z_bottom"] > 0.3
        ax.add_patch(Polygon(pts, fill=True, alpha=0.35, fc="orange" if over else "tab:red",
                             ec="orange" if over else "tab:red"))
        if abs(b["center_xy"][1]) < 2 and -0.5 < b["center_xy"][0] < 4:
            ax.text(*b["center_xy"], f"{b['z_bottom']:.2f}-{b['z_top']:.2f}", fontsize=6)
    ax.plot(0, 0, "co", ms=10)
    ax.arrow(0, 0, 0.4, 0, color="c", width=0.02)
    ax.plot(*args.goal, "y*", ms=16)
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"{args.name}: red = ground obstacle, orange = overhead (z bottom-top m)")
    fig.savefig(out / "twin.png", dpi=110)
    print(f"{out / 'task.json'}: {len(scene)} boxes ({len(attached)} attached to the robot left out), "
          f"goal {args.goal}, window {len(window)} frames ending {t_end - frames[0][0]:.1f} s into the recording")
    for b in scene:
        if -0.5 < b["center_xy"][0] < 3.5 and abs(b["center_xy"][1]) < 1.6:
            print("  ", b["center_xy"], b["half_xy"], b["z_bottom"], b["z_top"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
