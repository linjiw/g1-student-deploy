#!/usr/bin/env python3
"""Turn a real site, as the robot perceived it (`g1 site --json`), into a simulation course: a digital twin of the
first walk, to rehearse it before the robot does it.

The planner's own guide boxes (Flow memory_boxes on the robot's lidar memory) become the course obstacles, expressed
in the start frame (start at the origin facing +x, as every course). Boxes that intersect the robot's standing volume
at the start (radius 0.3 m, up to 1.35 m) are attached to the robot - harness, rope, gantry hook, self-returns - and
are left out of the physical scene (listed in the course description); the simulated lidar then shows what the real
one would without them. The goal is the site check's goal.

usage: site_to_course.py site.json --name lab-2m [--deadline 25]   ->  sim/courses/<name>/task.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("site_json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--deadline", type=float, default=25.0)
    ap.add_argument("--keep-attached", action="store_true", help="also keep boxes that intersect the robot at the start")
    args = ap.parse_args(argv)
    site = json.loads(Path(args.site_json).read_text())
    sx, sy = site["start_xy"]
    yaw = site["start_yaw"]
    c, s = math.cos(-yaw), math.sin(-yaw)

    def to_start(x, y):
        dx, dy = x - sx, y - sy
        return [round(c * dx - s * dy, 3), round(s * dx + c * dy, 3)]

    boxes, attached = [], []
    for b in site["boxes"]:
        cx, cy = to_start(*b["center_xy"])
        nb = {"center_xy": [cx, cy], "half_xy": b["half_xy"], "z_bottom": b["z_bottom"], "z_top": b["z_top"],
              "yaw": round(b["yaw"] - yaw, 4)}
        hx, hy = b["half_xy"]
        bc, bs = math.cos(nb["yaw"]), math.sin(nb["yaw"])
        local = (bc * -cx + bs * -cy, -bs * -cx + bc * -cy)  # the start (origin) in the box frame
        touches = abs(local[0]) <= hx + 0.3 and abs(local[1]) <= hy + 0.3 and b["z_bottom"] < 1.35
        if touches and not args.keep_attached:
            attached.append(nb)
            continue
        boxes.append(nb)
    a, l = site.get("goal_ahead_left") or to_start(*site["goal_xy"])
    task = {"task_id": f"site-{args.name}", "family": "site",
            "description": f"twin of a real site from {Path(args.site_json).name} (site verdict {site['verdict']}); "
                           f"{len(attached)} boxes attached to the robot at the start left out: "
                           + json.dumps([[bb["center_xy"], bb["half_xy"], bb["z_bottom"], bb["z_top"]] for bb in attached]),
            "scene_boxes_world": boxes, "initial_state_record": "sim/courses/stand_start.npz", "initial_frame": 0,
            "initial_root_xy": [0.0, 0.0], "goal_world_xy": [a, l], "deadline_s": args.deadline,
            "goal_radius_m": 0.4, "goal_speed_m_s": 2.0, "goal_hold_s": 0.1, "fall_height_m": 0.38, "fall_tilt_rad": 0.8,
            "obstacle_contact_threshold_n": 5.0, "source_site": site}
    out = ROOT / "sim/courses" / args.name / "task.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(task, indent=1) + "\n")
    print(f"{out}: {len(boxes)} boxes, {len(attached)} attached to the robot left out, goal {[a, l]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
