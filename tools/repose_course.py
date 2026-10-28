#!/usr/bin/env python3
"""Where did a finished robot session really start? Its standing map at GO registered to the scan its goal was picked
on (g1deploy/registration.py), and the scan's sim course moved to that start, to rehearse the walk the robot had.

The session needs plan dumps (`debug_plans` profiles: `plans/plan_0000_in.npz` is the node's memory and pose at the
first plan). The scan is the session's `scene_scan.txt` unless given. Prints the start check (moved, turned, the scan's
goal from the real start) and writes `<scan>/course-at-<session>/task.json`:

  g1 sim --profile P --task runs/.../scans/SCAN/course-at-SESSION/task.json --goal A L ...

usage: repose_course.py SESSION_DIR [--scan SCAN_DIR] [--goal A L] [--plan 0]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1deploy import registration, workspace  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--scan", default=None, help="scan directory (default: the session's scene_scan.txt)")
    ap.add_argument("--goal", type=float, nargs=2, default=None, help="goal on the scan (default: the session's)")
    ap.add_argument("--plan", type=int, default=0, help="which plan dump stands for GO (default the first)")
    args = ap.parse_args(argv)
    s = Path(args.session)
    scan = Path(args.scan) if args.scan else s.parent / "scans" / (s / "scene_scan.txt").read_text().strip()
    goal = args.goal or json.loads((s / "session.json").read_text()).get("goal")
    with np.load(scan / "memory.npz") as z:
        start = [float(v) for v in z["start_xy"]]
        grid = dict(workspace.code_grid(z["support"], z["ceiling"], z["lo"], float(z["res"]), start, 5.0, 0.1),
                    start=[*start, float(z["start_yaw"])])
    z = np.load(sorted((s / "plans").glob("plan_*_in.npz"))[args.plan])
    h = np.asarray(z["hist"][-1], float)
    yaw = math.atan2(2 * (h[3] * h[6] + h[4] * h[5]), 1 - 2 * (h[5] ** 2 + h[6] ** 2))
    live = dict(workspace.code_grid(z["support"], z["ceiling"], z["lo"], float(z["res"]), h[:2], 4.0, 0.1), ok=True,
                robot=[float(h[0]), float(h[1]), yaw])
    check = registration.start_check(grid, live, goal)
    print(json.dumps({k: v for k, v in check.items() if k != "t"}, default=float))
    if not check["ok"]:
        return 1
    task = registration.repose_task(json.loads((scan / "course" / "task.json").read_text()), check["theta_rad"], check["t"])
    out = scan / f"course-at-{s.name}" / "task.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(task, start_check=check), indent=1, default=float) + "\n")
    print(f"-> {out}  (rehearse with --goal {check['goal_now'][0]} {check['goal_now'][1]} for the scan's goal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
