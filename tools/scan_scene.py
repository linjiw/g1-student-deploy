#!/usr/bin/env python3
"""Scan the real scene and build its twin: the lidar scan, Deploy Studio's twin of it, and a sim course to rehearse.

  1. scan     `g1 site` (read-only): the node's own lidar memory around the standing robot, with the robot profile's
              mount calibration and rear blind, for --seconds; its verdict, the planner's boxes and route, a PNG and
              the memory in node dump format (memory.npz). --from-scan DIR reuses a scan; --from-session DIR uses a
              finished console session's final memory (node_memory.npz: what the robot saw during its walks).
  2. twin     Deploy Studio's own scan-to-twin fit (studio.twin.twin_from_grid, cell-exact rectangles, run with
              Studio's Python from G1_STUDIO_ROOT): twin.json in the memory's world frame.
  3. course   the twin in the start frame as a deploy-kit course (course/task.json, tools/site_to_course.py: boxes
              that touch the standing robot - harness, rope, gantry hook - are left out), for `g1 sim --task`.
  4. Studio   the twin saved as a Studio arena over Studio's local API (http://127.0.0.1:8790), so Arena, Navigate and
              Rehearse show the real scene. Skipped (and said so) when Studio is not running.

Everything lands in OUT (default runs/lab-day-YYYYMMDD/scans/scan-HHMMSS/), summarised in scan.json.

usage: scan_scene.py --robot lab-g1 --goal 2 0 [--seconds 8] [--out DIR] [--no-studio]
       scan_scene.py --from-scan DIR | --from-session SESSION_DIR [--goal A L]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIO_ROOT = Path(os.environ.get("G1_STUDIO_ROOT", Path.home() / "code/g1-deploy-studio"))
STUDIO_URL = os.environ.get("G1_STUDIO_URL", "http://127.0.0.1:8790")

TWIN_SCRIPT = r"""
import json, sys
import numpy as np
from studio import goal_check, twin
mem, sx, sy, syaw, gx, gy, label = sys.argv[1], *map(float, sys.argv[2:7]), sys.argv[7]
grid = goal_check.load_grid(mem, basis='deploy-kit lidar scan (node memory)')
goal = None if np.isnan(gx) else [gx, gy]
arena = twin.twin_from_grid(grid, start_xy=[sx, sy], start_yaw_rad=syaw, goal_xy=goal, label=label, min_area_m2=0.04)
print(json.dumps(arena, default=float))
"""


def run_scan(args, out: Path):
    cmd = [str(ROOT / "g1"), "site", "--robot", args.robot, "--goal", *map(str, args.goal), "--seconds", str(args.seconds),
           "--json", str(out / "site.json"), "--png", str(out / "site.png"), "--dump-memory", str(out / "memory.npz")]
    with open(out / "site.log", "w") as log:
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if not (out / "site.json").exists() or not (out / "memory.npz").exists():
        tail = (out / "site.log").read_text()[-600:]
        raise SystemExit(f"scan failed (exit {rc}): {tail}")
    return json.loads((out / "site.json").read_text())


def from_session(session: Path, out: Path, goal):
    """A finished session's final node memory as a scan: start = where the robot started (the odometry origin)."""
    z = np.load(session / "node_memory.npz", allow_pickle=False)
    np.savez_compressed(out / "memory.npz", **{k: z[k] for k in z.files}, start_xy=np.zeros(2), start_yaw=0.0)
    info = json.loads((session / "session.json").read_text())
    a, l = goal or info.get("goal") or [2.0, 0.0]
    return {"verdict": "SESSION", "reasons": [f"final map of session {session.name}"], "start_xy": [0.0, 0.0],
            "start_yaw": 0.0, "goal_xy": [a, l], "goal_ahead_left": [a, l], "boxes": json.loads(str(z["boxes"])),
            "source_session": str(session)}


def studio_twin(mem: Path, site: dict, label: str):
    py = STUDIO_ROOT / ".venv/bin/python"
    if not py.exists():
        return None, f"no Deploy Studio at {STUDIO_ROOT}"
    (sx, sy), syaw = site["start_xy"], site["start_yaw"]
    gx, gy = site.get("goal_xy") or [math.nan, math.nan]
    r = subprocess.run([str(py), "-c", TWIN_SCRIPT, str(mem), str(sx), str(sy), str(syaw), str(gx), str(gy), label],
                       cwd=str(STUDIO_ROOT), capture_output=True, text=True, timeout=120)
    if r.returncode:
        return None, "Studio twin failed: " + (r.stderr or r.stdout)[-400:]
    return json.loads(r.stdout), None


def studio_save(arena: dict, description: str):
    """POST the twin to the running Studio as a new arena (its same-origin token, as the browser does)."""
    host = STUDIO_URL.split("//", 1)[1]
    with urllib.request.urlopen(urllib.request.Request(f"{STUDIO_URL}/api/workbench", headers={"Host": host}), timeout=5) as r:
        token = json.loads(r.read())["csrf_token"]
    body = {"label": arena["label"][:120], "boxes": arena["boxes"], "bounds": arena.get("bounds"),
            "presets": arena.get("presets") or [], "description": description[:500]}
    req = urllib.request.Request(f"{STUDIO_URL}/api/arenas", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Origin": STUDIO_URL, "Host": host,
                                          "X-Studio-Token": token})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", default="lab-g1")
    ap.add_argument("--goal", type=float, nargs=2, default=None, metavar=("AHEAD", "LEFT"))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-scan", default=None)
    ap.add_argument("--from-session", default=None)
    ap.add_argument("--no-studio", action="store_true")
    args = ap.parse_args(argv)
    stamp = time.strftime("%H%M%S")
    if args.from_scan:
        out = Path(args.from_scan).resolve()
        site = json.loads((out / "site.json").read_text())
    else:
        default = ROOT / "runs" / f"lab-day-{time.strftime('%Y%m%d')}" / "scans" / (
            f"session-{Path(args.from_session).name}" if args.from_session else f"scan-{stamp}")
        out = Path(args.out or default).resolve()
        out.mkdir(parents=True, exist_ok=True)
        if args.from_session:
            site = from_session(Path(args.from_session), out, args.goal)
            (out / "site.json").write_text(json.dumps(site, indent=1) + "\n")
        else:
            site = run_scan(dict_args(args), out)
    name = out.name
    summary = {"name": name, "dir": str(out), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdict": site.get("verdict"),
               "reasons": site.get("reasons", []), "frames": site.get("frames"), "goal_ahead_left": site.get("goal_ahead_left"),
               "planner_boxes": len(site.get("boxes") or []), "twin_boxes": None, "course": None, "studio_arena": None,
               "warnings": []}
    arena, err = studio_twin(out / "memory.npz", site, f"Scan {name} ({time.strftime('%H:%M')})")
    if arena is None:  # no Studio here: the planner's own (merged) boxes are the twin
        summary["warnings"].append(err + "; the course uses the planner's merged boxes")
        twin_boxes = site.get("boxes") or []
    else:
        twin_boxes = arena["boxes"]
        summary["warnings"] += arena.get("warnings") or []
        (out / "twin.json").write_text(json.dumps(arena, indent=1, default=float) + "\n")
    summary["twin_boxes"] = len(twin_boxes)
    (out / "site_twin.json").write_text(json.dumps(dict(site, boxes=twin_boxes), indent=1, default=float) + "\n")
    sys.path.insert(0, str(ROOT / "tools"))
    import site_to_course

    course = out / "course" / "task.json"
    site_to_course.main([str(out / "site_twin.json"), "--name", name, "--out", str(course)])
    summary["course"] = str(course)
    task = json.loads(course.read_text())
    summary["course_boxes"] = len(task["scene_boxes_world"])
    if arena is not None and not args.no_studio:
        try:
            saved = studio_save(arena, f"Twin of the real scene scanned {summary['time']} ({name}, verdict "
                                       f"{summary['verdict']}): Studio's fit to the robot's lidar memory.")
            summary["studio_arena"] = {"id": saved.get("id"), "label": saved.get("label")}
        except Exception as exc:
            summary["warnings"].append(f"Studio did not take the twin ({type(exc).__name__}: {exc}); is it running?")
    (out / "scan.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


def dict_args(args):
    if args.goal is None:
        args.goal = [2.0, 0.0]
    return args


if __name__ == "__main__":
    sys.exit(main())
