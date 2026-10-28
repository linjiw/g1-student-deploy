#!/usr/bin/env python3
"""Odometry-drift gate: does a deploy profile still do the course with the odometry the robot really has?

Runs the full real-time chain (`g1 sim`: SONIC runner, the node, the raw lidar path) once per odometry level and
repeat, from the course's start, and scores each walk on the simulator's true path:
  arrived   the attempt's result (attempts.jsonl)
  gate      every segment in the task's `gates` ([x0, y0, x1, y1], world) crossed by the walked pelvis path (e.g. the
            opening between two towers: through, not around)
  contact   the largest obstacle contact force (sim.npz), > 5 N = touched
Odometry levels (`--levels`, `;`-separated):
  exact                  the simulator's odometry
  lio                    the sim LIO stand-in (true pose; the node registers raw frames with it)
  scale,rot              the stand-in with each step's displacement scaled and turned (deg): an error on true odometry
  motion[:PRESET]        the stand-in with the step-tied leg-odometry error of g1deploy/odom_error.py (lab-g1-leg,
                         training), a new draw per repeat
  leg                    the sim's own leg odometry (g1deploy/leg_odometry.py on simulated joints)
  fused                  the stand-in and leg odometry in one filter (node --odom fused, g1deploy/odom_fusion.py)
  lio+FAULT, fused+FAULT a LIO fault ~2 s into the walk (FAULTS: jump 0.5 m, runaway 0.5 m/s, freeze): the walk
                         should end in a safe stop (no contact), not an arrival
Results: OUT/<level>/<course>-r<k>/ (the session), OUT/summary.json and a table. Sessions run one at a time on the
isolated rehearsal ports (`--port-base 5656`); keep the machine quiet meanwhile.

usage: drift_gate.py --profile P --course gap-069 [--course ...|--task T.json] [--levels "exact;0.95,0;motion"]
                     [--repeats 3] --out runs/drift/NAME
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "wait:init,key:s,wait:control,wait:ready,sleep:3,key:g,wait:arrived,sleep:3"
CONTACT_N = 5.0


FAULTS = {"jump": {"kind": "jump", "m": 0.5}, "runaway": {"kind": "runaway", "m": 0.5}, "freeze": {"kind": "freeze", "m": 0.0}}
FAULT_AT_S = 31.0  # after the LIO stand-in starts: the scripted GO comes ~29 s in


def odom_args(level, seed):
    if "+" in level:
        odom, fault = level.split("+", 1)
        return ["--odom", odom, "--lio-noise", json.dumps({"fault": dict(FAULTS[fault], at_s=FAULT_AT_S)})]
    if level == "fused":
        return ["--odom", "fused"]
    if level == "exact":
        return []
    if level == "lio":
        return ["--odom", "lio"]
    if level == "leg":
        return ["--odom", "leg"]
    if level.startswith("motion"):
        preset = level.split(":", 1)[1] if ":" in level else "lab-g1-leg"
        return ["--odom", "lio", "--lio-noise", json.dumps({"motion": preset, "seed": seed})]
    sc, rot = (float(v) for v in level.split(","))
    return ["--odom", "lio", "--lio-noise", json.dumps({"scale": sc, "rotate_deg": rot})]


def crossed(path, gate):
    """Does the polyline `path` (N, 2) cross the segment `gate` (x0, y0, x1, y1)?"""
    a, b = np.asarray(gate[:2], float), np.asarray(gate[2:], float)
    p, q = path[:-1], path[1:]

    def side(u, v, w):  # sign of the cross product (v - u) x (w - u)
        return np.sign((v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0]))
    s1, s2 = side(a, b, p), side(a, b, q)
    s3, s4 = side(p, q, a[None]), side(p, q, b[None])
    return bool(np.any((s1 * s2 < 0) & (s3 * s4 < 0)))


def score(session: Path, task: dict):
    out = {"session": str(session), "result": None, "gates": None, "contact_n": None, "duration_s": None}
    rows = [json.loads(l) for l in (session / "attempts.jsonl").read_text().splitlines() if l.strip()] \
        if (session / "attempts.jsonl").exists() else []
    if rows:
        out["result"], out["duration_s"] = rows[-1].get("result"), rows[-1].get("duration_s")
    try:
        with np.load(session / "sim.npz") as z:
            path = z["root_pos"][:, :2]
            out["contact_n"] = round(float(np.max(z["contact_n"])), 1) if len(z["contact_n"]) else 0.0
        if task.get("gates"):
            out["gates"] = all(crossed(path, g) for g in task["gates"])
        out["end_xy"] = np.round(path[-1], 2).tolist()
    except (OSError, KeyError, ValueError):
        pass
    out["pass"] = bool(out["result"] == "arrived" and out["gates"] is not False
                       and (out["contact_n"] or 0.0) <= CONTACT_N)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--course", action="append", default=[], help="sim/courses/<name> (repeatable)")
    ap.add_argument("--task", action="append", default=[], help="a task.json, e.g. a scan's reposed twin (repeatable)")
    ap.add_argument("--goal", type=float, nargs=2, default=None, help="for every task (default: each task's own goal)")
    ap.add_argument("--levels", default="exact;lio;0.95,0;0.9,0;1.0,-2;1.0,-4;motion")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port-base", type=int, default=5656)
    args = ap.parse_args(argv)
    tasks = [(c, ROOT / "sim/courses" / c / "task.json") for c in args.course] + [(Path(t).parent.name, Path(t)) for t in args.task]
    if not tasks:
        raise SystemExit("give --course or --task")
    levels = [x.strip() for x in args.levels.split(";") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, tpath in tasks:
        task = json.loads(tpath.read_text())
        goal = args.goal or task["goal_world_xy"]
        for level in levels:
            for k in range(args.repeats):
                d = out / level.replace(",", "_").replace(":", "-") / f"{name}-r{k}"
                if not (d / "console_summary.json").exists():
                    if d.exists():
                        subprocess.run(["rm", "-rf", str(d)], check=True)
                    d.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [str(ROOT / "g1"), "sim", "--profile", args.profile, "--port-base", str(args.port_base),
                           "--task", str(tpath.resolve()), "--goal", *(str(float(v)) for v in goal), "--script", SCRIPT,
                           "--step-timeout", "40", "--record", str(d)] + odom_args(level, 1000 + k)
                    t0 = time.time()
                    with open(str(d) + ".out", "w") as log:
                        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       timeout=300)
                    print(f"{name} {level} r{k}: {time.time() - t0:.0f} s", flush=True)
                r = dict(score(d, task), course=name, level=level, repeat=k)
                rows.append(r)
                print(f"  {r['result']}, gates {r['gates']}, contact {r['contact_n']} N -> {'PASS' if r['pass'] else 'fail'}",
                      flush=True)
    table = {}
    for r in rows:
        t = table.setdefault(f"{r['course']} | {r['level']}", {"n": 0, "pass": 0, "arrived": 0, "gates": 0})
        t["n"] += 1
        t["pass"] += r["pass"]
        t["arrived"] += r["result"] == "arrived"
        t["gates"] += bool(r["gates"])
    (out / "summary.json").write_text(json.dumps({"profile": args.profile, "levels": levels, "rows": rows,
                                                  "table": table}, indent=1) + "\n")
    print(f"\n{'course | odometry':<40s} pass  arrived  through gates")
    for k, t in table.items():
        print(f"{k:<40s} {t['pass']}/{t['n']}   {t['arrived']}/{t['n']}      {t['gates']}/{t['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
