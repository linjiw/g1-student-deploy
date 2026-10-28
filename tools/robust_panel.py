#!/usr/bin/env python3
"""Robustness panel: the lab findings of 2026-10-23 as lock-step courses, scored under several conditions.

Every course starts standing at the origin facing +x and walks the deployed sensor chain: MID-360 scan
pattern with per-point timing, range noise and the self filter, a stand of `--prescan-s` (operators stand
15-20 s before GO, and the memory keeps everything it saw meanwhile), sensor route, 0.3 s planner latency
with predictive anchoring, LIO-like odometry noise. Courses are authored here, not taken from any
evaluation packet:

  open-2m      clear floor, 2 m ahead (the lab twin's clear path)
  pillar       one column on the path
  slalom       three columns, alternating sides
  room         closed room 5.4 x 6.1 m, robot inside (memory_boxes made it one solid box)
  l-shape      L-shaped 0.9 m furniture beside and behind the start (one box around the robot)
  table        1.2 x 0.8 m table, top 0.92-0.97 m, beside the goal (top filed as a ceiling)
  dense        6 x 7 m room with 26 furniture boxes along the walls and in the middle, clear lane
  wall-start   start 0.42 m beside a long wall, goal along it (the self mask must not hide a real wall)
  wall-close   the same wall 0.33 m from the pelvis: closer than any GO site would allow, the self mask's edge
  rope-behind  a 2 cm "safety rope" from 0.2 m behind the pelvis, 1.0-1.3 m high (the lab door test, 2026-10-24;
               static here, on the robot it follows the robot)
  door         a 1.0 m wide door with its beam underside at 1.17 m, 1.5 m ahead (ducking: --guards deploy)
  door-rope    the door with the rope behind: the lab test that E-stopped
  gap-069      a 0.69 m gap between two block towers 1.8 m ahead (the lab-g1 walk of 2026-10-27)

Conditions (`--conditions`, comma separated, see CONDITIONS): self-filter margin, guard preset, head
occlusion bands, odometry (`*-legdrift`: lab-g1's leg-odometry error, g1deploy/odom_error.py). Results: <out>/<condition>/<course>__s<seed>/ (score.json, plans.json, closed_loop.npz),
<out>/summary.json and a table on stdout.

usage: robust_panel.py --out runs/robust/panel1 [--conditions tag,kit,kit-guards,leaky-guards] [--seeds 3]
                       [--courses open-2m,pillar] [--workers 2]
       robust_panel.py --realtime --out runs/robust/rt1 ...   the same courses through sim/realtime/rehearse.py
                       (SONIC C++ runner, DDS on loopback, raw MID-360 path, the node), one run at a time
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIO = {"xy_rw_m": 0.01, "yaw_rw_rad": 0.005, "xy_white_m": 0.002, "yaw_white_rad": 0.001}
LAB_BANDS = {"bands": [[-45, -35], [40, 50]]}
CONDITIONS = {
    "tag": dict(margin=0.03, guards=None),                       # deploy-2026-10-23 as tagged
    "kit": dict(margin=0.05, guards=None),                       # deploy-kit: self-filter margin 5 cm
    "kit-guards": dict(margin=0.05, guards="deploy"),            # + g1deploy/guards.py
    "leaky-guards": dict(margin=0.03, guards="deploy"),          # guards with the leaky 3 cm filter
    "kit-bands": dict(margin=0.05, guards=None, occlusion=LAB_BANDS),
    "kit-guards-bands": dict(margin=0.05, guards="deploy", occlusion=LAB_BANDS),
    "kit-walls": dict(margin=0.05, guards="walls"),
    "kit-slow": dict(margin=0.05, guards=None, lookahead=1.0),  # --goal-lookahead 1.0: the first-walk speed
    "kit-guards-slow": dict(margin=0.05, guards="deploy", lookahead=1.0),
    "kit-walls-slow": dict(margin=0.05, guards="walls", lookahead=1.0),
    # odometry like lab-g1's leg odometry instead of LIO noise (g1deploy/odom_error.py: step-tied, smears the memory)
    "kit-guards-legdrift": dict(margin=0.05, guards="deploy", odom={"motion": "lab-g1-leg"}),
    "kit-guards-trainingdrift": dict(margin=0.05, guards="deploy", odom={"motion": "training"}),
}
WALL_H = 1.2


def box(cx, cy, hx, hy, top, bottom=0.0):
    return {"center_xy": [float(cx), float(cy)], "half_xy": [float(hx), float(hy)], "yaw": 0.0,
            "z_bottom": float(bottom), "z_top": float(top), "velocity_xy": [0.0, 0.0], "t_ref": 0.0}


def walls(x0, x1, y0, y1, t=0.05, h=WALL_H):
    """Four walls of thickness 2t around the interior [x0, x1] x [y0, y1]."""
    cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    return [box(cx, y0 - t, hx + 2 * t, t, h), box(cx, y1 + t, hx + 2 * t, t, h),
            box(x0 - t, cy, t, hy, h), box(x1 + t, cy, t, hy, h)]


ROPE = box(-0.5, 0.0, 0.3, 0.01, 1.3, 1.0)
DOOR = [box(1.5, 0.55, 0.05, 0.05, 2.0), box(1.5, -0.55, 0.05, 0.05, 2.0), box(1.5, 0.0, 0.05, 0.6, 1.4, 1.17)]


def courses():
    out = {
        "open-2m": dict(boxes=[], goal=[2.0, 0.0], deadline=12.0),
        "pillar": dict(boxes=[box(1.5, 0.05, 0.15, 0.15, 1.8)], goal=[3.0, 0.0], deadline=14.0),
        "slalom": dict(boxes=[box(1.2, 0.35, 0.12, 0.12, 1.8), box(2.2, -0.35, 0.12, 0.12, 1.8),
                              box(3.2, 0.35, 0.12, 0.12, 1.8)], goal=[4.2, 0.0], deadline=18.0),
        "room": dict(boxes=walls(-1.2, 4.2, -3.05, 3.05), goal=[2.5, 0.4], deadline=16.0),
        "l-shape": dict(boxes=[box(0.4, -0.75, 1.4, 0.25, 0.9), box(-1.25, 0.65, 0.25, 1.15, 0.9)],
                        goal=[2.6, 0.4], deadline=14.0),
        "table": dict(boxes=[box(2.5, 0.95, 0.6, 0.4, 0.97, 0.92)]
                      + [box(2.5 + sx * 0.55, 0.95 + sy * 0.35, 0.025, 0.025, 0.92) for sx in (-1, 1) for sy in (-1, 1)],
                      goal=[2.5, 0.2], deadline=14.0),
        "wall-start": dict(boxes=[box(1.5, 0.52, 3.0, 0.1, WALL_H)], goal=[3.0, 0.0], deadline=14.0),
        "wall-close": dict(boxes=[box(1.5, 0.43, 3.0, 0.1, WALL_H)], goal=[3.0, -0.1], deadline=14.0),
        "rope-behind": dict(boxes=[ROPE], goal=[2.0, 0.0], deadline=12.0),
        "door": dict(boxes=DOOR, goal=[3.0, 0.0], deadline=16.0),
        "door-rope": dict(boxes=DOOR + [ROPE], goal=[3.0, 0.0], deadline=16.0),
        # the 0.69 m gap between two block towers of 2026-10-27, 1.8 m ahead (sim/courses/gap-069)
        "gap-069": dict(boxes=[box(1.8, -0.465, 0.12, 0.12, 1.0), box(1.8, 0.525, 0.12, 0.18, 1.25)], goal=[2.8, 0.0],
                        deadline=14.0),
    }
    rng = np.random.default_rng(20261023)
    dense = walls(-1.5, 4.5, -3.5, 3.5)
    while len(dense) < 30:
        hx, hy = rng.uniform(0.2, 0.6, 2)
        cx, cy = rng.uniform(-1.3 + hx, 4.3 - hx), rng.uniform(-3.3 + hy, 3.3 - hy)
        if abs(cy) - hy < 0.75 and cx + hx > -1.0:  # keep a 1.5 m lane along y = 0 clear
            continue
        if any(abs(cx - b["center_xy"][0]) < hx + b["half_xy"][0] + 0.1 and abs(cy - b["center_xy"][1]) < hy + b["half_xy"][1] + 0.1
               for b in dense[4:]):
            continue
        dense.append(box(cx, cy, hx, hy, float(rng.choice([0.45, 0.75, 0.9, 1.2]))))
    out["dense"] = dict(boxes=dense, goal=[3.0, 0.0], deadline=16.0)
    return out


def write_tasks(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    stand = dest / "stand.npz"
    np.savez(stand, root_position_m=np.array([[0.0, 0.0, 0.793]]), root_quaternion_wxyz=np.array([[1.0, 0, 0, 0]]))
    paths = {}
    for name, c in courses().items():
        task = dict(task_id=name, family=name, scene_boxes_world=c["boxes"], goal_world_xy=c["goal"],
                    deadline_s=c["deadline"], initial_state_record=str(stand.resolve()), initial_frame=0,
                    initial_root_xy=[0.0, 0.0], ground_z_m=0.0, goal_radius_m=0.4, goal_speed_m_s=2.0, goal_hold_s=0.1,
                    fall_height_m=0.38, fall_tilt_rad=0.8, obstacle_contact_threshold_n=5.0,
                    source="tools/robust_panel.py (authored course; not an evaluation packet)")
        paths[name] = dest / f"{name}.json"
        paths[name].write_text(json.dumps(task, indent=1) + "\n")
    return paths


def _run(job):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch

    torch.set_num_threads(2)
    from sim.lockstep import run_episode

    cond = CONDITIONS[job["condition"]]
    out = Path(job["out"])
    if (out / "score.json").is_file():
        return job
    if out.exists():
        import shutil

        shutil.rmtree(out)
    lidar_cfg = {"occlusion": cond["occlusion"]} if cond.get("occlusion") else None
    sensor = {"mounts": ["mid360_inverted"], "latency_s": 0.1, "range_m": 4.0, "self_margin_m": cond["margin"]}
    t0 = time.time()
    try:
        run_episode(job["task"], job["seed"], out, latency_s=job["latency_s"], anchor_mode="predict", start="stand",
                    route_source="sensor", lidar="mid360", lidar_cfg=lidar_cfg, prescan_s=job["prescan_s"],
                    odom_noise=cond.get("odom", LIO),
                    planner_overrides={"sensor": sensor}, guards=cond.get("guards"), dump_memory=True,
                    goal_lookahead=cond.get("lookahead"))
    except Exception as exc:  # recorded as runtime_error in score.json by run_episode
        print(f"{out}: {type(exc).__name__}: {exc}", flush=True)
    job["wall_s"] = time.time() - t0
    return job


def _run_realtime(job):
    import subprocess

    cond = CONDITIONS[job["condition"]]
    out = Path(job["out"])
    if (out / "score.json").is_file():
        return job
    if out.exists():
        import shutil

        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "sim/realtime/rehearse.py"), "--task", job["task"], "--out", str(out),
           "--prescan-s", str(job["prescan_s"]), "--self-margin", str(cond["margin"]), "--seed", str(job["seed"])]
    cmd += ["--guards", cond["guards"]] if cond.get("guards") else []
    cmd += ["--goal-lookahead", str(cond["lookahead"])] if cond.get("lookahead") else []
    cmd += ["--lidar-cfg", json.dumps({"occlusion": cond["occlusion"]})] if cond.get("occlusion") else []
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0 or not (out / "score.json").is_file():
        print(f"{out}: rehearse exited {r.returncode}\n{r.stderr[-1500:]}", flush=True)
        if not (out / "score.json").is_file():  # scored as a failure, not left out of the panel
            outcome = "never_streamed" if "node never started streaming" in r.stderr else "rehearse_error"
            if outcome == "never_streamed" and lidar_points(out) == 0:
                outcome = "no_lidar"  # the node fused no lidar point: the rig, not the policy (unmeasured)
            (out / "score.json").write_text(json.dumps({"outcome": outcome, "error": r.stderr[-600:]}, indent=1) + "\n")
    job["wall_s"] = time.time() - t0
    return job


def lidar_points(d: Path):
    """Points the node fused, from its --dump-memory file; None if unknown."""
    f = d / "node_memory.npz"
    if not f.is_file():
        return None
    try:
        return int(json.loads(str(np.load(f)["stats"]))["points"])
    except (KeyError, ValueError, OSError):
        return None


UNMEASURED = ("no_lidar", "rehearse_error")


def episode_metrics(d: Path):
    s = json.loads((d / "score.json").read_text())
    realtime = any((d / f).is_file() for f in ("node_plans.json", "node.log", "runner.log", "config.json")) \
        or s.get("outcome") in ("never_streamed",) + UNMEASURED
    if realtime:  # a real-time rehearsal: plan log + events (or only the failure score.json written by _run_realtime)
        return realtime_metrics(d, s)
    plans = json.loads((d / "plans.json").read_text()) if (d / "plans.json").is_file() else []
    out = dict(outcome=s.get("outcome", "incomplete"), elapsed_s=s.get("elapsed_s"), min_pelvis_m=s.get("minimum_root_height_m"),
               max_contact_n=s.get("maximum_obstacle_contact_n"), max_tilt_rad=s.get("maximum_root_tilt_rad"),
               final_dist_m=s.get("final_goal_distance_xy_m"), plans=len(plans),
               latency_p95_s=(s.get("plan_latency_s") or {}).get("p95"))
    out["first_rule"] = plans[0]["rule"] if plans else None
    out["first_no_safe"] = bool(plans and (plans[0].get("no_safe_candidate") or (plans[0].get("guard") or {}).get("no_safe")))
    out["no_safe_plans"] = sum(bool(p.get("no_safe_candidate") or (p.get("guard") or {}).get("no_safe")) for p in plans)
    out["fallback_go"] = sum(p["rule"] == "fallback_go_max_clearance" for p in plans)
    out["calm_stand"] = sum(p["rule"] == "calm_stand" for p in plans)
    out["calm_stop"] = sum(p["rule"] == "calm_stop" for p in plans)
    out["envelope_reselect"] = sum("+envelope" in p["rule"] for p in plans)
    out["dropped"] = sum(p["rule"] == "dropped" for p in plans)
    plans = [p for p in plans if p["rule"] != "dropped"]
    out["plan_knee_max"] = max((p.get("plan_knee_max_rad") or 0.0 for p in plans), default=None)
    out["plan_zmin"] = min((min(p["plan_root_z"]) for p in plans), default=None)
    out["plan_tilt_max"] = max((p.get("plan_tilt_max_rad") or 0.0 for p in plans), default=None)
    miss = [p["anchor_miss_m"] for p in plans if p.get("anchor_miss_m") is not None]
    out["anchor_miss_p50"] = float(np.median(miss)) if miss else None
    out["anchor_miss_max"] = float(max(miss)) if miss else None
    out["predicted_frac"] = float(np.mean([bool(p.get("predicted")) for p in plans[1:]])) if len(plans) > 1 else None
    out["guide_boxes_max"] = max(((p.get("guard") or {}).get("guide_boxes") or 0 for p in plans), default=0)
    out["self_masked_max"] = max(((p.get("guard") or {}).get("self_masked_cells") or 0 for p in plans), default=0)
    npz = d / "closed_loop.npz"
    if npz.is_file():
        q = np.load(npz)["realized_qpos"]
        if len(q):
            out["realized_knee_max"] = float(q[:, [7 + 3, 7 + 9]].max())
    return out


def realtime_metrics(d: Path, s):
    plans = json.loads((d / "node_plans.json").read_text()).get("plans", []) if (d / "node_plans.json").is_file() else []
    events = []
    if (d / "node.log").is_file():
        for line in (d / "node.log").read_text().splitlines():
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
    kinds = [e.get("event") for e in events]
    out = dict(outcome=s.get("outcome"), elapsed_s=s.get("elapsed_s"), min_pelvis_m=s.get("minimum_root_height_m"),
               max_contact_n=s.get("maximum_obstacle_contact_n"), max_tilt_rad=s.get("maximum_root_tilt_rad"),
               final_dist_m=s.get("final_goal_distance_xy_m"), plans=len(plans),
               latency_p95_s=(s.get("plan_latency_s") or {}).get("p95"))
    out["first_rule"] = plans[0]["rule"] if plans else None
    out["first_no_safe"] = bool(plans and plans[0].get("no_safe"))
    out["no_safe_plans"] = sum(bool(p.get("no_safe")) for p in plans)
    out["fallback_go"] = sum(p["rule"] == "fallback_go_max_clearance" for p in plans)
    out["calm_stand"] = sum(p["rule"] == "calm_stand" for p in plans)
    out["calm_stop"] = sum(p["rule"] == "calm_stop" for p in plans)
    out["envelope_reselect"] = sum("+envelope" in p["rule"] for p in plans)
    out["dropped"] = kinds.count("plan_dropped")
    out["anchor_miss_events"] = kinds.count("anchor_miss")
    out["estop"] = "estop" in kinds
    out["anchor_miss_max"] = max((e["miss_m"] for e in events if e.get("event") == "anchor_miss"), default=None)
    return out


def summarize(root: Path, conditions, names, seeds):
    rows, table = {}, []
    for c in conditions:
        eps = {}
        for n in names:
            for s in seeds:
                d = root / c / f"{n}__s{s}"
                if (d / "score.json").is_file():
                    eps[f"{n}__s{s}"] = episode_metrics(d)
        rows[c] = eps
        if not eps:
            continue
        unmeasured = sum(e["outcome"] in UNMEASURED for e in eps.values())
        v = [e for e in eps.values() if e["outcome"] not in UNMEASURED]  # the rig failed, not the policy
        if not v:
            table.append(dict(condition=c, n=0, unmeasured=unmeasured))
            continue
        agg = lambda k, f: f([e[k] for e in v if e.get(k) is not None]) if any(e.get(k) is not None for e in v) else None
        table.append(dict(condition=c, n=len(v), unmeasured=unmeasured, success=sum(e["outcome"] == "goal_completion" for e in v),
                          fall=sum(e["outcome"] == "fall" for e in v), collision=sum(e["outcome"] == "obstacle_collision" for e in v),
                          timeout=sum(e["outcome"] in ("timeout", "never_streamed") for e in v),
                          first_no_safe=sum(e["first_no_safe"] for e in v),
                          fallback_go=agg("fallback_go", sum), calm_stand=agg("calm_stand", sum), calm_stop=agg("calm_stop", sum),
                          reselect=agg("envelope_reselect", sum), dropped=agg("dropped", sum),
                          min_pelvis=agg("min_pelvis_m", min),
                          knee_max=agg("realized_knee_max", max), lat_p95=agg("latency_p95_s", max),
                          miss_max=agg("anchor_miss_max", max)))
    return rows, table


def print_table(table, rows):
    cols = ["condition", "n", "unmeasured", "success", "fall", "collision", "timeout", "first_no_safe", "fallback_go", "calm_stand",
            "calm_stop", "reselect", "dropped", "min_pelvis", "knee_max", "lat_p95", "miss_max"]
    print(" ".join(f"{c[:12]:>12}" for c in cols))
    for r in table:
        print(" ".join(f"{(round(r.get(c), 3) if isinstance(r.get(c), float) else r.get(c))!s:>12}" for c in cols))
    print()
    for c, eps in rows.items():
        bad = {k: (e["outcome"], e["first_rule"]) for k, e in eps.items() if e["outcome"] != "goal_completion"}
        if bad:
            print(f"{c}: not completed: {bad}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", default="tag,kit,kit-guards,leaky-guards")
    ap.add_argument("--courses", default=",".join(courses()))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prescan-s", type=float, default=15.0)
    ap.add_argument("--latency-s", type=float, default=0.3)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="sim/realtime/rehearse.py per run (sequential)")
    args = ap.parse_args(argv)
    root = Path(args.out)
    conditions = args.conditions.split(",")
    names = args.courses.split(",")
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    unknown = [c for c in conditions if c not in CONDITIONS] + [n for n in names if n not in courses()]
    if unknown:
        raise SystemExit(f"unknown: {unknown}")
    if not args.summary_only:
        tasks = write_tasks(root / "tasks")
        jobs = [dict(condition=c, task=str(tasks[n]), seed=s, out=str(root / c / f"{n}__s{s}"), prescan_s=args.prescan_s,
                     latency_s=args.latency_s) for s in seeds for n in names for c in conditions]
        import multiprocessing as mp

        t0 = time.time()
        if args.realtime:
            results = map(_run_realtime, jobs)
        else:
            pool = mp.get_context("spawn").Pool(args.workers, maxtasksperchild=12)
            results = pool.imap_unordered(_run, jobs)
        for k, job in enumerate(results, 1):
            print(f"[{k}/{len(jobs)} {time.time() - t0:.0f}s] {job['condition']} {Path(job['out']).name} "
                  f"{job.get('wall_s', 0):.0f}s", flush=True)
        if not args.realtime:
            pool.close()
            pool.join()
    rows, table = summarize(root, conditions, names, seeds)
    (root / "summary.json").write_text(json.dumps(dict(conditions={c: CONDITIONS[c] for c in conditions}, table=table,
                                                        episodes=rows, prescan_s=args.prescan_s,
                                                        latency_s=None if args.realtime else args.latency_s,
                                                        mode="realtime" if args.realtime else "lockstep"),
                                                   indent=1, default=float) + "\n")
    print_table(table, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
