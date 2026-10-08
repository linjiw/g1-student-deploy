#!/usr/bin/env python3
"""Run a panel of lock-step MuJoCo episodes and pair them with the Isaac packet they mirror.

The episode list comes from an Isaac packet (default: the held-out read of the deployed student,
`closed-loop-test-v4`): each `<arm>__<scenario>__s<seed>/callback/run_config.json` names the task
file and seed. Every MuJoCo episode reuses that task + seed; the condition flags (latency, start,
route, lidar, odometry noise) are applied uniformly. Episodes run in `--workers` processes
(planner on the GPU, ~0.4 GB each; SONIC + physics on CPU).

usage: eval_panel.py --name g3-base [--packet DIR] [--limit N] [--workers 3] [condition flags]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PACKET = Path("/home/linjiw/flow/artifacts/closed-loop-test-v4")


def episodes(packet: Path, families=None, limit=None):
    out = []
    for name in json.loads((packet / "episodes.json").read_text()):
        rc = json.loads((packet / name / "callback/run_config.json").read_text())
        task = json.loads(Path(rc["task_path"]).read_text())
        if families and task.get("family") not in families and task.get("scenario_id") not in families:
            continue
        isaac = packet / name / "callback/score.json"
        out.append(dict(name=name.split("__", 1)[1], task_path=rc["task_path"], seed=int(rc["seed"]),
                        isaac_outcome=json.loads(isaac.read_text())["outcome"] if isaac.is_file() else None,
                        family=task.get("family") or task.get("scenario_id")))
    return out[:limit] if limit else out


def _run(job, cond, out_dir):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch

    torch.set_num_threads(2)
    from sim.lockstep import run_episode

    d = Path(out_dir) / job["name"]
    if (d / "score.json").is_file():
        return json.loads((d / "score.json").read_text())
    if d.exists():
        import shutil

        shutil.rmtree(d)
    try:
        return run_episode(job["task_path"], job["seed"], d, **cond)
    except Exception as exc:  # recorded in score.json by run_episode
        return {"outcome": "runtime_error", "runtime_error": f"{type(exc).__name__}: {exc}", "task_id": job["name"]}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--worker"]:
        return worker(argv[1], argv[argv.index("--out-dir") + 1])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--packet", default=str(DEFAULT_PACKET))
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--latency-s", type=float, default=0.0)
    ap.add_argument("--anchor", choices=("measured", "predict"), default="measured")
    ap.add_argument("--inflight", type=int, default=1)
    ap.add_argument("--start", choices=("recorded", "stand"), default="recorded")
    ap.add_argument("--route", choices=("privileged", "sensor"), default="privileged")
    ap.add_argument("--lidar", choices=("analytic", "mjray", "mid360"), default="analytic")
    ap.add_argument("--lidar-cfg", default=None)
    ap.add_argument("--odom-noise", default=None)
    ap.add_argument("--prescan-s", type=float, default=None)
    ap.add_argument("--goal-lookahead", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    jobs = episodes(Path(args.packet), args.families, args.limit)
    cond = dict(device=args.device, latency_s=args.latency_s, anchor_mode=args.anchor, inflight=args.inflight,
                start=args.start,
                route_source=args.route,
                lidar=args.lidar, odom_noise=json.loads(args.odom_noise) if args.odom_noise else None,
                prescan_s=args.prescan_s, lidar_cfg=json.loads(args.lidar_cfg) if args.lidar_cfg else None,
                goal_lookahead=args.goal_lookahead)
    out_dir = ROOT / "runs" / args.name
    print(f"{len(jobs)} episodes -> {out_dir}  condition {cond}  workers {args.workers}", flush=True)
    if args.dry_run:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "condition.json").write_text(json.dumps(dict(cond, packet=args.packet, jobs=jobs), indent=1) + "\n")
    t0 = time.time()
    # One plain subprocess per worker, each running every n-th episode and exiting hard: process pools left
    # orphaned workers holding CUDA contexts (and the caller's stdout pipe) after the parent finished.
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", f"{k}/{args.workers}", "--out-dir", str(out_dir)],
                              cwd=str(ROOT)) for k in range(args.workers)]
    done = set()
    while True:
        alive = [p for p in procs if p.poll() is None]
        for j in jobs:
            if j["name"] not in done and (out_dir / j["name"] / "score.json").is_file():
                done.add(j["name"])
                r = json.loads((out_dir / j["name"] / "score.json").read_text())
                print(f"[{len(done)}/{len(jobs)} {time.time() - t0:5.0f}s] {j['name']:28s} isaac={str(j['isaac_outcome']):20s} "
                      f"mujoco={r.get('outcome')}", flush=True)
        if not alive:
            break
        time.sleep(2)
    rows = []
    for j in jobs:
        f = out_dir / j["name"] / "score.json"
        r = json.loads(f.read_text()) if f.is_file() else {"outcome": "missing"}
        rows.append(dict(name=j["name"], family=j["family"], seed=j["seed"], isaac=j["isaac_outcome"],
                         mujoco=r.get("outcome"), elapsed_s=r.get("elapsed_s"), error=r.get("runtime_error")))
    (out_dir / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    from tools.compare import summarize

    print(summarize(rows, title=args.name), flush=True)
    return 0


def worker(spec, out_dir):
    k, n = map(int, spec.split("/"))
    c = json.loads((Path(out_dir) / "condition.json").read_text())
    jobs, cond = c.pop("jobs"), {key: v for key, v in c.items() if key != "packet"}
    for i, job in enumerate(jobs):
        if i % n == k:
            _run(job, cond, out_dir)
    sys.stdout.flush()
    os._exit(0)  # skip CUDA / onnxruntime teardown, which can hang


if __name__ == "__main__":
    sys.exit(main())
