#!/usr/bin/env python3
"""Hardware ladder H0: measure the student planner's latency on THIS machine (the one that will run the node).

Replays a recorded beam episode's history + lidar memory and times `planner.plan` (the exact deployed config:
K=16, 10 ODE steps, 2 chunks, guidance, sensor selection, leg grounding). Compares the result with the
simulated latency arms (docs/RUNBOOK.md): with predictive anchoring, 0.32 s cost ~4/98 episodes vs lock-step.

usage: bench_planner.py [--device cuda|cpu] [--n 20]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1deploy import bundle  # noqa: E402

EPISODE = Path("/home/linjiw/flow/artifacts/closed-loop-test-v4/student-guide10m8-grow06__beam-100__s8871/callback")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--episode", default=str(EPISODE))
    args = ap.parse_args(argv)
    import torch

    flow = bundle.load_flow()
    cfg = bundle.student_config()
    ep = Path(args.episode)
    task = json.loads((ep / "task.json").read_text())
    pre = np.load(ep / "closed_loop.npz")["pre_action_qpos"]
    pc = flow.planner.PlannerConfig(**dict(cfg["planner"], device=args.device, seed=0))
    planner = flow.planner.FlowPlanner(pc)
    sm = flow.sensor_map
    mem = sm.RayCastMemory(task["scene_boxes_world"], pre[0][:2], rng=np.random.default_rng(0), **pc.sensor)
    for k in range(60):
        mem.step(k * 0.02, pre[k][:3], pre[k][3:7])
    from sim.lockstep import make_goal_fn

    route = flow.route.Route(pre[0][:2], np.asarray(task["goal_world_xy"]), task["scene_boxes_world"])
    goal_fn = make_goal_fn(route, 1.5, "bearing_near", 0.5, None, flow.callback.goal_vector,
                           flow.callback.route_goal_heading)
    sel = sm.HeightFieldSelector(mem, planner.certifier.tree, planner.certifier.bodies, planner.shapes)

    def map_fn(xy, yaw):
        return mem.ego_map(xy, yaw, pc.datum_offset_m, 0.8)[0]

    gb = sm.memory_boxes(mem, grow_overhead_m=pc.guide_grow_overhead_m)
    hist = list(pre[54:60].astype(np.float32))
    sync = torch.cuda.synchronize if args.device.startswith("cuda") else (lambda: None)
    planner.plan(np.asarray(hist), [], goal_fn, map_fn, sel, guide_boxes=gb)
    ts = []
    for _ in range(args.n):
        t = time.perf_counter()
        planner.plan(np.asarray(hist), [], goal_fn, map_fn, sel, guide_boxes=gb)
        sync()
        ts.append(time.perf_counter() - t)
    p50, p95 = float(np.median(ts)), float(np.quantile(ts, 0.95))
    verdict = ("OK: within the 0.32 s tested with predictive anchoring" if p95 <= 0.32 else
               "MARGINAL: test this latency in sim (eval_panel --latency-s <p95> --anchor predict)" if p95 <= 0.5 else
               "TOO SLOW: expect large losses; use a faster GPU or reduce planner compute")
    print(json.dumps({"device": args.device, "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
                      "plan_s_p50": round(p50, 3), "plan_s_p95": round(p95, 3), "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
