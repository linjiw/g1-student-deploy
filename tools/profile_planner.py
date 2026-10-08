#!/usr/bin/env python3
"""Where does a plan's time go? Stage wall times (wrapped methods) + CUDA kernel time vs wall (torch.profiler).

usage: profile_planner.py [--n 10] [--fast]     (--fast profiles g1deploy.fastplan.FastFlowPlanner)
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.bench_planner import EPISODE  # noqa: E402


def setup(fast=False, device="cuda", seed=0, **over):
    from g1deploy import bundle
    from sim.lockstep import make_goal_fn

    flow = bundle.load_flow()
    cfg = bundle.student_config()
    task = json.loads((EPISODE / "task.json").read_text())
    pre = np.load(EPISODE / "closed_loop.npz")["pre_action_qpos"]
    pc = flow.planner.PlannerConfig(**dict(cfg["planner"], device=device, seed=seed, **over))
    if fast:
        from g1deploy.fastplan import FastFlowPlanner

        planner = FastFlowPlanner(pc)
    else:
        planner = flow.planner.FlowPlanner(pc)
    sm = flow.sensor_map
    mem = sm.RayCastMemory(task["scene_boxes_world"], pre[0][:2], rng=np.random.default_rng(0), **pc.sensor)
    for k in range(60):
        mem.step(k * 0.02, pre[k][:3], pre[k][3:7])
    route = flow.route.Route(pre[0][:2], np.asarray(task["goal_world_xy"]), task["scene_boxes_world"])
    goal_fn = make_goal_fn(route, 1.5, "bearing_near", 0.5, None, flow.callback.goal_vector,
                           flow.callback.route_goal_heading)
    sel = sm.HeightFieldSelector(mem, planner.certifier.tree, planner.certifier.bodies, planner.shapes)

    def map_fn(xy, yaw):
        return mem.ego_map(xy, yaw, pc.datum_offset_m, 0.8)[0]

    gb = sm.memory_boxes(mem, grow_overhead_m=pc.guide_grow_overhead_m)
    hist = np.asarray(pre[54:60].astype(np.float32))
    return planner, (hist, [], goal_fn, map_fn, sel), dict(guide_boxes=gb), mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    import torch

    planner, a, kw, mem = setup(args.fast)
    if args.fast:
        planner.warmup()
    timers = collections.defaultdict(float)
    targets = {"_candidates": planner, "_condition": planner, "_sample_guided": planner, "_clearance": planner}
    import motion2scene_flow.closed_loop.grounding as G
    for name, obj in targets.items():
        f = getattr(obj, name)

        def wrap(*x, _f=f, _n=name, **y):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = _f(*x, **y)
            torch.cuda.synchronize()
            timers[_n] += time.perf_counter() - t
            return r

        setattr(obj, name, wrap)
    g0 = G.ground_legs

    def gwrap(*x, **y):
        t = time.perf_counter()
        r = g0(*x, **y)
        timers["ground_legs"] += time.perf_counter() - t
        return r

    G.ground_legs = gwrap
    planner.plan(*a, **kw)
    timers.clear()
    t0 = time.perf_counter()
    for _ in range(args.n):
        planner.plan(*a, **kw)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / args.n
    print(f"plan wall {wall * 1000:.1f} ms")
    for k_, v in sorted(timers.items(), key=lambda x: -x[1]):
        print(f"  {k_:16s} {v / args.n * 1000:7.1f} ms")
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        planner.plan(*a, **kw)
        torch.cuda.synchronize()
    ev = prof.key_averages()
    cuda_us = sum(e.self_device_time_total for e in ev)
    launches = sum(e.count for e in ev if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"))
    syncs = sum(e.count for e in ev if "Synchronize" in e.key or e.key in ("cudaMemcpyAsync", "cudaStreamSynchronize"))
    print(f"CUDA kernel time {cuda_us / 1000:.1f} ms per plan, kernel launches {launches}, memcpy/sync calls {syncs}")


if __name__ == "__main__":
    main()
