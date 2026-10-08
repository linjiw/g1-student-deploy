#!/usr/bin/env python3
"""Hardware ladder H0: measure the student planner's latency on THIS machine (the one that will run the node).

Replays recorded beam-episode histories + lidar memory and times `planner.plan` with the deployed config
(K=16, 10 ODE steps, 2 chunks, guidance, sensor selection, leg grounding). Implementations
(g1deploy/fastplan.py): `eager` = the bundle's FlowPlanner as evaluated; `graph` = CUDA-graph replay +
vectorised geometry (bitwise-equal kernels); `fused` (deployed default) = graph + per-chunk model
constants + compiled step pieces. Several `--impl`s are timed interleaved (same GPU load) and, from the
same random state, checked for plan parity (chosen candidate, rule, max joint difference).

`--sequence` replans every 0.5 s along several recorded episodes (beam, pillars, corridor, slalom,
detour), so the latency distribution includes resampling (no safe candidate -> 3K more) and the obstacle
counts a run really sees.

usage: bench_planner.py [--impl fused] [--impl eager --impl fused] [--sequence] [--n 20] [--device cuda]
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

FLOW_EPISODES = Path("/home/linjiw/flow/artifacts/closed-loop-test-v4")
EPISODE = FLOW_EPISODES / "student-guide10m8-grow06__beam-100__s8871/callback"
SEQUENCE = ("beam-100__s8871", "pillars-101__s8871", "corridor_beam-103__s8871", "slalom-101__s8871",
            "detour-103__s8871", "beams2-105__s8871")
GATE_S = 0.32  # tested with predictive anchoring; the success cliff starts right after it (READINESS.md)


def make(flow, cfg, impl, device, seed, over):
    pc = flow.planner.PlannerConfig(**dict(cfg["planner"], device=device, seed=seed, **over))
    if impl == "eager":
        return flow.planner.FlowPlanner(pc)
    from g1deploy.fastplan import fast_planner_class

    planner = fast_planner_class()(pc, graphs=True, fuse=(impl == "fused"))
    if device.startswith("cuda"):
        planner.warmup()
    return planner


def episode_states(flow, cfg, ep, every, device):
    """(name, args, kwargs) for plan() at every `every` ticks of a recorded episode."""
    from g1deploy.node import memory_view
    from sim.lockstep import make_goal_fn

    sm = flow.sensor_map
    pc = flow.planner.PlannerConfig(**dict(cfg["planner"], device=device))
    task = json.loads((ep / "task.json").read_text())
    pre = np.load(ep / "closed_loop.npz")["pre_action_qpos"]
    mem = sm.RayCastMemory(task["scene_boxes_world"], pre[0][:2], rng=np.random.default_rng(0), **pc.sensor)
    route = flow.route.Route(pre[0][:2], np.asarray(task["goal_world_xy"]), task["scene_boxes_world"])
    goal_fn = make_goal_fn(route, 1.5, "bearing_near", 0.5, None, flow.callback.goal_vector, flow.callback.route_goal_heading)
    out = []
    for k in range(len(pre)):
        mem.step(k * 0.02, pre[k][:3], pre[k][3:7])
        if k >= 60 and (k - 60) % every == 0:
            view = memory_view(mem)  # what the node sends its planner workers
            out.append((f"{ep.parent.name.split('__', 1)[1]}@{k}", view, pre[k - 5:k + 1].astype(np.float32), goal_fn))
    return out


def plan_args(flow, planner, view, hist, goal_fn):
    sm, pc = flow.sensor_map, planner.cfg
    sel = sm.HeightFieldSelector(view, planner.certifier.tree, planner.certifier.bodies, planner.shapes)

    def map_fn(xy, yaw):
        return view.ego_map(xy, yaw, pc.datum_offset_m, pc.support_clip_m or 0.8)[0]

    gb = sm.memory_boxes(view, grow_overhead_m=pc.guide_grow_overhead_m)
    return (np.asarray(hist), [], goal_fn, map_fn, sel), {"guide_boxes": gb}


def stats(ts):
    ts = np.asarray(ts)
    return {"n": int(len(ts)), "p50": round(float(np.median(ts)), 3), "p95": round(float(np.quantile(ts, 0.95)), 3),
            "p99": round(float(np.quantile(ts, 0.99)), 3), "max": round(float(ts.max()), 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--impl", action="append", choices=("eager", "graph", "fused"),
                    help="planner implementation(s); repeat to compare interleaved (default: fused)")
    ap.add_argument("--n", type=int, default=20, help="timed plans per implementation (fixed state)")
    ap.add_argument("--sequence", action="store_true", help="replan along recorded episodes instead of one state")
    ap.add_argument("--every", type=int, default=25, help="--sequence: ticks between replans (50 Hz)")
    ap.add_argument("--episode", default=str(EPISODE))
    ap.add_argument("--candidates", type=int, default=None, help="override K (deployed: 16)")
    ap.add_argument("--ode-steps", type=int, default=None, help="override ODE steps (deployed: 10)")
    ap.add_argument("--force-fallback", action="store_true",
                    help="worst case: no candidate counts as safe -> 3K resampled + K stop candidates per plan")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (numerics change; not validated)")
    ap.add_argument("--out", default=None, help="write per-plan records (JSON) here")
    args = ap.parse_args(argv)
    import torch

    if args.tf32:
        torch.set_float32_matmul_precision("high")
    impls = args.impl or ["fused"]
    if args.device == "cpu":
        impls = ["eager" if i == "eager" else "graph" for i in impls]  # no CUDA graphs / compile on CPU
    flow = bundle.load_flow()
    cfg = bundle.student_config()
    over = {k: v for k, v in (("candidates", args.candidates), ("ode_steps", args.ode_steps)) if v}
    if args.force_fallback:
        over["select_margin_m"] = 9.0  # above the selector's largest margin: every pool is "unsafe"
    t0 = time.perf_counter()
    planners = {i: make(flow, cfg, i, args.device, 0, over) for i in impls}
    setup_s = time.perf_counter() - t0
    if args.sequence:
        states = []
        for name in SEQUENCE:
            ep = FLOW_EPISODES / f"student-guide10m8-grow06__{name}/callback"
            if ep.exists():
                states += episode_states(flow, cfg, ep, args.every, args.device)
    else:
        ep = Path(args.episode)
        task = json.loads((ep / "task.json").read_text())
        pre = np.load(ep / "closed_loop.npz")["pre_action_qpos"]
        sm = flow.sensor_map
        pc0 = next(iter(planners.values())).cfg
        mem = sm.RayCastMemory(task["scene_boxes_world"], pre[0][:2], rng=np.random.default_rng(0), **pc0.sensor)
        for k in range(60):
            mem.step(k * 0.02, pre[k][:3], pre[k][3:7])
        from sim.lockstep import make_goal_fn

        route = flow.route.Route(pre[0][:2], np.asarray(task["goal_world_xy"]), task["scene_boxes_world"])
        goal_fn = make_goal_fn(route, 1.5, "bearing_near", 0.5, None, flow.callback.goal_vector,
                               flow.callback.route_goal_heading)
        states = [("fixed", mem, pre[54:60].astype(np.float32), goal_fn)] * args.n
    sync = torch.cuda.synchronize if args.device.startswith("cuda") else (lambda: None)
    first = next(iter(planners.values()))
    for p in planners.values():  # warm caches
        a, kw = plan_args(flow, p, *states[0][1:])
        p.plan(*a, **kw)
    times = {i: [] for i in impls}
    records = []
    for si, (name, view, hist, goal_fn) in enumerate(states):
        gstate = first.generator.get_state()
        outs = {}
        order = impls if si % 2 == 0 else impls[::-1]  # alternate order: neither side always runs second
        for i in order:
            p = planners[i]
            p.generator.set_state(gstate)  # same random stream -> plans comparable
            a, kw = plan_args(flow, p, view, hist, goal_fn)
            sync()
            t = time.perf_counter()
            outs[i] = p.plan(*a, **kw)
            sync()
            times[i].append(time.perf_counter() - t)
        rec = {"state": name, **{f"t_{i}": round(times[i][-1], 4) for i in impls},
               **{f"rule_{i}": outs[i]["rule"] for i in impls}}
        if len(impls) > 1:
            r = outs[impls[0]]
            for i in impls[1:]:
                rec[f"same_choice_{i}"] = bool(outs[i]["choice"] == r["choice"] and outs[i]["rule"] == r["rule"])
                rec[f"qpos_maxdiff_{i}"] = float(np.abs(outs[i]["qpos"] - r["qpos"]).max())
        records.append(rec)
    summary = {"force_fallback": bool(args.force_fallback), "device": args.device, "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
               "K": first.cfg.candidates, "ode_steps": first.cfg.ode_steps, "states": "sequence" if args.sequence else "fixed",
               "setup_s": round(setup_s, 1), "latency_s": {i: stats(times[i]) for i in impls}}
    if args.sequence:
        summary["resampled_plans"] = int(sum(r[f"rule_{impls[0]}"] != "safe_min_goal_distance" for r in records))
    if len(impls) > 1:
        summary["parity_vs_" + impls[0]] = {
            i: {"same_choice_and_rule": f"{sum(r[f'same_choice_{i}'] for r in records)}/{len(records)}",
                "qpos_maxdiff_median": float(np.median([r[f'qpos_maxdiff_{i}'] for r in records])),
                "qpos_maxdiff_max": float(np.max([r[f'qpos_maxdiff_{i}'] for r in records]))} for i in impls[1:]}
    deployed = impls[-1]
    p95 = summary["latency_s"][deployed]["p95"]
    summary["verdict"] = (f"OK: {deployed} p95 {p95} s within the {GATE_S} s tested with predictive anchoring" if p95 <= GATE_S else
                          f"MARGINAL: {deployed} p95 {p95} s; test it in sim (eval_panel --latency-s {p95} --anchor predict)"
                          if p95 <= 0.5 else "TOO SLOW: expect large losses; free the GPU or reduce planner compute")
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
