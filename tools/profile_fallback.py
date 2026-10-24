#!/usr/bin/env python3
"""Where does a no-safe plan's time go? Stage timings of the deployed planner on recorded node inputs.

When no candidate of the first K is safe, the frozen FlowPlanner.plan (fallback "resample_stop") samples 3K more and
then K stop candidates, and picks the max-clearance one: the slowest plan the node issues (0.42 s on the deploy
PC's RTX 4070 Ti SUPER in real time, over the 0.32 s gate; docs/ROBUSTNESS.md). This tool replays exact node inputs
(`--debug-plans` dumps: <session>/plans/plan_NNNN_in.npz) through g1deploy.fastplan.build_planner, as the node's
worker builds it, twice per state:

  as deployed        the state's own outcome (usually the K go candidates are enough)
  forced fallback    select_margin_m above every selector margin, so no pool is "safe": K go + 3K resample + K stop

and splits each plan, with a CUDA sync around every stage, into
  go / resample / stop   FlowPlanner._candidates per pool (conditioning, guided sampling, host transfer; the fused
                         planner also checks earlier chunks' height-field hits while later chunks sample)
  clearance              FlowPlanner._clearance per pool (the height-field selector)
  grounding              leg grounding of the chosen candidate (and any re-materialised one)
  rest                   selection, materialize's other steps, the deployment's tie-break (planner_compat)

The frozen code is not changed: the stages are timed by wrapping the planner instance's methods. The syncs remove
overlap between stages, so `sum of stages` can exceed an unwrapped plan; the unwrapped wall time is reported too.

usage: profile_fallback.py [SESSION ...] [--plans 0 3] [--n 10] [--impl fused] [--out runs/profile/x.json]
       (default: the lab-site twin sessions under runs/robust with dumps, the 4 states with the most guide boxes)
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import bundle  # noqa: E402

STAGES = ("go", "resample", "stop", "clearance", "grounding", "rest")


def view_from(sm, z):
    v = sm.SensorMapMemory.__new__(sm.SensorMapMemory)
    v.support, v.ceiling = z["support"].astype(np.float64), z["ceiling"].astype(np.float64)
    v.lo, v.res, v.n = z["lo"], float(z["res"]), int(z["n"])
    return v


class Stages:
    """Wrap one planner instance's stage methods with synchronized timers."""

    def __init__(self, planner, sync):
        self.planner, self.sync, self.cur = planner, sync, None
        self._cand, self._clear = planner._candidates, planner._clearance

    def _timed(self, fn, label):
        def run(*a, **kw):
            self.sync()
            t = time.perf_counter()
            out = fn(*a, **kw)
            self.sync()
            name = label() if callable(label) else label
            self.cur[name] = self.cur.get(name, 0.0) + time.perf_counter() - t
            return out
        return run

    def _pool_label(self):
        n = self.cur.setdefault("_pools", 0)
        self.cur["_pools"] = n + 1
        return ("go", "resample", "stop")[min(n, 2)]

    @contextlib.contextmanager
    def on(self):
        import g1deploy.fastplan as fp

        self.cur = {}
        ground = fp.ground_legs
        self.planner._candidates = self._timed(self._cand, self._pool_label)
        self.planner._clearance = self._timed(self._clear, "clearance")
        fp.ground_legs = self._timed(ground, "grounding")  # FastFlowPlanner.plan patches this into grounding
        try:
            yield self.cur
        finally:
            fp.ground_legs = ground
            del self.planner._candidates, self.planner._clearance


def default_states(k=4):
    """(session, plan index, guide boxes) for the recorded states with the most guide boxes."""
    out = []
    for f in sorted(glob.glob(str(ROOT / "runs/robust/*/plans/plan_*_in.npz")) +
                    glob.glob(str(ROOT / "runs/debug/*/plans/plan_*_in.npz"))):
        z = np.load(f)
        out.append((Path(f).parents[1], int(Path(f).name[5:9]), int((np.isfinite(z["support"]) | np.isfinite(z["ceiling"])).sum())))
    out.sort(key=lambda s: -s[2])
    seen, pick = set(), []
    for s in out:  # spread over sessions
        if s[0] not in seen:
            pick.append(s)
            seen.add(s[0])
        if len(pick) == k:
            break
    return [(s[0], s[1]) for s in pick]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--plans", type=int, nargs="*", default=[0])
    ap.add_argument("--n", type=int, default=10, help="timed plans per state and mode (after one warm-up)")
    ap.add_argument("--impl", default="fused", choices=("fused", "graph", "eager"))
    ap.add_argument("--precision", default="fp32", choices=("fp32", "tf32", "bf16"))
    ap.add_argument("--body", default="dex3")
    ap.add_argument("--out", default=str(ROOT / "runs/profile/fallback.json"))
    args = ap.parse_args(argv)
    os.environ.setdefault("G1_BODY_MODEL", args.body)
    import torch

    flow = bundle.load_flow()
    sm, cb = flow.sensor_map, flow.callback
    cfg = bundle.student_config()
    episode = dict(cfg["episode"], arrival_radius_m=0.4)
    route_cfg = cfg["route_variants"]["sensor"].get("route", {})
    pdict = dict(cfg["planner"], device="cuda" if torch.cuda.is_available() else "cpu")
    from g1deploy.fastplan import build_planner
    from sim.lockstep import make_goal_fn

    sync = torch.cuda.synchronize if pdict["device"] == "cuda" else (lambda: None)
    t0 = time.perf_counter()
    planners = {"deployed": build_planner(flow.planner.PlannerConfig(**pdict), args.impl, args.precision),
                # above the selector's largest margin: every pool is "unsafe" (tools/bench_planner.py --force-fallback)
                "forced": build_planner(flow.planner.PlannerConfig(**dict(pdict, select_margin_m=9.0)), args.impl,
                                        args.precision)}
    setup_s = time.perf_counter() - t0
    states = [(Path(s), p) for s in args.sessions for p in args.plans] if args.sessions else default_states()
    records = []
    for d, n in states:
        z = np.load(d / "plans" / f"plan_{n:04d}_in.npz")
        view, hist, goal = view_from(sm, z), np.asarray(z["hist"]), z["goal"]
        xy = np.asarray(hist[-1][:2], float)
        route = flow.route.SensorRoute(xy, goal, view, **route_cfg)
        route.refresh(xy, 1.0, force=True)
        boxes = sm.memory_boxes(view, grow_overhead_m=planners["deployed"].cfg.guide_grow_overhead_m)
        for mode, planner in planners.items():
            pc = planner.cfg
            sel = sm.HeightFieldSelector(view, planner.certifier.tree, planner.certifier.bodies, planner.shapes)

            def map_fn(xy_, yaw_, _v=view, _pc=pc):
                return _v.ego_map(xy_, yaw_, _pc.datum_offset_m, _pc.support_clip_m or 0.8)[0]

            def call():
                goal_fn = make_goal_fn(route, episode["goal_lookahead_m"], episode["goal_heading"],
                                       episode["steer_lookahead_m"], None, cb.goal_vector, cb.route_goal_heading,
                                       arrival_radius_m=episode["arrival_radius_m"], arrival_scope=1)
                goal_fn.arrival_anchor = hist[-1].copy()
                return planner.plan(hist, [], goal_fn, map_fn, sel, prev=None, guide_boxes=boxes)

            call()  # warm: graphs for this (K, bucket) exist after the first plan
            stages = Stages(planner, sync)
            walls, splits, rules = [], [], []
            for _ in range(args.n):
                sync()
                t = time.perf_counter()
                out = call()
                sync()
                walls.append(time.perf_counter() - t)
                with stages.on() as cur:
                    t = time.perf_counter()
                    call()
                    sync()
                    total = time.perf_counter() - t
                pools = cur.pop("_pools", 0)
                cur["rest"] = total - sum(cur.values())
                cur["total"] = total
                splits.append(dict(cur, pools=pools))
                rules.append(out["rule"])
            rec = {"session": d.name, "plan": n, "mode": mode, "guide_boxes": len(boxes), "rules": sorted(set(rules)),
                   "pools": splits[-1]["pools"],
                   "wall_s": {"p50": round(float(np.median(walls)), 4), "max": round(float(np.max(walls)), 4)},
                   "stages_ms": {k: round(1e3 * float(np.median([s.get(k, 0.0) for s in splits])), 1)
                                 for k in (*STAGES, "total")}}
            records.append(rec)
            print(json.dumps(rec), flush=True)
    summary = {"gpu": torch.cuda.get_device_name() if pdict["device"] == "cuda" else None, "impl": args.impl,
               "precision": args.precision, "K": planners["deployed"].cfg.candidates, "setup_s": round(setup_s, 1),
               "n": args.n, "records": records}
    for mode in planners:
        rs = [r for r in records if r["mode"] == mode]
        summary[f"{mode}_median_ms"] = {k: round(float(np.median([r["stages_ms"][k] for r in rs])), 1)
                                        for k in (*STAGES, "total")}
        summary[f"{mode}_wall_p50_s"] = round(float(np.median([r["wall_s"]["p50"] for r in rs])), 4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    for mode in planners:
        m = summary[f"{mode}_median_ms"]
        print(f"{mode:>9}: wall p50 {summary[f'{mode}_wall_p50_s'] * 1e3:.0f} ms | " +
              " ".join(f"{k} {m[k]:.0f}" for k in (*STAGES, "total")) + " (ms, synced)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
