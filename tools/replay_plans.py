#!/usr/bin/env python3
"""Re-plan recorded robot states with the node's planner path, for any bundle (and its deploy rules plugin).

For every chosen plan of a recorded session (`<session>/plans/plan_NNNN_in.npz`, written with --debug-plans: the
exact history and goal the node planned from), the node's lidar memory is rebuilt from the recorded raw MID-360
frames up to that instant, on the node's own grid, each frame deskewed and registered with the pose of its own time
(the node's pelvis position from its status stream and measured plan histories, IMU orientation and joints from
rt/lowstate, the session's mount calibration; memory restarted at the node's start or its last clear_map). The route
is built on it as the node builds it, and `g1deploy.node.plan_job` - the planner workers' own function - plans with
the session's guards. Every row reports how well the rebuild matches the node's own memory at that plan (IoU of
occupied and ceiling cells within 2 m). The bundle is the process's (G1_STUDENT_BUNDLE, plugin G1_DEPLOY_RULES): run
once per bundle and compare the JSON (tools/compare_replays.py). `--rear-blind-deg / --rear-blind-range-m` re-fuse
the frames with that lidar blind (DEPLOY_KIT.md §11).

usage: replay_plans.py SESSION [--every 20] [--plans 0 5 ...] [--impl fused|eager] [--rear-blind-deg 180
                       --rear-blind-range-m 1.0] [--out runs/replays/NAME.json]
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

from g1deploy import bundle  # noqa: E402
from g1deploy.recording import open_session  # noqa: E402

DT = 0.02


def node_events(d):
    out = []
    for line in (d / "node.log").read_text(errors="replace").splitlines():
        _, _, text = line.partition(" ")
        if text.startswith('{"event"'):
            try:
                out.append(json.loads(text))
            except json.JSONDecodeError:
                pass
    return out


def node_frame_state(rec, plan_files):
    """RecordedState in the node's own odometry frame. Leg odometry recomputed offline does not reproduce the node's
    (other samples, other drift: 0.85 vs 1.03 m on the h5-1 walk), so the pelvis position is the node's: its status
    stream (5 Hz) plus the measured 6-frame histories of the plans it anchored on the measured state (lead 0, 20 ms
    apart), interpolated; orientation and joints from rt/lowstate (`--odom leg` uses the raw IMU quaternion).
    Returns (state, number of pose samples)."""
    from tools.debug_plan import RecordedState

    low = rec.stream("lowstate", keys=["q", "dq", "imu_quat"])
    st = [s for s in rec.node_status() if s["status"].get("xy") is not None and s["status"].get("pelvis_z_m") is not None]
    ts = [s["t"] for s in st]
    pos = [[*s["status"]["xy"], s["status"]["pelvis_z_m"]] for s in st]
    for f in plan_files:
        z = np.load(f)
        if "lead" in z.files and int(z["lead"]) != 0:
            continue  # a predicted history, not a measured one
        h = np.asarray(z["hist"], float)
        for i in range(len(h)):
            ts.append(float(z["t"]) - (len(h) - 1 - i) * DT)
            pos.append(h[i, :3].tolist())
    order = np.argsort(ts)
    ts, pos = np.asarray(ts)[order], np.asarray(pos)[order]
    t = low["t"]
    p = np.stack([np.interp(t, ts, pos[:, k]) for k in range(3)], 1)
    return RecordedState(t, np.c_[p, low["imu_quat"].astype(np.float64), low["q"][:, :29].astype(np.float64)],
                         low["dq"][:, :29].astype(np.float64)), len(ts)


class Memory:
    """The node's memory, fused frame by frame up to a time; restarted at the node's start / clear_map times."""

    def __init__(self, rec, state, pcfg, session, starts, rear_blind_deg, rear_blind_range_m, self_margin=None):
        from g1deploy.perception import mid360
        from g1deploy.perception.unitree_lidar import Mount

        self.frames = list(rec.lidar_frames())
        self.state, self.pcfg, self.starts = state, pcfg, sorted(starts)
        mc = (session.get("mount_calib") or {}).get("content")  # the calibration in force that day, not today's file
        self.mount = Mount(np.asarray(mc["pos_torso"], float), mid360.rpy_matrix(tuple(mc["rpy"]))) if mc else Mount.nominal()
        self.blind = (rear_blind_deg, rear_blind_range_m)
        self.self_margin = self_margin
        self.memory, self.i, self.start = None, 0, None

    def _new(self, t0, lo=None):
        from g1deploy.perception.cloud_memory import CloudMemory
        from g1deploy.perception.unitree_lidar import CloudFromSensorFrame

        sensor = dict(self.pcfg.sensor)
        if self.self_margin is not None:
            sensor["self_margin_m"] = self.self_margin
        half = float(sensor.get("half_extent_m", 6.0))
        # the node's own grid (its dumped `lo`): cells line up with the node's memory, so the rebuild can be checked
        origin = np.asarray(lo) + half if lo is not None else np.asarray(self.state.qpos_at(t0)[:2])
        self.memory = CloudMemory(origin, rng=np.random.default_rng(0), **sensor)
        self.memory.latency = 0.0
        self.conv = CloudFromSensorFrame(self.memory.body_filter, self.mount, transport_s=0.01,
                                         rear_blind_deg=self.blind[0], rear_blind_range_m=self.blind[1])
        self.start = t0
        self.i = int(np.searchsorted([f[0] for f in self.frames], t0, side="left"))

    def at(self, t, lo=None):
        start = max([s for s in self.starts if s <= t], default=self.starts[0])
        if self.memory is None or start != self.start:
            self._new(start, lo)
        used = 0
        while self.i < len(self.frames) and self.frames[self.i][0] <= t:
            t_f, pts, t_s = self.frames[self.i]
            self.i += 1
            self.state.until(t_f + 0.05)  # the ring as the node had it when this frame arrived (deskew needs its pose)
            if self.state.qpos_at(t_f) is None:
                continue
            p_w, org = self.conv(pts, t_s, t_f, self.state.qpos_at)
            self.memory.add_cloud(0.0, p_w, org, None)
            self.memory.step(t_f - self.start)
            used += 1
        return self.memory, used


def rebuild_check(memory, z):
    """The rebuilt layers against the node's own (its --debug-plans dump at this plan), within 2 m of the robot:
    IoU of occupied cells (support > 5 cm), IoU of ceiling cells, and the p95 support difference where both agree."""
    if not np.allclose(memory.lo, z["lo"]) or memory.n != int(z["n"]):
        return {"grid": "differs"}
    xy = np.asarray(z["hist"][-1][:2])
    gi = (np.arange(memory.n) * memory.res + memory.lo[0])[:, None]
    gj = (np.arange(memory.n) * memory.res + memory.lo[1])[None, :]
    near = np.hypot(gi - xy[0], gj - xy[1]) <= 2.0
    a_s, b_s = np.nan_to_num(memory.support, nan=0.0), np.nan_to_num(z["support"].astype(float), nan=0.0)
    a_c, b_c = np.isfinite(memory.ceiling) & (memory.ceiling < 1e2), np.isfinite(z["ceiling"]) & (z["ceiling"] < 1e2)
    occ_a, occ_b = (a_s > 0.05) & near, (b_s > 0.05) & near
    iou = lambda p, q: round(float((p & q).sum() / max(1, (p | q).sum())), 3)  # noqa: E731
    both = occ_a & occ_b
    return {"occupied_iou": iou(occ_a, occ_b), "ceiling_iou": iou(a_c & near, b_c & near),
            "occupied": [int(occ_a.sum()), int(occ_b.sum())],
            "support_p95_diff_m": round(float(np.percentile(np.abs(a_s - b_s)[both], 95)), 3) if both.any() else None}


def make_route(flow, rules, route_cfg, memory, xy, goal, guards):
    """The node's route on this memory (node.py perception_loop)."""
    from g1deploy import guards as G

    mem = G.route_memory(memory, guards)
    r = rules.make_route(flow, xy, goal, mem) if rules is not None else flow.route.SensorRoute(xy, goal, mem, **route_cfg)
    r.refresh(xy, 0.0, force=True)
    return r


def summarize(out, hist):
    q = out.get("qpos")
    g = out.get("guard") or {}
    row = {"rule": out.get("rule"), "error": out.get("error"), "no_safe": out.get("no_safe_candidate"),
           "guard": g.get("action"), "rejected": out.get("rejected"), "self_masked": g.get("self_masked_cells"),
           "guide_boxes": g.get("guide_boxes"), "ms": g.get("ms")}
    if q is not None:
        q = np.asarray(q)
        tilt = np.arccos(np.clip(1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2), -1, 1))  # pelvis z vs world z (wxyz quat)
        w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        h0 = np.asarray(hist[-1], float)
        y0 = np.arctan2(2 * (h0[3] * h0[6] + h0[4] * h0[5]), 1 - 2 * (h0[5] ** 2 + h0[6] ** 2))
        fwd = (q[-1, :2] - h0[:2]) @ np.array([np.cos(y0), np.sin(y0)])  # net travel along the starting heading
        v = np.diff(q[:, :2], axis=0) / DT
        v_fwd = v[:, 0] * np.cos(yaw[:-1]) + v[:, 1] * np.sin(yaw[:-1])  # pelvis velocity along its own heading
        row.update(min_pelvis_z=round(float(q[:, 2].min()), 3), max_tilt=round(float(tilt.max()), 3),
                   backward_m=round(float(max(0.0, -fwd)), 3),
                   max_backward_speed=round(float(max(0.0, -v_fwd.min())), 3),
                   advance_m=round(float(np.linalg.norm(q[-1, :2] - np.asarray(hist[-1][:2]))), 3),
                   end_xy=np.round(q[-1, :2], 3).tolist(), chosen_clearance_m=out.get("chosen_clearance_m"))
    if out.get("rules_info") is not None:  # what a deploy rules plugin reports about this plan
        row["rules_info"] = out["rules_info"]
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--plans", type=int, nargs="*", default=None)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--impl", default="fused", choices=("fused", "graph", "eager"))
    ap.add_argument("--guards", default=None, help="guard preset (default: the session's)")
    ap.add_argument("--goal-lookahead", type=float, default=None, help="default: the session's (robot: 1.0)")
    ap.add_argument("--rear-blind-deg", type=float, default=0.0)
    ap.add_argument("--rear-blind-range-m", type=float, default=0.0)
    ap.add_argument("--body", default="dex3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    os.environ.setdefault("G1_BODY_MODEL", args.body)
    import torch

    from g1deploy import guards as G
    from g1deploy.fastplan import build_planner
    from g1deploy.node import memory_view, plan_job

    d = Path(args.session)
    session = json.loads((d / "session.json").read_text())
    events = node_events(d)
    t_up = next(e["t"] for e in events if e["event"] == "node_up")
    starts = [t_up] + [e["t"] for e in events if e["event"] == "map_cleared"]
    files = sorted((d / "plans").glob("plan_*_in.npz"))
    idx = [int(f.name[5:9]) for f in files]
    chosen = args.plans if args.plans is not None else idx[::max(1, args.every)]
    flow = bundle.load_flow()
    cfg = bundle.student_config()
    episode = dict(cfg["episode"], arrival_radius_m=0.4)
    look = args.goal_lookahead or session.get("goal_lookahead")
    if look:
        episode["goal_lookahead_m"] = float(look)
    rules = bundle.load_rules(cfg, episode["goal_lookahead_m"])
    route_cfg = {} if rules is not None else cfg["route_variants"]["sensor"].get("route", {})
    gcfg = G.preset(args.guards or session.get("guards") or "off")
    pdict = dict(cfg["planner"], device="cuda" if torch.cuda.is_available() else "cpu", seed=0)
    pcfg = flow.planner.PlannerConfig(**pdict)
    t0 = time.perf_counter()
    planner = build_planner(pcfg, args.impl, "fp32", rules=rules)
    build_s = time.perf_counter() - t0
    rec = open_session(d)
    state, pose_samples = node_frame_state(rec, files)
    mem = Memory(rec, state, pcfg, session, starts, args.rear_blind_deg, args.rear_blind_range_m)
    live = {p["req_tick"]: p for p in json.loads((d / "node_plans.json").read_text())["plans"]} \
        if (d / "node_plans.json").exists() else {}
    rows, state_rules = [], None
    for n in sorted(chosen):
        z = np.load(d / "plans" / f"plan_{n:04d}_in.npz")
        t, hist, goal = float(z["t"]), [np.asarray(h, np.float64) for h in z["hist"]], np.asarray(z["goal"], float)
        memory, used = mem.at(t, z["lo"])
        xy = np.asarray(hist[-1][:2], float)
        route = make_route(flow, rules, route_cfg, memory, xy, goal, gcfg)
        view = memory_view(memory, True) if gcfg.self_mask else memory_view(memory)
        if rules is not None:
            view = rules.extend_view(view, memory)
        r = __import__("copy").copy(route)
        r.memory = None
        job = {"id": n, "hist": hist, "route": r, "memory": view, "arrival_anchor": hist[-1].copy(),
               "navigation_generation": 1, "measured": hist[-1].copy(), "standing": bool(z["standing"]),
               "allow_drop": bool(z["allow_drop"]), "calm_move_max_m": None, "rules_state": state_rules}
        t1 = time.perf_counter()
        out = plan_job(planner, flow, job, episode, gcfg, rules)
        lat = time.perf_counter() - t1
        if "rules_state" in out:
            state_rules = out["rules_state"]
        row = {"plan": n, "t_s": round(t - t_up, 2), "xy": np.round(xy, 3).tolist(), "frames": used,
               "rebuild": rebuild_check(memory, z) if not (args.rear_blind_deg or args.rear_blind_range_m) else None,
               "route_found": bool(getattr(route, "found", False)), "latency_s": round(lat, 3),
               **summarize(out, hist)}
        lv = live.get(int(z["tick"]))
        if lv is not None:
            row["live"] = {"rule": lv.get("rule"), "no_safe": lv.get("no_safe"), "guard": lv.get("guard"),
                           "rejected": lv.get("rejected")}
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
    lat = np.array([r["latency_s"] for r in rows[1:]] or [np.nan])
    flags = sorted({k for r in rows for k, v in (r.get("rules_info") or {}).items() if isinstance(v, bool)})
    summary = {"session": str(d), "bundle": str(bundle.ROOT), "deploy_rules": os.environ.get("G1_DEPLOY_RULES"),
               "impl": args.impl,
               "guards": gcfg.describe() if gcfg.any() else "off", "goal_lookahead_m": episode["goal_lookahead_m"],
               "rear_blind": [args.rear_blind_deg, args.rear_blind_range_m], "plans": len(rows),
               "pose_samples": pose_samples, "planner_build_s": round(build_s, 1),
               "latency_p50_s": float(np.nanmedian(lat)), "latency_p95_s": float(np.nanpercentile(lat, 95)),
               "no_safe": int(sum(bool(r.get("no_safe")) for r in rows)),
               "rules": {k: sum(r["rule"] == k for r in rows) for k in sorted({str(r["rule"]) for r in rows})},
               "guards_fired": {k: sum(r.get("guard") == k for r in rows) for k in sorted({str(r.get("guard")) for r in rows})},
               "rules_flags": {k: int(sum(bool((r.get("rules_info") or {}).get(k)) for r in rows)) for k in flags},
               "rebuild_occupied_iou_min": min([r["rebuild"]["occupied_iou"] for r in rows
                                                if (r.get("rebuild") or {}).get("occupied_iou") is not None] or [None],
                                               key=lambda v: 9 if v is None else v),
               "advance_p50_m": float(np.nanmedian([r.get("advance_m", np.nan) for r in rows])),
               "min_pelvis_z": float(np.nanmin([r.get("min_pelvis_z", np.nan) for r in rows]))}
    print(json.dumps(summary), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str) + "\n")
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
