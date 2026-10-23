#!/usr/bin/env python3
"""Why did the student choose that? Re-run the planner on the exact state of a recorded plan and draw its options.

Inputs, best first:
  1. <session>/plans/plan_NNNN_in.npz + _out.npz, written by the node with `g1 sim|robot --debug-plans`: the exact
     history, lidar memory, goal and route the node handed its planner, and the plan it got back;
  2. otherwise a rebuild from the recording: the robot state ring (rec/lowstate + rt/odostate in simulation, node
     status otherwise) and the node's memory re-fused from every recorded raw MID-360 frame (needs rec/lidar).
Then it calls the deployed planner (g1deploy.fastplan.build_planner, like node.py's worker) and reads its candidate
pools (FlowPlanner.last_pools): every candidate's pelvis path, height and clearance, its pool (go / resample / stop),
which were safe, which was chosen; the live plan (from the dump) is drawn on top.

Writes <session>/debug/plan_NNN_sS.png (top view, pelvis heights, the network's ego map) and debug/plans.json.

usage: debug_plan.py SESSION [--plans 0 1 2] [--seeds 1] [--impl fused|eager]
The planner's random stream differs from the live run (the node's worker had drawn for every earlier plan); --seeds N
repeats each plan with N seeds to separate what is systematic from what is chance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import bundle  # noqa: E402
from g1deploy.recording import open_session  # noqa: E402

DT = 0.02
COLORS = {"go": "tab:green", "resample": "tab:blue", "stop": "tab:purple", "continue": "tab:orange"}


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


class RecordedState:
    """The node's DdsState ring, filled from the recording (same interpolation code)."""

    def __init__(self, t, qpos36, dq):
        from g1deploy.node import DdsState

        self._t, self._q, self._dq = t, qpos36, dq
        self.lock = threading.Lock()
        self.ring = deque(maxlen=1000)
        self.history = DdsState.history.__get__(self)
        self.qpos_at = DdsState.qpos_at.__get__(self)

    def until(self, t_end):
        j = int(np.searchsorted(self._t, t_end, side="right"))
        self.ring = deque(((self._t[i], self._q[i], self._dq[i]) for i in range(max(0, j - 1000), j)), maxlen=1000)


def recorded_state(rec):
    low = rec.stream("lowstate", keys=["q", "dq", "imu_quat"])
    t = low["t"]
    if "odostate" in rec.streams():
        odo = rec.stream("odostate", keys=["position", "orientation"])
        j = np.clip(np.searchsorted(odo["t"], t), 0, len(odo["t"]) - 1)
        pos, quat = odo["position"][j].astype(np.float64), odo["orientation"][j].astype(np.float64)  # sim: wxyz
    else:
        st = rec.node_status()
        ts = np.array([s["t"] for s in st])
        xy = np.array([s["status"].get("xy") or [np.nan, np.nan] for s in st], float)
        z = np.array([s["status"].get("pelvis_z_m", np.nan) for s in st], float)
        j = np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)
        pos, quat = np.c_[xy[j], z[j]], low["imu_quat"].astype(np.float64)
    return RecordedState(t, np.c_[pos, quat, low["q"][:, :29].astype(np.float64)], low["dq"][:, :29].astype(np.float64))


def yaw_of(q):
    w, x, y, z = q[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--plans", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--impl", default="fused", choices=("fused", "graph", "eager"))
    ap.add_argument("--body", default="dex3")
    args = ap.parse_args(argv)
    os.environ.setdefault("G1_BODY_MODEL", args.body)
    d = Path(args.session)
    out_dir = d / "debug"
    out_dir.mkdir(exist_ok=True)
    events = node_events(d)
    t_up = next(e["t"] for e in events if e["event"] == "node_up")
    goal_launch = np.asarray(next(e["goal_odom"] for e in events if e["event"] == "goal"), float)
    plans_log = json.loads((d / "node_plans.json").read_text())["plans"]
    session = json.loads((d / "session.json").read_text())

    import torch

    flow = bundle.load_flow()
    sm, cb = flow.sensor_map, flow.callback
    cfg = bundle.student_config()
    episode = dict(cfg["episode"], arrival_radius_m=0.4)
    route_cfg = cfg["route_variants"]["sensor"].get("route", {})
    pdict = dict(cfg["planner"], device="cuda" if torch.cuda.is_available() else "cpu")
    pcfg0 = flow.planner.PlannerConfig(**pdict)
    from g1deploy.fastplan import build_planner
    from sim.lockstep import make_goal_fn

    planners = {}
    summary = []
    rebuild = None

    def view_from(support, ceiling, lo, res, n):
        v = sm.SensorMapMemory.__new__(sm.SensorMapMemory)
        v.support, v.ceiling, v.lo, v.res, v.n = support, ceiling, lo, res, n
        return v

    for n in args.plans:
        if n >= len(plans_log):
            break
        entry = plans_log[n]
        fin = d / "plans" / f"plan_{n:04d}_in.npz"
        live = None
        if fin.exists():
            z = np.load(fin)
            view = view_from(z["support"].astype(np.float64), z["ceiling"].astype(np.float64), z["lo"], float(z["res"]),
                             int(z["n"]))
            hist, goal, t_req = list(z["hist"]), z["goal"], float(z["t"])
            source = "exact node inputs (--debug-plans)"
            fout = d / "plans" / f"plan_{n:04d}_out.npz"
            if fout.exists():
                o = np.load(fout)
                live = {"qpos": o["qpos"], "rule": str(o["rule"]), "rejected": str(o["rejected"])}
        else:
            if rebuild is None:
                rebuild = Rebuild(d, flow, pdict, session, t_up)
            t_req = t_up + entry["req_tick"] * DT
            view, hist, used = rebuild.at(t_req)
            goal = goal_launch
            source = f"rebuilt from the recording ({used} lidar frames re-fused)"
        xy = np.asarray(hist[-1][:2], float)
        route = flow.route.SensorRoute(xy, goal, view, **route_cfg)
        route.refresh(xy, t_req - t_up, force=True)
        boxes = sm.memory_boxes(view, grow_overhead_m=pcfg0.guide_grow_overhead_m)
        for seed in range(args.seeds):
            if (args.impl, seed) not in planners:
                planners[(args.impl, seed)] = build_planner(flow.planner.PlannerConfig(**dict(pdict, seed=seed)),
                                                            args.impl, "fp32")
            planner = planners[(args.impl, seed)]
            pc = planner.cfg
            sel = sm.HeightFieldSelector(view, planner.certifier.tree, planner.certifier.bodies, planner.shapes)

            def map_fn(xy_, yaw_, _v=view, _pc=pc):
                m, _ = _v.ego_map(xy_, yaw_, _pc.datum_offset_m, _pc.support_clip_m or 0.8)
                return m

            goal_fn = make_goal_fn(route, episode["goal_lookahead_m"], episode["goal_heading"],
                                   episode["steer_lookahead_m"], None, cb.goal_vector, cb.route_goal_heading,
                                   arrival_radius_m=episode["arrival_radius_m"], arrival_scope=1)
            goal_fn.arrival_anchor = np.asarray(hist[-1]).copy()
            plan = planner.plan(np.asarray(hist), [], goal_fn, map_fn, sel, prev=None, guide_boxes=boxes)
            rows = []
            for i, (cand, clear, kind) in enumerate(planner.last_pools):
                name = "resample" if kind == "go" and i > 0 else kind
                for j in range(len(cand["qpos"])):
                    rows.append({"pool": name, "clearance": float(clear[j]), "safe": bool(clear[j] >= pc.select_margin_m),
                                 "xy": cand["qpos"][j, :, :2], "z": cand["qpos"][j, :, 2]})
            ceil_near = sum(1 for (i, j) in np.argwhere(np.isfinite(view.ceiling) & (view.ceiling < 1.35))
                            if np.hypot(*(view.lo + np.array([i, j]) * view.res - xy)) < 1.0)
            info = {"plan": n, "seed": seed, "source": source, "t_from_node_up_s": round(t_req - t_up, 2),
                    "rule_live": entry.get("rule"), "rejected_live": entry.get("rejected"),
                    "rule_offline": plan["rule"], "robot_xy": xy.round(3).tolist(), "yaw_deg": round(math.degrees(yaw_of(hist[-1])), 1),
                    "pelvis_z_now_m": round(float(hist[-1][2]), 3), "goal": np.round(goal, 2).tolist(),
                    "route_found": bool(getattr(route, "found", False)),
                    "pools": {k: sum(r["pool"] == k for r in rows) for k in COLORS},
                    "safe": {k: sum(r["safe"] for r in rows if r["pool"] == k) for k in COLORS},
                    "select_margin_m": pc.select_margin_m,
                    "best_go_clearance_m": round(max([r["clearance"] for r in rows if r["pool"] == "go"] or [float("nan")]), 3),
                    "offline_chosen_min_pelvis_z_m": round(float(plan["qpos"][:, 2].min()), 3),
                    "offline_chosen_end_xy": plan["qpos"][-1, :2].round(2).tolist(),
                    "live_min_pelvis_z_m": None if live is None else round(float(live["qpos"][:, 2].min()), 3),
                    "live_end_xy": None if live is None else live["qpos"][-1, :2].round(2).tolist(),
                    "boxes": len(boxes), "ceiling_cells_below_1p35_within_1m": int(ceil_near),
                    "boxes_near": [[np.round(b["center_xy"], 2).tolist(), np.round(b["half_xy"], 2).tolist(),
                                    round(b["z_bottom"], 2), round(b["z_top"], 2)] for b in boxes
                                   if np.hypot(*(np.asarray(b["center_xy"]) - xy)) < 1.5 + max(b["half_xy"])]}
            summary.append(info)
            print(json.dumps(info), flush=True)
            draw(out_dir / f"plan_{n:03d}_s{seed}.png", view, boxes, route, rows, plan["qpos"], live, hist, goal,
                 map_fn, info)
    (out_dir / "plans.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"figures in {out_dir}")
    return 0


class Rebuild:
    """The node's memory at time t, re-fused from the recorded raw lidar frames (no --debug-plans dumps)."""

    def __init__(self, d, flow, pdict, session, t_up):
        from g1deploy.perception.unitree_lidar import Mount

        rec = open_session(d)
        self.flow, self.pdict, self.t_up = flow, pdict, t_up
        self.state = recorded_state(rec)
        self.frames = list(rec.lidar_frames())
        mc = session.get("mount_calib")
        self.mount = Mount.load(mc["path"]) if mc else Mount.nominal()

    def at(self, t_req):
        from g1deploy.node import memory_view
        from g1deploy.perception.cloud_memory import CloudMemory
        from g1deploy.perception.unitree_lidar import CloudFromSensorFrame

        self.state.until(self.t_up + 0.05)
        q0 = self.state.qpos_at(self.t_up + 0.05)
        memory = CloudMemory(np.asarray(q0[:2]), rng=np.random.default_rng(0),
                             **dict(self.flow.planner.PlannerConfig(**self.pdict).sensor))
        memory.latency = 0.0
        conv = CloudFromSensorFrame(memory.body_filter, self.mount, transport_s=0.01)
        self.state.until(t_req)
        used = 0
        for t_f, pts, t_s in self.frames:
            if t_f > t_req:
                break
            if t_f >= self.t_up:
                p_w, org = conv(pts, t_s, t_f, self.state.qpos_at)
                memory.add_cloud(0.0, p_w, org, None)
                memory.step(t_f - self.t_up)
                used += 1
        return memory_view(memory), self.state.history(t_req, 6), used


def draw(path, view, boxes, route, rows, chosen, live, hist, goal, map_fn, info):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    q = np.asarray(hist[-1])
    xy, yaw = q[:2], yaw_of(q)
    fig = plt.figure(figsize=(17, 6.4))
    ax = fig.add_subplot(1, 3, 1)
    ext = [view.lo[0], view.lo[0] + view.n * view.res, view.lo[1], view.lo[1] + view.n * view.res]
    ax.imshow(np.nan_to_num(view.support, nan=-0.2).T, origin="lower", extent=ext, cmap="Greys_r", vmin=-0.2, vmax=1.2)
    cy, cx = np.nonzero(np.isfinite(view.ceiling.T) & (view.ceiling.T < 1.35))
    ax.scatter(view.lo[0] + cx * view.res, view.lo[1] + cy * view.res, s=6, c="magenta", label="ceiling < 1.35 m")
    for b in boxes:
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        hx, hy = b["half_xy"]
        corners = [np.asarray(b["center_xy"]) + np.array([c * u - s * v, s * u + c * v])
                   for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
        ax.add_patch(Polygon(corners, fill=False, ec="red" if b["z_bottom"] < 0.3 else "orange", lw=1.2))
    path_ = getattr(route, "path", None)
    if path_ is not None and len(path_) > 1:
        ax.plot(np.asarray(path_)[:, 0], np.asarray(path_)[:, 1], "c--", lw=1.5, label="route")
    for r in rows:
        ax.plot(r["xy"][:, 0], r["xy"][:, 1], color=COLORS[r["pool"]], lw=0.8, alpha=0.9 if r["safe"] else 0.3,
                ls="-" if r["safe"] else ":")
    ax.plot(chosen[:, 0], chosen[:, 1], "y-", lw=3, label="offline chosen")
    if live is not None:
        ax.plot(live["qpos"][:, 0], live["qpos"][:, 1], "r-", lw=2.5, label=f"LIVE plan ({live['rule']})")
    ax.plot(*xy, "co", ms=9)
    ax.arrow(xy[0], xy[1], 0.4 * math.cos(yaw), 0.4 * math.sin(yaw), color="c", width=0.02)
    ax.plot(*goal, "y*", ms=15)
    ax.set_xlim(xy[0] - 2.5, xy[0] + 3.5)
    ax.set_ylim(xy[1] - 3, xy[1] + 3)
    ax.set_title(f"plan {info['plan']} t+{info['t_from_node_up_s']} s: live {info['rule_live']}"
                 f"{' REJECTED' if info['rejected_live'] else ''} / offline {info['rule_offline']}", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax2 = fig.add_subplot(1, 3, 2)
    for r in rows:
        ax2.plot(np.arange(len(r["z"])) * DT, r["z"], color=COLORS[r["pool"]], lw=0.8, alpha=0.9 if r["safe"] else 0.3,
                 ls="-" if r["safe"] else ":")
    ax2.plot(np.arange(len(chosen)) * DT, chosen[:, 2], "y-", lw=3)
    if live is not None:
        ax2.plot(np.arange(len(live["qpos"])) * DT, live["qpos"][:, 2], "r-", lw=2.5)
    ax2.axhline(0.35, color="r", lw=1, ls="--")
    ax2.text(0.02, 0.36, "node rejects pelvis < 0.35 m", color="r", fontsize=7)
    ax2.set_xlabel("s ahead")
    ax2.set_ylabel("pelvis z (m)")
    ax2.set_title("pelvis height: green go, blue resample, purple stop (dotted = unsafe), yellow offline, red live",
                  fontsize=8)
    ax3 = fig.add_subplot(1, 3, 3)
    m = np.asarray(map_fn(xy, yaw))
    m = m.reshape(-1, *m.shape[-2:]) if m.ndim > 2 else m[None]
    im = ax3.imshow(np.concatenate(list(m), axis=1), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax3, fraction=0.03)
    ax3.set_title(f"ego map the network sees ({m.shape[0]} channel(s), robot-centred, heading = +x)", fontsize=8)
    fig.text(0.01, 0.01, f"{info['source']} | route found {info['route_found']}; pools {info['pools']}; safe "
             f"{info['safe']} (margin {info['select_margin_m']} m, best go clearance {info['best_go_clearance_m']} m); "
             f"pelvis now {info['pelvis_z_now_m']}, offline chosen min {info['offline_chosen_min_pelvis_z_m']}, live min "
             f"{info['live_min_pelvis_z_m']}; ceiling cells <1.35 m within 1 m: {info['ceiling_cells_below_1p35_within_1m']}; "
             f"boxes {info['boxes']}", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
