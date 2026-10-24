#!/usr/bin/env python3
"""Path A: lock-step closed-loop episode of the student planner + SONIC in MuJoCo.

A counterpart of the Isaac callback (`closed_loop/callback.py`, frozen in the
bundle): same history buffer, sensor memory, planner call, feed target, SONIC
input and TaskScorer rules, with tracked deployment route, goal and selection
adapters described in README.md. The physics engine and SONIC runtime are
MuJoCo and fused ONNX. Deployment effects can be switched on one at a time:

  --latency-s L   a plan requested at tick t becomes usable at t + L (physics keeps running on
                  the previous plan, exactly what the real-time streamer will do); the next
                  request is issued when the planner is free and `replan_ticks` have passed.
  --anchor predict  plan from the predicted state at t + L (g1deploy/latency.py) instead of t.
  --inflight N    up to N requests in flight (pipelined planner workers), one every `replan_ticks`;
                  predictions use only the plan that has landed, as on hardware.
  --odom-noise    random-walk xy/yaw error on the planner's pose (flow `StateEstimate` config JSON).
  --start stand   start standing at SONIC's default pose instead of the recorded walk.
  --route sensor  lidar route (A* on the sensor memory, 0.3 s pre-scan) instead of the true boxes.
  --prescan-s P   lidar captures during P s before the first plan (hardware: the robot stands while
                  the memory fills). Isaac's privileged-route protocol used 0.
  --lidar mjray   MuJoCo ray-cast MID-360 (scene incl. the robot's own body) + label-free memory.
  --guards deploy the deploy-side guards of g1deploy/guards.py (self mask, split guidance boxes, walk
                  envelope, calm fallback, anchor check); `walls` adds overhead-as-wall (H5-H6).

Outputs one directory per episode: score.json, plans.json, closed_loop.npz (realized qpos, references).

usage: lockstep.py --task <task.json> --seed N --out <dir> [options]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1deploy import bundle, guards as G  # noqa: E402
from g1deploy.latency import predicted_history  # noqa: E402
from g1deploy.scoring import TaskScorer  # noqa: E402
from sim.episode import Physics, read_json, stand_start_state, task_start_state  # noqa: E402


def _finite(v):
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def write_json(path, value):
    Path(path).write_text(json.dumps(_finite(value), indent=1, allow_nan=False, default=float) + "\n")


def make_goal_fn(route, lookahead, heading_mode, steer_lookahead, turn_min_frac, goal_vector, route_goal_heading,
                 *, arrival_radius_m=None, arrival_scope=None):
    """Source pursuit semantics with a deployment occupancy-prefix guard.

    Shared by this rehearsal and g1deploy.node's planner workers. The long
    pursuit ray can overshoot a route bend; return to the visible route when
    that ray enters occupied cells, falling back to its free prefix if needed.
    """
    from g1deploy.goal_guard import REVISION, limit_goal, update_arrival_latch

    if arrival_radius_m is not None and (not math.isfinite(arrival_radius_m) or arrival_radius_m < 0):
        raise ValueError("arrival_radius_m must be finite and nonnegative")

    diagnostics = {}
    arrival_state = None

    def set_anchor(anchor, state):
        nonlocal arrival_state
        radius = arrival_radius_m if heading_mode == "bearing_near" else None
        entry_heading = None
        if radius is not None and np.linalg.norm(np.asarray(anchor)[:2] - route.goal) <= radius:
            w, x, y, z = anchor[3:7]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            _, _, entry_heading, _ = pursuit_goal(anchor[:2], yaw)
        arrival_state = update_arrival_latch(state, anchor, route.goal, radius,
                                            max(radius or 0.0, 0.5 * lookahead), arrival_scope, entry_heading)
        return arrival_state

    def reset_diagnostics():
        diagnostics.clear()
        diagnostics.update(revision=REVISION, queries=0, limited=0, redirected=0,
                           occupied_starts=0, arrival_heading_held=0, first=None,
                           arrival_radius_m=arrival_radius_m, arrival_latch=arrival_state)

    def pursuit_goal(xy, yaw):
        w, h, rem = route.waypoint(xy, lookahead, update=False)
        if heading_mode == "bearing_near" and rem >= 0.5 * lookahead:
            wn, _, _ = route.waypoint(xy, steer_lookahead, update=False)
            d = np.asarray(wn) - np.asarray(xy)
            b = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else h
            dist = lookahead
            if turn_min_frac is not None:
                err = abs(math.atan2(math.sin(b - yaw), math.cos(b - yaw)))
                dist = lookahead * max(float(turn_min_frac), 1.0 - err / (0.5 * math.pi))
            w = np.asarray(xy) + min(dist, rem) * np.array([math.cos(b), math.sin(b)])
            heading = b
        else:
            heading = route_goal_heading(w, h, rem, xy, heading_mode, lookahead)
        guarded, status = limit_goal(route, xy, w, lookahead)
        if status == "redirected_to_visible_route" and heading_mode != "path":
            delta = guarded - np.asarray(xy)
            heading = math.atan2(delta[1], delta[0])
        return w, guarded, heading, status

    def goal_fn(xy, yaw):
        w, guarded, heading, status = pursuit_goal(xy, yaw)
        # Arrival is entered only by the plan request's anchor, not by whichever
        # sampled candidate first predicts a nearby point. Preserve the source
        # approach heading, not measured execution that may lag a requested turn.
        # A fixed WORLD yaw restores orientation throughout the reference window.
        hold_heading = arrival_state is not None
        if hold_heading:
            heading = arrival_state["heading_rad"]
        diagnostics["arrival_heading_held"] += hold_heading
        diagnostics["queries"] += 1
        diagnostics["limited"] += status == "limited_at_occupied_grid"
        diagnostics["redirected"] += status == "redirected_to_visible_route"
        diagnostics["occupied_starts"] += status == "occupied_start_uncertified"
        if diagnostics["first"] is None:
            diagnostics["first"] = dict(origin_xy=np.asarray(xy).tolist(),
                                        requested_xy=np.asarray(w).tolist(),
                                        returned_xy=guarded.tolist(), status=status,
                                        candidate_yaw_rad=float(yaw), heading_rad=float(heading),
                                        heading_policy="latched_arrival_world_yaw" if hold_heading else "route_pursuit")
        return goal_vector(guarded, heading, xy, yaw)

    reset_diagnostics()
    goal_fn.reset_diagnostics = reset_diagnostics
    goal_fn.set_anchor = set_anchor
    goal_fn.diagnostics = diagnostics
    return goal_fn


_PLANNERS = {}


def _dump_memory(path, memory):
    near = memory.near_only() if hasattr(memory, "near_only") else np.zeros_like(memory.support, dtype=bool)
    np.savez_compressed(path, support=memory.support, ceiling=memory.ceiling, lo=memory.lo, res=memory.res, near_only=near)


def cached_planner(flow, pcfg, impl="fused", precision="fp32"):
    """One planner per (config, implementation) per process, re-seeded for each episode.

    Equal to a fresh `FlowPlanner(pcfg)`: its only per-run state is the generator (re-seeded here) and the
    call counter; the fast planner's compiled graphs are then built once per worker, not per episode."""
    import dataclasses

    key = (impl, precision, json.dumps({k: v for k, v in dataclasses.asdict(pcfg).items() if k != "seed"}, sort_keys=True, default=str))
    planner = _PLANNERS.get(key)
    if planner is None:
        from g1deploy.fastplan import build_planner

        planner = _PLANNERS[key] = build_planner(pcfg, impl, precision)
    planner.cfg = pcfg
    planner.generator.manual_seed(int(pcfg.seed))
    planner.calls = 0
    planner._g1_arrival_heading = None  # repeated independent episodes must not share an arrival latch
    return planner


def run_episode(task_path, seed, out, *, device="cuda", latency_s=0.0, anchor_mode="measured", inflight=1,
                start="recorded",
                route_source="privileged", lidar="analytic", odom_noise=None, prescan_s=None, lidar_cfg=None,
                goal_lookahead=None, planner_overrides=None, planner_impl="fused", planner_precision="fp32",
                deadline_s=None, save_npz=True, guards=None, dump_memory=False):
    import torch

    flow = bundle.load_flow()
    cfg = bundle.student_config()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    task = read_json(task_path)
    if deadline_s:
        task["deadline_s"] = float(deadline_s)
    contract = json.loads((bundle.ROOT / "config/sonic_model_contract.json").read_text())
    limits = np.asarray([j["range_rad"] for j in contract["joints"] if j["type"] != "free"])

    pdict = dict(cfg["planner"], device=device, seed=int(seed), **(planner_overrides or {}))
    pcfg = flow.planner.PlannerConfig(**pdict)
    planner = cached_planner(flow, pcfg, planner_impl, planner_precision)
    ep = dict(cfg["episode"])
    ep["arrival_radius_m"] = float(task["goal_radius_m"])
    if goal_lookahead:
        ep["goal_lookahead_m"] = float(goal_lookahead)  # the local goal distance sets the planner's walking speed
    replan, lookahead = int(ep["replan_ticks"]), float(ep["goal_lookahead_m"])
    variant = cfg["route_variants"][route_source]
    route_cfg = variant.get("route", {})
    prescan = float(variant.get("sensor_prescan_s", 0.0) if prescan_s is None else prescan_s)
    lidar_cfg = dict(lidar_cfg or {})
    gcfg = G.preset(guards)
    run_cfg = dict(task_path=str(task_path), seed=int(seed), planner=pdict, episode=ep, route_source=route_source,
                   lidar_cfg=lidar_cfg,
                   route=route_cfg, sensor_prescan_s=prescan, latency_s=latency_s, anchor_mode=anchor_mode, inflight=inflight,
                   start=start,
                   lidar=lidar,
                   odom_noise=odom_noise, engine="mujoco", sonic_onnx=str(bundle.sonic_onnx()), planner_impl=planner_impl,
                   planner_precision=planner_precision, guards=gcfg.describe() if gcfg.any() else None)
    write_json(out / "run_config.json", run_cfg)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    boxes = task["scene_boxes_world"]
    sim = Physics(boxes, bundle.sonic_onnx())
    if start == "recorded":
        s0 = task_start_state(task)
    elif start == "stand":
        z = np.load(task["initial_state_record"])
        q0 = z["root_quaternion_wxyz"][int(task.get("initial_frame", 0))]
        yaw0 = math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))
        s0 = stand_start_state(task.get("initial_root_xy", z["root_position_m"][0][:2]), yaw0)
    else:
        raise ValueError(start)
    sim.reset(s0)
    if odom_noise and odom_noise.get("leg"):  # leg odometry from joints + IMU attitude (g1deploy/leg_odometry.py)
        from g1deploy.leg_odometry import LegOdomEstimate

        estimate = LegOdomEstimate(odom_noise)
    else:
        estimate = flow.callback.StateEstimate(odom_noise, np.random.default_rng([int(seed), 7]))

    m0 = sim.measure()
    start_q = m0["qpos"]
    goal_xy = np.asarray(task["goal_world_xy"], dtype=float)
    history = flow.callback.backfill_history(start_q, m0["dq"], m0["root_linear_velocity_w"],
                                             m0["root_angular_velocity_w"], planner.history)
    sm = flow.sensor_map
    if lidar == "analytic":
        memory = sm.RayCastMemory(boxes, start_q[:2], rng=np.random.default_rng(int(seed)), **dict(pcfg.sensor))
    elif lidar == "mjray":
        from g1deploy.perception.cloud_memory import MjRayMemory

        memory = MjRayMemory(sim, start_q[:2], rng=np.random.default_rng(int(seed)), **dict(pcfg.sensor))
    elif lidar == "mid360":  # real scan pattern, per-point timing, mount error, deskew mode (mid360_scan.py)
        from g1deploy.perception.cloud_memory import CloudMemory
        from g1deploy.perception.mid360_scan import Mid360Scanner

        memory = CloudMemory(start_q[:2], rng=np.random.default_rng(int(seed)), **dict(pcfg.sensor))
        scanner = Mid360Scanner(sim.model, sim.data, range_m=pcfg.sensor.get("range_m", 4.0),
                                rng=np.random.default_rng([int(seed), 11]), body_filter=memory.body_filter,
                                **lidar_cfg)
        sim.hooks.append(scanner.substep)

        def drain_scanner():
            for pts, org, t_end in scanner.pop_frames():
                memory.add_cloud(t_end + memory.latency, pts, org, None)  # self-filtered per slice in the scanner
    else:
        raise ValueError(lidar)
    if gcfg.min_high_points and hasattr(memory, "min_high"):
        memory.min_high = int(gcfg.min_high_points)
    if prescan > 0 and lidar == "mid360":  # standing/frozen pre-scan: the sensor runs, physics does not
        for t_ in np.arange(-prescan, 1e-9, 0.005):
            scanner.substep(float(t_))
        drain_scanner()
        memory.step(0.0)
    elif prescan > 0:
        memory.next_capture = -prescan
        for t_ in np.arange(-prescan, 1e-9, 0.02):
            memory.step(float(t_), start_q[:3], start_q[3:7])
    clearance_fn = sm.HeightFieldSelector(memory, planner.certifier.tree, planner.certifier.bodies, planner.shapes)

    def map_fn(xy, yaw):
        m, _ = memory.ego_map(xy, yaw, pcfg.datum_offset_m, pcfg.support_clip_m or 0.8)
        return m

    if route_source == "sensor":
        route = flow.route.SensorRoute(start_q[:2], goal_xy, memory, **route_cfg)
    else:
        route = flow.route.Route(start_q[:2], goal_xy, boxes, **route_cfg)
    goal_fn = make_goal_fn(route, lookahead, ep["goal_heading"], float(ep["steer_lookahead_m"]), ep.get("turn_min_frac"),
                           flow.callback.goal_vector, flow.callback.route_goal_heading,
                           arrival_radius_m=ep["arrival_radius_m"], arrival_scope=str(out.resolve()))
    scorer = TaskScorer(task)
    lat_ticks = int(round(latency_s / 0.02))

    active = None  # (plan, anchor_tick, anchor_q, anchor_dq)
    pending = []  # [(ready_tick, plan, anchor_tick, anchor_q, anchor_dq)], oldest first
    measured_until = -1  # anchor check: plan from the measured state until this tick
    drops = dropped_total = 0  # walking envelope drops in a row / in all
    last_request = -10**9
    plans, references, realized, pre_q, est_q, wall = [], [], [], [], [], []
    outcome, error = None, None
    try:
        for tick in range(math.ceil(task["deadline_s"] / 0.02)):
            t0 = time.perf_counter()
            meas = sim.measure()
            true_q = meas["qpos"]
            qpos = estimate(true_q, 0.02)
            history.append(qpos.astype(np.float32))
            history = history[-(planner.history + 2):]
            pre_q.append(true_q.copy())
            if estimate.cfg:
                est_q.append(qpos.copy())
                memory.pose_error = tuple(estimate.err)
            if lidar == "mid360":
                drain_scanner()
            memory.step(tick * 0.02, true_q[:3], true_q[3:7])
            while pending and tick >= pending[0][0]:
                landed = pending.pop(0)[1:]
                if landed[0] is None:  # a dropped plan: the node learns of it when it lands; the active plan plays on
                    continue
                active = landed
                if active[0].get("predicted"):  # the predicted anchor vs the measured pose it was meant for
                    miss = float(np.linalg.norm(np.asarray(active[2][:2]) - qpos[:2]))
                    active[0]["_log"]["anchor_miss_m"] = miss
                    if gcfg.anchor_check_m is not None and miss > gcfg.anchor_check_m:
                        measured_until = tick + gcfg.anchor_hold_plans * replan + lat_ticks
            if len(pending) < inflight and (active is None or tick - last_request >= replan):
                if route_source == "sensor":
                    route.refresh(qpos[:2], tick * 0.02)
                wp, heading, remaining = route.waypoint(qpos[:2], lookahead)
                guide_boxes = sm.memory_boxes(memory, grow_overhead_m=pcfg.guide_grow_overhead_m) \
                    if pcfg.guide_w > 0 and pcfg.guide_source == "sensor" and not gcfg.any() else None
                predictive = anchor_mode == "predict" and lat_ticks > 0 and active is not None
                standing = active is None or G.is_standing(history[-6:], gcfg)
                forced = False
                if predictive and gcfg.anchor_check_m is not None and (
                        tick < measured_until or active[0]["rule"] in G.FALLBACK_RULES or active[0].get("no_safe_candidate")):
                    forced, predictive = True, False  # never extrapolate a fallback or a plan the robot did not follow
                plan_hist = predicted_history(history, active, tick, lat_ticks, planner.history + 2) \
                    if predictive else history
                # a measured plan from a standstill starts when it lands (as the first plan), not lat_ticks into it
                anchor_at = tick + lat_ticks if predictive or (forced and standing) else tick
                prev = None if active is None else (active[0], anchor_at - active[1])
                tp = time.perf_counter()
                goal_fn.arrival_anchor = qpos.copy()  # observed estimate, not latency-predicted history
                if gcfg.any():
                    plan = G.plan_guarded(planner, plan_hist, boxes, goal_fn, G.memory_view(memory, gcfg.self_mask), gcfg,
                                          measured_q=qpos, standing=standing, prev=prev,
                                          guide_grow_overhead_m=pcfg.guide_grow_overhead_m,
                                          allow_drop=active is not None and drops < gcfg.max_drops)
                    if plan.get("rejected") and active is None:
                        plan = dict(plan, **G.stand_plan(plan_hist[-1], len(plan["qpos"])), rule="stand_rejected")
                    elif plan.get("rejected"):  # dropped, as by the node: the active plan keeps playing
                        plans.append(dict(tick=tick, rule="dropped", rejected=plan["rejected"], guard=plan.get("guard"),
                                          latency_s=time.perf_counter() - tp, chosen_clearance_m=None,
                                          plan_xy=plan["qpos"][:, :2].tolist(), plan_root_z=plan["qpos"][:, 2].tolist(),
                                          predicted=bool(predictive), no_safe_candidate=plan.get("no_safe_candidate"),
                                          candidate_pool=plan.get("candidate_pool"),
                                          plan_knee_max_rad=float(plan["qpos"][:, [7 + k for k in G.KNEES]].max()),
                                          plan_tilt_max_rad=G.plan_metrics(plan)["tilt"]))
                        last_request = tick
                        drops, dropped_total = drops + 1, dropped_total + 1
                        if lat_ticks:
                            pending.append((tick + lat_ticks, None, None, None, None))  # occupies the planner slot
                        plan = None
                else:
                    plan = planner.plan(np.asarray(plan_hist), boxes, goal_fn, map_fn, clearance_fn, prev=prev,
                                        guide_boxes=guide_boxes)
                if plan is not None:
                    latency = time.perf_counter() - tp
                    drops = 0
                    plan["predicted"] = bool(predictive)
                    if dump_memory and not plans:
                        _dump_memory(out / "memory_first.npz", memory)
                    anchor = (plan, anchor_at, np.asarray(plan_hist[-1], dtype=np.float64).copy(),
                              flow.feed.finite_difference_dq(plan_hist))
                    last_request = tick
                    if lat_ticks == 0 or active is None:
                        # The first plan is computed before the stream starts (on hardware: while standing),
                        # so only replans pay the latency.
                        active = anchor
                    else:
                        pending.append((tick + lat_ticks,) + anchor)
                    plans.append(dict(tick=tick, waypoint_xy=np.asarray(wp).tolist(), route_remaining_m=remaining,
                                      rule=plan["rule"], chosen_clearance_m=plan["chosen_clearance_m"],
                                      latency_s=latency, plan_xy=plan["qpos"][:, :2].tolist(),
                                      plan_root_z=plan["qpos"][:, 2].tolist(),
                                      no_safe_candidate=plan.get("no_safe_candidate"), candidate_pool=plan.get("candidate_pool"),
                                      predicted=plan["predicted"], guard=plan.get("guard"),
                                      plan_knee_max_rad=float(plan["qpos"][:, [7 + k for k in G.KNEES]].max()),
                                      plan_tilt_max_rad=G.plan_metrics(plan)["tilt"]))
                    plan["_log"] = plans[-1]
            plan, anchor_tick, anchor_q, anchor_dq = active
            tgt, clamped = flow.feed.target(plan, anchor_q, anchor_dq, tick - anchor_tick, qpos, limits)
            pos, quat, force, _ = sim.step(tgt[:, :29], tgt[:, 29:58], tgt[:, 58:64])
            references.append(tgt)
            post = sim.measure()
            realized.append(post["qpos"])
            wall.append(time.perf_counter() - t0)
            outcome = scorer.update(post["qpos"][:3], post["root_linear_velocity_w"], physics_root_positions=pos,
                                    physics_root_quaternions=quat, physics_contact_force_n=force)
            if outcome is not None:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report = scorer.report()
        if error:
            report["outcome"], report["completed"] = "runtime_error", False
        lat = [p["latency_s"] for p in plans if p["rule"] != "dropped"]
        report.update(runtime_error=error, task_id=task.get("task_id"), family=task.get("family"), seed=int(seed),
                      replans=sum(p["rule"] != "dropped" for p in plans), dropped_plans=dropped_total,
                      route_found=bool(route.found),
                      plan_latency_s={"p50": float(np.median(lat)), "p95": float(np.quantile(lat, 0.95))} if lat else None,
                      selection_rules={r: sum(p["rule"] == r for p in plans) for r in sorted({p["rule"] for p in plans})},
                      control_step_wall_p50_s=float(np.median(wall)) if wall else None, **{k: run_cfg[k] for k in (
                          "latency_s", "anchor_mode", "inflight", "start", "lidar", "route_source")})
        write_json(out / "score.json", report)
        write_json(out / "plans.json", plans)
        if dump_memory:
            _dump_memory(out / "memory_last.npz", memory)
        if save_npz:
            np.savez_compressed(out / "closed_loop.npz", pre_action_qpos=np.asarray(pre_q),
                                realized_qpos=np.asarray(realized), references=np.asarray(references),
                                estimated_qpos=np.asarray(est_q))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--latency-s", type=float, default=0.0)
    ap.add_argument("--anchor", choices=("measured", "predict"), default="measured")
    ap.add_argument("--inflight", type=int, default=1, help="planner requests in flight (pipelined planners)")
    ap.add_argument("--start", choices=("recorded", "stand"), default="recorded")
    ap.add_argument("--route", choices=("privileged", "sensor"), default="privileged")
    ap.add_argument("--lidar", choices=("analytic", "mjray", "mid360"), default="analytic")
    ap.add_argument("--lidar-cfg", default=None,
                    help='mid360 options as JSON, e.g. {"mount_error": {"pitch_deg": 1.5}, "deskew": "none", "dropout": 0.1}')
    ap.add_argument("--odom-noise", default=None, help="JSON dict, e.g. '{\"xy_rw_m\": 0.02, \"yaw_rw_rad\": 0.01}'")
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--goal-lookahead", type=float, default=None, help="override (1.5 evaluated); sets walking speed")
    ap.add_argument("--guards", default=None, choices=sorted(G.PRESETS), help="deploy-side guards (g1deploy/guards.py)")
    ap.add_argument("--prescan-s", type=float, default=None,
                    help="lidar captures before the first plan (default: the route variant's; Isaac privileged = 0)")
    args = ap.parse_args(argv)
    r = run_episode(args.task, args.seed, args.out, device=args.device, latency_s=args.latency_s,
                    anchor_mode=args.anchor, inflight=args.inflight, start=args.start,
                    route_source=args.route, lidar=args.lidar,
                    lidar_cfg=json.loads(args.lidar_cfg) if args.lidar_cfg else None,
                    odom_noise=json.loads(args.odom_noise) if args.odom_noise else None, deadline_s=args.deadline_s,
                    prescan_s=args.prescan_s, goal_lookahead=args.goal_lookahead, guards=args.guards)
    print(json.dumps({k: r[k] for k in ("task_id", "outcome", "elapsed_s", "replans", "plan_latency_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
