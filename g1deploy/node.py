#!/usr/bin/env python3
"""Real-time student planner node: G1 state + MID-360 cloud -> student planner -> SONIC ZMQ stream.

This is the process that runs next to the SONIC C++ runner, in the MuJoCo rehearsal and on the robot:

  inputs   rt/lowstate (DDS: 29 joints, pelvis IMU)          via unitree_sdk2py
           odometry: rt/odostate (DDS, sim ground truth) | `--odom zmq` (LIO adapter, docs/HARDWARE.md)
           cloud: ZMQ SUB topic `cloud` (points [N,3] in the odometry frame, origin [3], stamp)
  output   ZMQ PUB :5556 topic `pose`, SONIC protocol v1 (g1deploy/zmq_stream.py)
  events   JSON lines on stdout ({"event": ...}); `estop` asks the operator/orchestrator to stop the runner

Loop at 50 Hz (tick = the stream's frame_index):
  history <- odometry pose + joints;  cloud -> CloudMemory (self filter, label-free layers)
  every replan_ticks, up to `inflight` planner jobs (one FlowPlanner per worker thread), each planned
  from the predicted state at its landing tick (g1deploy/latency.py) and streamed from that tick on.
Phases: WAIT (no stream) -> PRESCAN (fill memory while standing) -> RUN -> ARRIVED | STOPPED.

Safety supervisor (never relaxes): tilt > --tilt-estop -> estop; state/odometry stale > 0.5 s -> estop;
cloud stale > --cloud-timeout -> no new plans (the runner holds the last frames); planned pelvis
height / speed / finiteness checks reject a plan (the previous one keeps playing).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g1deploy import bundle, contract as C  # noqa: E402
from g1deploy.latency import predicted_history  # noqa: E402
from g1deploy.zmq_stream import LOOKAHEAD_FRAMES, PoseStreamer, unpack  # noqa: E402

DT = 0.02


def emit(event, **kw):
    print(json.dumps({"event": event, "t": round(time.monotonic(), 3), **kw}, default=float), flush=True)


def quat_tilt(q):
    w, x, y, z = q
    return math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))


class DdsState:
    """Latest rt/lowstate + odometry snapshot (DDS callbacks run on the SDK's threads)."""

    def __init__(self, iface="lo", domain=0, odom="dds"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, OdoState_

        ChannelFactoryInitialize(domain, iface)
        self.lock = threading.Lock()
        self.low, self.odo, self.t_low, self.t_odo = None, None, -1.0, -1.0
        self.sub_low = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub_low.Init(self._on_low, 1)
        if odom == "dds":
            self.sub_odo = ChannelSubscriber("rt/odostate", OdoState_)
            self.sub_odo.Init(self._on_odo, 1)

    def _on_low(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(29)])
        dq = np.array([msg.motor_state[i].dq for i in range(29)])
        imu = np.array(msg.imu_state.quaternion, dtype=float)  # w x y z
        gyro = np.array(msg.imu_state.gyroscope, dtype=float)
        with self.lock:
            self.low, self.t_low = (q, dq, imu, gyro), time.monotonic()

    def _on_odo(self, msg):
        with self.lock:
            # gear_sonic_sim writes wxyz into `orientation` (despite the IDL comment)
            self.odo = (np.array(msg.position, dtype=float), np.array(msg.orientation, dtype=float),
                        np.array(msg.linear_velocity, dtype=float), np.array(msg.angular_velocity, dtype=float))
            self.t_odo = time.monotonic()

    def set_odometry(self, pos, quat_wxyz, lin_w, ang_w):
        """External odometry (LIO adapter) entry point."""
        with self.lock:
            self.odo, self.t_odo = (np.asarray(pos), np.asarray(quat_wxyz), np.asarray(lin_w), np.asarray(ang_w)), time.monotonic()

    def snapshot(self):
        with self.lock:
            if self.low is None or self.odo is None:
                return None
            (q, dq, imu, gyro), (pos, oq, lin, ang) = self.low, self.odo
            now = time.monotonic()
            return dict(qpos=np.r_[pos, oq, q], dq=dq, imu=imu, gyro=gyro, lin_w=lin, ang_w=ang,
                        age_low=now - self.t_low, age_odo=now - self.t_odo)


class CloudInput:
    def __init__(self, connect="tcp://localhost:5560"):
        import zmq

        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 4)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"cloud")
        self.sock.connect(connect)
        self.last = -1.0

    def poll(self):
        import zmq

        out = []
        while True:
            try:
                msg = self.sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                return out
            _, f = unpack(msg, topic=b"cloud")
            self.last = time.monotonic()
            out.append((f["points"].astype(np.float64), f["origin"], float(f["stamp"][0])))


def frozen_memory(memory):
    snap = copy.copy(memory)
    snap.support, snap.ceiling = memory.support.copy(), memory.ceiling.copy()
    snap.pending = []
    return snap


class StudentNode:
    def __init__(self, args):
        self.args = args
        self.flow = bundle.load_flow()
        cfg = bundle.student_config()
        ep = cfg["episode"]
        self.replan, self.lookahead = int(ep["replan_ticks"]), float(ep["goal_lookahead_m"])
        self.heading_mode, self.steer = ep["goal_heading"], float(ep["steer_lookahead_m"])
        variant = cfg["route_variants"][args.route]
        self.route_cfg = variant.get("route", {})
        pdict = dict(cfg["planner"], device=args.device)
        self.pcfg = self.flow.planner.PlannerConfig(**pdict)
        self.planners = [self.flow.planner.FlowPlanner(self.flow.planner.PlannerConfig(**dict(pdict, seed=args.seed + i)))
                         for i in range(args.inflight)]
        self.free_planners = list(self.planners)
        contract = json.loads((bundle.ROOT / "config/sonic_model_contract.json").read_text())
        self.limits = np.asarray([j["range_rad"] for j in contract["joints"] if j["type"] != "free"])
        self.pool = ThreadPoolExecutor(max_workers=args.inflight)
        self.jobs = []  # (future, planner, anchor_tick, anchor_q, anchor_dq, request_tick, t_request)
        self.latency_est = args.latency_guess_s
        self.latencies = []
        self.goal = np.asarray(args.goal, dtype=float)
        self.streamer = None if args.dry_run else PoseStreamer(args.stream_bind)
        self.history, self.tick, self.phase = [], 0, "WAIT"
        self.active = None  # (plan, anchor_tick, anchor_q, anchor_dq)
        self.last_streamed_anchor = -10**9
        self.last_request = -10**9
        self.stream_end = -1
        self.log = []
        self.stop_flag = threading.Event()

    # -- perception ------------------------------------------------------------------------
    def make_memory(self, xy):
        from g1deploy.perception.cloud_memory import CloudMemory

        self.memory = CloudMemory(xy, rng=np.random.default_rng(self.args.seed), **dict(self.pcfg.sensor))
        self.memory.latency = 0.0  # real latency is already in the data; fuse on arrival

    def route_now(self, xy, t):
        """A frozen copy of the route for one planner job (refresh swaps in new arrays)."""
        if not hasattr(self, "route"):
            if self.args.route == "sensor":
                self.route = self.flow.route.SensorRoute(xy, self.goal, self.memory, **self.route_cfg)
                self.route.refresh(xy, t, force=True)
            else:
                boxes = json.loads(Path(self.args.boxes).read_text()) if self.args.boxes else []
                self.route = self.flow.route.Route(xy, self.goal, boxes, **self.route_cfg)
        elif self.args.route == "sensor":
            self.route.refresh(xy, t)
        return copy.copy(self.route)

    def goal_fn_for(self, route):
        from sim.lockstep import make_goal_fn

        cb = self.flow.callback
        return make_goal_fn(route, self.lookahead, self.heading_mode, self.steer, None, cb.goal_vector,
                            cb.route_goal_heading)

    # -- planning ------------------------------------------------------------------------------
    def _plan_job(self, planner, hist, route, memory):
        sm = self.flow.sensor_map
        sel = sm.HeightFieldSelector(memory, planner.certifier.tree, planner.certifier.bodies, planner.shapes)

        def map_fn(xy, yaw):
            m, _ = memory.ego_map(xy, yaw, self.pcfg.datum_offset_m, self.pcfg.support_clip_m or 0.8)
            return m

        guide = sm.memory_boxes(memory, grow_overhead_m=self.pcfg.guide_grow_overhead_m) if self.pcfg.guide_w > 0 else None
        t0 = time.monotonic()
        plan = planner.plan(np.asarray(hist), [], self.goal_fn_for(route), map_fn, sel, prev=None, guide_boxes=guide)
        return plan, time.monotonic() - t0

    def request(self, now_tick):
        planner = self.free_planners.pop(0)
        lead = 0 if self.active is None else int(math.ceil(self.latency_est / DT)) + 1
        hist = self.history if self.active is None or lead == 0 else predicted_history(
            self.history, self.active, now_tick, lead, planner.history + 2)
        anchor_tick = now_tick + lead
        route = self.route_now(np.asarray(self.history[-1][:2], dtype=float), now_tick * DT)
        fut = self.pool.submit(self._plan_job, planner, list(hist), route, frozen_memory(self.memory))
        anchor_q = np.asarray(hist[-1], dtype=np.float64)
        anchor_dq = self.flow.feed.finite_difference_dq(hist)
        self.jobs.append((fut, planner, anchor_tick, anchor_q, anchor_dq, now_tick, time.monotonic()))
        self.last_request = now_tick

    def stand(self, now_tick, reason, settle_s=1.0, hold_s=3.0):
        """Stream a settle-to-stand chunk from the measured pose: joints -> SONIC default pose over
        `settle_s`, pelvis yaw kept, then hold. Used at arrival and as the soft stop."""
        q = np.asarray(self.history[-1], dtype=np.float64)
        n_settle, n = int(settle_s / DT), int((settle_s + hold_s) / DT)
        a = np.clip(np.arange(1, n + 1) / n_settle, 0.0, 1.0)[:, None]
        a = a * a * (3 - 2 * a)  # smoothstep
        frames = np.repeat(q[None], n, 0)
        frames[:, 7:] = (1 - a) * q[7:] + a * C.DEFAULT_MJ
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        plan = {"qpos": frames, "rotation": np.repeat(rot[None], n, 0),
                "joint_velocity": np.vstack([np.zeros((1, 29)), np.diff(frames[:, 7:], axis=0) / DT])}
        self.jobs = []  # results of in-flight plans are discarded from now on
        self.active = (plan, now_tick, q, np.zeros(29))
        self.last_streamed_anchor = now_tick + 10**6
        if self.streamer is not None:
            self.stream_end = self.streamer.send_plan(plan, now_tick, q, np.zeros(29), now_tick, self.limits)[1]
        emit("stand", reason=reason, tick=now_tick)

    def sane(self, plan):
        q = plan["qpos"]
        if not np.isfinite(q).all() or not np.isfinite(plan["joint_velocity"]).all():
            return "nonfinite"
        if q[:, 2].min() < self.args.min_pelvis_z or q[:, 2].max() > 1.0:
            return f"pelvis_z {q[:, 2].min():.2f}..{q[:, 2].max():.2f}"
        # Planned pelvis xy sways (0.2 s-window peaks 1.7-2.1 m/s in Isaac-evaluated plans) and SONIC does not
        # read root position; bound the plan's mean speed and the joint velocities SONIC will be asked to track.
        v = float(np.linalg.norm(q[-1, :2] - q[0, :2]) / (DT * (len(q) - 1)))
        if v > self.args.max_speed:
            return f"mean speed {v:.2f}"
        return None

    def collect(self, now_tick):
        keep = []
        for job in self.jobs:
            fut, planner, anchor_tick, anchor_q, anchor_dq, req_tick, t_req = job
            if not fut.done():
                keep.append(job)
                continue
            self.free_planners.append(planner)
            plan, lat = fut.result()
            self.latencies.append(lat)
            self.latency_est = float(np.quantile(self.latencies[-8:], 0.9))
            reason = self.sane(plan)
            late = now_tick - anchor_tick
            self.log.append(dict(tick=now_tick, req_tick=req_tick, anchor_tick=anchor_tick, latency_s=lat,
                                 late_ticks=late, rule=plan["rule"], rejected=reason))
            if reason or anchor_tick <= self.last_streamed_anchor:
                emit("plan_dropped", reason=reason or "stale", anchor_tick=anchor_tick)
                continue
            clipped = float(np.abs(plan["joint_velocity"]).max())
            if clipped > self.args.max_joint_vel:
                plan = dict(plan, joint_velocity=np.clip(plan["joint_velocity"], -self.args.max_joint_vel,
                                                         self.args.max_joint_vel))
                emit("dq_clipped", max_rad_s=clipped, anchor_tick=anchor_tick)
            self.active = (plan, anchor_tick, anchor_q, anchor_dq)
            self.last_streamed_anchor = anchor_tick
            if self.streamer is not None:
                start = max(anchor_tick, now_tick - 2)  # never before the runner's window start (playback - 5)
                s0, s1 = self.streamer.send_plan(plan, anchor_tick, anchor_q, anchor_dq, start, self.limits)
                self.stream_end = s1
        self.jobs = keep

    # -- main loop -------------------------------------------------------------------------------
    def run(self, state: DdsState, cloud: CloudInput, start_event: threading.Event | None = None):
        a = self.args
        t0 = time.monotonic()
        prescan_until, last_hist = None, None
        emit("node_up", inflight=a.inflight, route=a.route, goal=self.goal.tolist())
        while not self.stop_flag.is_set():
            tick_t = t0 + self.tick * DT
            if (rest := tick_t - time.monotonic()) > 0:
                time.sleep(rest)
            snap = state.snapshot()
            if snap is None:
                self.tick += 1
                continue
            q = snap["qpos"]
            if snap["age_low"] > 0.5 or snap["age_odo"] > 0.5:
                emit("estop", reason="state stale", age_low=snap["age_low"], age_odo=snap["age_odo"])
                return "stale_state"
            tilt = quat_tilt(snap["imu"])
            if self.phase == "RUN" and tilt > a.tilt_estop:
                emit("estop", reason="tilt", tilt=tilt)
                self.phase = "STOPPED"
                return "estop_tilt"
            if last_hist is None:
                self.history = self.flow.callback.backfill_history(q, snap["dq"], snap["lin_w"], snap["ang_w"], 4)
            self.history.append(q.astype(np.float32))
            self.history = self.history[-8:]
            last_hist = q
            if not hasattr(self, "memory"):
                self.make_memory(q[:2])
            for pts, origin, stamp in cloud.poll():
                self.memory.add_cloud(0.0, pts, origin, q)
            self.memory.step(self.tick * DT)
            cloud_age = time.monotonic() - cloud.last if cloud.last > 0 else 1e9

            if self.phase == "WAIT" and (start_event is None or start_event.is_set()):
                self.phase, prescan_until = "PRESCAN", self.tick + int(a.prescan_s / DT)
                emit("prescan")
            elif self.phase == "PRESCAN" and self.tick >= prescan_until and cloud_age < a.cloud_timeout:
                self.phase = "RUN"
                self.request(self.tick)
                self.jobs[0][0].result()  # first plan synchronously, while standing
                self.collect(self.tick)
                emit("run", tick=self.tick, first_plan_s=self.log[-1]["latency_s"] if self.log else None)
            elif self.phase == "RUN":
                self.collect(self.tick)
                dist = float(np.linalg.norm(q[:2] - self.goal))
                if dist <= a.goal_radius:
                    self.phase = "ARRIVED"
                    emit("arrived", tick=self.tick, dist=dist)
                    self.stand(self.tick, "arrived")
                elif cloud_age > a.cloud_timeout:
                    self.phase = "STOPPED"
                    self.stand(self.tick, f"cloud stale {cloud_age:.2f}s")
                elif self.free_planners and self.tick - self.last_request >= self.replan:
                    self.request(self.tick)
                if self.stream_end >= 0 and self.stream_end - self.tick < LOOKAHEAD_FRAMES:
                    emit("stall_risk", ahead=self.stream_end - self.tick)
            if self.phase == "ARRIVED" and self.tick % 50 == 0:
                pass  # the last plan ends standing near the goal; the orchestrator ends the episode
            if a.max_ticks and self.tick >= a.max_ticks:
                return "max_ticks"
            self.tick += 1
        return "stopped"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--goal", type=float, nargs=2, required=True, help="goal xy in the odometry frame")
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--odom", choices=("dds", "zmq"), default="dds")
    ap.add_argument("--cloud", default="tcp://localhost:5560")
    ap.add_argument("--stream-bind", default="tcp://*:5556")
    ap.add_argument("--route", choices=("sensor", "privileged"), default="sensor")
    ap.add_argument("--boxes", default=None, help="JSON list of boxes for --route privileged (a measured course)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--latency-guess-s", type=float, default=0.3)
    ap.add_argument("--prescan-s", type=float, default=0.5)
    ap.add_argument("--tilt-estop", type=float, default=0.75)
    ap.add_argument("--cloud-timeout", type=float, default=0.5)
    ap.add_argument("--min-pelvis-z", type=float, default=0.35)
    ap.add_argument("--max-speed", type=float, default=2.5,
                    help="mean planned pelvis speed bound (m/s); panels are walked at ~1.2-1.6 m/s")
    ap.add_argument("--max-joint-vel", type=float, default=30.0,
                    help="reference joint velocities are clipped to this (the runner aborts at 35 rad/s measured)")
    ap.add_argument("--goal-radius", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--dry-run", action="store_true", help="plan but do not open the stream socket")
    ap.add_argument("--start-file", default=None, help="wait until this file exists before the prescan")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    node = StudentNode(args)
    state = DdsState(args.iface, args.domain, args.odom)
    cloud = CloudInput(args.cloud)
    start = threading.Event()
    if args.start_file:
        def wait_file():
            while not Path(args.start_file).exists():
                time.sleep(0.05)
            start.set()

        threading.Thread(target=wait_file, daemon=True).start()
    else:
        start.set()
    import signal

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: node.stop_flag.set())
    result = "interrupted"
    try:
        result = node.run(state, cloud, start)
    finally:
        node.pool.shutdown(wait=False, cancel_futures=True)
        emit("node_exit", result=result, plans=len(node.log))
        if args.log:
            Path(args.log).write_text(json.dumps(dict(result=result, plans=node.log), indent=1, default=float) + "\n")
        if node.streamer is not None:
            node.streamer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
