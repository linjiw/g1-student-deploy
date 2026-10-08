#!/usr/bin/env python3
"""Real-time student planner node: G1 state + MID-360 cloud -> student planner -> SONIC ZMQ stream.

This is the process that runs next to the SONIC C++ runner, in the MuJoCo rehearsal and on the robot:

  inputs   rt/lowstate (DDS: 29 joints, pelvis IMU)          via unitree_sdk2py
           odometry: rt/odostate (DDS, sim ground truth) | `--odom zmq` (LIO bridge, docs/HARDWARE.md)
           cloud: ZMQ SUB topic `cloud` (points [N,3] in the odometry frame, origin [3], stamp)
  output   ZMQ PUB :5556 topic `pose`, SONIC protocol v1 (g1deploy/zmq_stream.py)
  events   JSON lines on stdout ({"event": ...}); `estop` asks the operator/orchestrator to stop the runner

Threads / processes (the first rehearsals showed one Python process cannot keep a 50 Hz loop while
planning and fusing clouds; the GIL starved the loop for up to 1.1 s):
  main thread        50 Hz: history (resampled at exact 20 ms steps from timestamped DDS samples),
                     supervisor, plan requests, streaming
  perception thread  cloud -> CloudMemory (self filter + carving, ~15 ms per cloud)
  planner processes  `--inflight` workers, each its own FlowPlanner; a job carries the history, a route
                     copy and a snapshot of the memory layers; plans are predictively anchored at their
                     landing tick (g1deploy/latency.py)
Phases: WAIT -> PRESCAN (fill memory while standing) -> RUN -> ARRIVED | STOPPED.

Safety supervisor (never relaxes): tilt > --tilt-estop -> estop; state/odometry stale > 0.5 s -> estop;
cloud stale > --cloud-timeout -> settle to stand; non-finite / out-of-range plans are rejected (the
previous plan keeps playing); reference joint velocities are clipped to --max-joint-vel.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import sys
import threading
import time
from collections import deque
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


# ---------------------------------------------------------------------------------------------- inputs
class DdsState:
    """rt/lowstate + odometry, kept as a 2 s ring of timestamped full-state samples."""

    def __init__(self, iface="lo", domain=0, odom="dds"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, OdoState_

        ChannelFactoryInitialize(domain, iface)
        self.lock = threading.Lock()
        self.low, self.odo, self.t_low, self.t_odo = None, None, -1.0, -1.0
        self.ring = deque(maxlen=1000)  # (t, qpos36, dq29)
        self.sub_low = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub_low.Init(self._on_low, 1)
        if odom == "dds":  # "zmq"/"leg"/"none": odometry arrives through set_odometry()
            self.sub_odo = ChannelSubscriber("rt/odostate", OdoState_)
            self.sub_odo.Init(self._on_odo, 1)

    def _on_low(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(29)])
        dq = np.array([msg.motor_state[i].dq for i in range(29)])
        imu = np.array(msg.imu_state.quaternion, dtype=float)  # w x y z
        gyro = np.array(msg.imu_state.gyroscope, dtype=float)
        now = time.monotonic()
        with self.lock:
            self.low, self.t_low = (q, dq, imu, gyro), now
            if self.odo is not None:
                self.ring.append((now, np.r_[self.odo[0], self.odo[1], q], dq))

    def _on_odo(self, msg):
        with self.lock:
            # gear_sonic_sim writes wxyz into `orientation` (despite the IDL comment)
            self.odo = (np.array(msg.position, dtype=float), np.array(msg.orientation, dtype=float),
                        np.array(msg.linear_velocity, dtype=float), np.array(msg.angular_velocity, dtype=float))
            self.t_odo = time.monotonic()

    def set_odometry(self, pos, quat_wxyz, lin_w, ang_w):
        """External odometry (LIO bridge) entry point."""
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

    def qpos_at(self, t):
        h = self.history(t, 1)
        return None if h is None else np.asarray(h[0], dtype=np.float64)

    def history(self, t_end, n):
        """n world qpos frames at exactly 20 ms spacing ending at t_end (linear in time, quats renormalised)."""
        with self.lock:
            if len(self.ring) < 2:
                return None
            ts = np.array([r[0] for r in self.ring])
            qs = np.stack([r[1] for r in self.ring])
        out = []
        for k in range(n):
            t = t_end - (n - 1 - k) * DT
            j = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
            a = float(np.clip((t - ts[j - 1]) / max(ts[j] - ts[j - 1], 1e-6), 0.0, 1.0))
            q0, q1 = qs[j - 1], qs[j]
            if np.dot(q0[3:7], q1[3:7]) < 0:
                q1 = q1.copy()
                q1[3:7] *= -1
            q = (1 - a) * q0 + a * q1
            q[3:7] /= np.linalg.norm(q[3:7])
            out.append(q.astype(np.float32))
        return out


class OdomInput:
    """ZMQ `odom` topic from the LIO bridge (g1deploy/adapters/ros2_lio_bridge.py) -> DdsState.set_odometry."""

    def __init__(self, state, connect="tcp://localhost:5561"):
        import zmq

        self.state = state
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"odom")
        self.sock.connect(connect)

    def poll(self):
        import zmq

        try:
            msg = self.sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        _, f = unpack(msg, topic=b"odom")
        self.state.set_odometry(f["pos"], f["quat_wxyz"], f["lin_w"], f["ang_w"])


class LegOdomSource:
    """`--odom leg`: pelvis pose from joints + pelvis IMU at 200 Hz (g1deploy/leg_odometry.py), no LIO needed.
    The odometry frame starts at the first sample: origin under the pelvis, yaw from the IMU."""

    def __init__(self, state: "DdsState"):
        from g1deploy.leg_odometry import LegOdometry

        self.state, self.odo = state, LegOdometry()
        self.stop = threading.Event()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        last_p, last_t = None, None
        while not self.stop.is_set():
            with self.state.lock:
                low = self.state.low
            if low is not None:
                q, _, imu, gyro = low
                p = self.odo.reset((0.0, 0.0), imu, q) if self.odo.p is None else self.odo.update(imu, q)
                now = time.monotonic()
                v = np.zeros(3) if last_p is None else (p - last_p) / max(now - last_t, 1e-3)
                last_p, last_t = p, now
                r = np.array(imu, dtype=float)
                self.state.set_odometry(p, r, v, np.asarray(gyro, dtype=float))
            time.sleep(0.005)


class DdsCloudInput:
    """The G1's own lidar topic (rt/utlidar/cloud_livox_mid360): raw sensor frame, per-point time, placeholders.
    Frames are decoded on the DDS thread; header stamps are ignored (PC1 clock), points are timed on arrival."""

    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

        from g1deploy.perception.unitree_lidar import TOPIC_CLOUD

        self.lock, self.queue, self.last = threading.Lock(), [], -1.0
        self.sub = ChannelSubscriber(TOPIC_CLOUD, PointCloud2_)
        self.sub.Init(self._on, 2)

    def _on(self, msg):
        from g1deploy.perception.unitree_lidar import decode_cloud

        recv = time.monotonic()
        pts, t_s, _ = decode_cloud(msg)
        with self.lock:
            self.queue.append(dict(raw=True, points=pts, t_s=t_s, wall=recv))
            self.queue = self.queue[-3:]
            self.last = recv

    def poll(self):
        with self.lock:
            out, self.queue = self.queue, []
        return out


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
            out.append(dict(points=f["points"].astype(np.float64), origin=f["origin"], stamp=float(f["stamp"][0]),
                            origins=f.get("origins"), dt=f.get("dt"), wall=float(f["wall"][0]) if "wall" in f else self.last))


# ------------------------------------------------------------------------------------------ planners
def memory_view(memory):
    """Picklable snapshot of the memory layers with the parent class's ego_map()."""
    flow = bundle.load_flow()
    view = flow.sensor_map.SensorMapMemory.__new__(flow.sensor_map.SensorMapMemory)
    view.support, view.ceiling = memory.support.copy(), memory.ceiling.copy()
    view.lo, view.res, view.n = memory.lo.copy(), memory.res, memory.n
    return view


def route_copy(route):
    r = copy.copy(route)
    if hasattr(r, "memory"):
        r.memory = None
    return r


def _worker(conn, pdict, episode, seed):
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent shuts workers down through the pipe

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import torch

    torch.set_num_threads(4)
    flow = bundle.load_flow()
    from sim.lockstep import make_goal_fn

    planner = flow.planner.FlowPlanner(flow.planner.PlannerConfig(**dict(pdict, seed=seed)))
    pc = planner.cfg
    sm, cb = flow.sensor_map, flow.callback
    conn.send("ready")
    while True:
        job = conn.recv()
        if job is None:
            break
        hist, route, view = job["hist"], job["route"], job["memory"]
        t0 = time.monotonic()
        sel = sm.HeightFieldSelector(view, planner.certifier.tree, planner.certifier.bodies, planner.shapes)

        def map_fn(xy, yaw):
            m, _ = view.ego_map(xy, yaw, pc.datum_offset_m, pc.support_clip_m or 0.8)
            return m

        guide = sm.memory_boxes(view, grow_overhead_m=pc.guide_grow_overhead_m) if pc.guide_w > 0 else None
        goal_fn = make_goal_fn(route, episode["goal_lookahead_m"], episode["goal_heading"], episode["steer_lookahead_m"],
                               None, cb.goal_vector, cb.route_goal_heading)
        try:
            plan = planner.plan(np.asarray(hist), [], goal_fn, map_fn, sel, prev=None, guide_boxes=guide)
            out = {k: plan[k] for k in ("qpos", "rotation", "joint_velocity", "rule", "chosen_clearance_m")}
        except Exception as exc:  # reported, never fatal to the node
            out = {"error": f"{type(exc).__name__}: {exc}"}
        conn.send((job["id"], out, time.monotonic() - t0))


class PlannerPool:
    """`n` planner processes (spawn); each holds its own CUDA context (~0.4 GB) and model copy."""

    def __init__(self, n, pdict, episode, seed):
        ctx = mp.get_context("spawn")
        self.workers = []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, pdict, episode, seed + i), daemon=True)
            p.start()
            self.workers.append({"proc": p, "conn": a, "busy": None})
        for w in self.workers:
            if w["conn"].recv() != "ready":
                raise RuntimeError("planner worker failed to start")

    def free(self):
        return [w for w in self.workers if w["busy"] is None]

    def submit(self, w, job):
        w["busy"] = job["id"]
        w["conn"].send(job)

    def poll(self):
        done = []
        for w in self.workers:
            if w["busy"] is not None and w["conn"].poll():
                jid, out, lat = w["conn"].recv()
                w["busy"] = None
                done.append((jid, out, lat))
        return done

    def close(self):
        for w in self.workers:
            try:
                w["conn"].send(None)
            except Exception:
                pass
        for w in self.workers:
            w["proc"].join(timeout=2)
            if w["proc"].is_alive():
                w["proc"].kill()


# ---------------------------------------------------------------------------------------------- node
class StudentNode:
    def __init__(self, args):
        self.args = args
        self.flow = bundle.load_flow()
        cfg = bundle.student_config()
        self.episode = dict(cfg["episode"])
        if args.goal_lookahead:
            self.episode["goal_lookahead_m"] = args.goal_lookahead
        self.replan = int(self.episode["replan_ticks"])
        variant = cfg["route_variants"][args.route]
        self.route_cfg = variant.get("route", {})
        self.pdict = dict(cfg["planner"], device=args.device)
        self.pcfg = self.flow.planner.PlannerConfig(**self.pdict)
        contract = json.loads((bundle.ROOT / "config/sonic_model_contract.json").read_text())
        self.limits = np.asarray([j["range_rad"] for j in contract["joints"] if j["type"] != "free"])
        self.jobs = {}  # id -> (anchor_tick, anchor_q, anchor_dq, request_tick, first)
        self.next_id = 0
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
        self.mem_lock = threading.Lock()
        self.overruns, self.skipped, self.max_lag_s = 0, 0, 0.0
        self.pool = PlannerPool(args.inflight, self.pdict, self.episode, args.seed)

    # -- perception ---------------------------------------------------------------------------
    def make_memory(self, xy):
        from g1deploy.perception.cloud_memory import CloudMemory

        self.memory = CloudMemory(xy, rng=np.random.default_rng(self.args.seed), **dict(self.pcfg.sensor))
        self.memory.latency = 0.0  # real latency is already in the data; fuse on arrival
        from g1deploy.perception.unitree_lidar import CloudFromSensorFrame, Mount

        mount = Mount.load(self.args.mount_calib) if self.args.mount_calib else Mount.nominal()
        self.from_sensor = CloudFromSensorFrame(self.memory.body_filter, mount, transport_s=self.args.lidar_transport_s)

    def perception_loop(self, cloud, state, t0, stop):
        """Clouds -> memory, and the route refresh (A* on the memory grid, up to ~0.2 s) - both off the 50 Hz loop."""
        while not self.stop_flag.is_set() and not stop.is_set():
            got = cloud.poll()
            snap = state.snapshot()
            if got and snap is not None:
                for c in got:
                    if c.get("raw"):  # sensor-frame frame from the robot's DDS topic: deskew + register ourselves
                        pts, org = self.from_sensor(c["points"], c["t_s"], c["wall"], state.qpos_at)
                        with self.mem_lock:
                            self.memory.add_cloud(0.0, pts, org, None)
                        continue
                    pts, org = c["points"], c["origins"] if c["origins"] is not None else c["origin"]
                    if c["dt"] is not None and len(pts):
                        # self filter per firing-time slice with the joint state of that instant (MID-360 points
                        # carry their own time; one pose per 0.1 s frame leaks swinging hands into the memory)
                        keep = np.ones(len(pts), dtype=bool)
                        for dt in np.unique(c["dt"]):
                            sel = c["dt"] == dt
                            q = state.qpos_at(c["wall"] + float(dt))
                            if q is not None:
                                keep[sel] = self.memory.body_filter.mask(pts[sel], q)
                        pts = pts[keep]
                        org = np.broadcast_to(org, (len(keep), 3))[keep]
                        qf = None
                    else:
                        qf = snap["qpos"]
                    with self.mem_lock:
                        self.memory.add_cloud(0.0, pts, org, qf)
                with self.mem_lock:
                    self.memory.step(time.monotonic() - t0)
            if snap is not None and self.phase in ("PRESCAN", "RUN"):
                xy, t = np.asarray(snap["qpos"][:2], dtype=float), time.monotonic() - t0
                if not hasattr(self, "route"):
                    if self.args.route == "sensor":
                        with self.mem_lock:
                            r = self.flow.route.SensorRoute(xy, self.goal, self.memory, **self.route_cfg)
                            r.refresh(xy, t, force=True)
                    else:
                        boxes = json.loads(Path(self.args.boxes).read_text()) if self.args.boxes else []
                        r = self.flow.route.Route(xy, self.goal, boxes, **self.route_cfg)
                    self.route = r
                elif self.args.route == "sensor" and t - self.route.last_refresh >= self.route.period_s:
                    with self.mem_lock:
                        fresh = copy.copy(self.route)
                        fresh.refresh(xy, t, force=True)  # assigns new arrays on the copy; readers keep theirs
                    self.route = fresh
            time.sleep(0.005)

    def route_now(self, xy, t):
        return route_copy(self.route)

    # -- planning --------------------------------------------------------------------------------
    def request(self, now_tick, worker):
        first = self.active is None
        lead = 0 if first else int(math.ceil(self.latency_est / DT)) + 1
        hist = self.history if lead == 0 else predicted_history(self.history, self.active, now_tick, lead, len(self.history))
        route = self.route_now(np.asarray(self.history[-1][:2], dtype=float), now_tick * DT)
        with self.mem_lock:
            view = memory_view(self.memory)
        jid = self.next_id
        self.next_id += 1
        self.pool.submit(worker, {"id": jid, "hist": [np.asarray(h) for h in hist], "route": route, "memory": view})
        self.jobs[jid] = (now_tick + lead, np.asarray(hist[-1], dtype=np.float64),
                          self.flow.feed.finite_difference_dq(hist), now_tick, first)
        self.last_request = now_tick

    def sane(self, plan):
        q = plan["qpos"]
        if not np.isfinite(q).all() or not np.isfinite(plan["joint_velocity"]).all():
            return "nonfinite"
        if q[:, 2].min() < self.args.min_pelvis_z or q[:, 2].max() > 1.0:
            return f"pelvis_z {q[:, 2].min():.2f}..{q[:, 2].max():.2f}"
        # Planned pelvis xy sways (0.2 s-window peaks 1.7-2.1 m/s in Isaac-evaluated plans) and SONIC does not
        # read root position; bound the plan's mean speed instead.
        v = float(np.linalg.norm(q[-1, :2] - q[0, :2]) / (DT * (len(q) - 1)))
        if v > self.args.max_speed:
            return f"mean speed {v:.2f}"
        return None

    def collect(self, now_tick):
        for jid, out, lat in self.pool.poll():
            if jid not in self.jobs:
                continue  # discarded (stand issued meanwhile)
            anchor_tick, anchor_q, anchor_dq, req_tick, first = self.jobs.pop(jid)
            if "error" in out:
                emit("plan_error", error=out["error"])
                continue
            plan = out
            if first:
                anchor_tick = now_tick  # computed while standing still: start it when it lands
            self.latencies.append(lat)
            self.latency_est = float(np.quantile(self.latencies[-8:], 0.9))
            reason = self.sane(plan)
            self.log.append(dict(tick=now_tick, req_tick=req_tick, anchor_tick=anchor_tick, latency_s=lat,
                                 late_ticks=now_tick - anchor_tick, rule=plan["rule"], rejected=reason))
            if reason or anchor_tick <= self.last_streamed_anchor:
                emit("plan_dropped", reason=reason or "stale", anchor_tick=anchor_tick)
                continue
            peak = float(np.abs(plan["joint_velocity"]).max())
            if peak > self.args.max_joint_vel:
                plan = dict(plan, joint_velocity=np.clip(plan["joint_velocity"], -self.args.max_joint_vel,
                                                         self.args.max_joint_vel))
                emit("dq_clipped", max_rad_s=peak, anchor_tick=anchor_tick)
            self.active = (plan, anchor_tick, anchor_q, anchor_dq)
            self.last_streamed_anchor = anchor_tick
            if self.streamer is not None:
                start = max(anchor_tick, now_tick - 2)  # never before the runner's window start (playback - 5)
                self.stream_end = self.streamer.send_plan(plan, anchor_tick, anchor_q, anchor_dq, start, self.limits)[1]
            if first:
                emit("run", tick=now_tick, first_plan_s=lat)

    def ceiling_near(self, xy, radius_m=0.8, below_m=1.35):
        """Is a remembered overhead obstacle lower than standing head height within `radius_m`? (Standing up there
        drives the head into it: verified in the rehearsal, 566 N, operator takeover under a beam.)"""
        m = self.memory
        with self.mem_lock:
            c = m.ceiling
            ij = np.round((np.asarray(xy) - m.lo) / m.res).astype(int)
            k = int(np.ceil(radius_m / m.res))
            a0, a1 = max(ij[0] - k, 0), min(ij[0] + k + 1, m.n)
            b0, b1 = max(ij[1] - k, 0), min(ij[1] + k + 1, m.n)
            win = c[a0:a1, b0:b1]
            return bool(np.any(np.isfinite(win) & (win < below_m)))

    def stand(self, now_tick, reason, settle_s=1.0, hold_s=3.0):
        """Stream a settle-to-stand chunk from the measured pose: joints -> SONIC default pose over
        `settle_s`, pelvis yaw kept, then hold. Used at arrival and as the soft stop. Near a remembered ceiling the
        stand is deferred: the active plan (checked against the memory when it was made) keeps playing and the
        stand is issued once clear (`self.pending_stand`)."""
        q = np.asarray(self.history[-1], dtype=np.float64)
        if self.active is not None and self.ceiling_near(q[:2]):
            if getattr(self, "pending_stand", None) is None:
                emit("stand_deferred", reason=reason, tick=now_tick)
            self.pending_stand = reason
            self.jobs = {}
            self.last_streamed_anchor = now_tick + 10**6  # no new plans; the current one plays out
            return
        self.pending_stand = None
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
        self.jobs = {}  # results of in-flight plans are discarded from now on
        self.active = (plan, now_tick, q, np.zeros(29))
        self.last_streamed_anchor = now_tick + 10**6
        if self.streamer is not None:
            self.stream_end = self.streamer.send_plan(plan, now_tick, q, np.zeros(29), now_tick, self.limits)[1]
        emit("stand", reason=reason, tick=now_tick)

    # -- main loop -------------------------------------------------------------------------------
    def reset_run_state(self):
        """Everything one run accumulates; a restarted run (after a stop or stand) must not inherit it - e.g. a
        stand pushes last_streamed_anchor far ahead, which would reject every later plan as stale."""
        self.tick, self.phase, self.active, self.jobs = 0, "WAIT", None, {}
        self.pending_stand = None
        self.last_streamed_anchor, self.last_request, self.stream_end = -10**9, -10**9, -1
        if self.streamer is not None:
            self.streamer.last_end = None  # the runner resets playback on the new run's first chunk anyway
        for jid, _, _ in self.pool.poll():  # results of the previous run are dropped
            pass

    def run(self, state: DdsState, cloud: CloudInput, start_event: threading.Event | None = None, odom=None):
        a = self.args
        self.reset_run_state()
        t0 = time.monotonic()
        prescan_until = None
        emit("node_up", inflight=a.inflight, route=a.route, goal=self.goal.tolist())
        while not self.stop_flag.is_set():
            tick_t = t0 + self.tick * DT
            if (rest := tick_t - time.monotonic()) > 0:
                time.sleep(rest)
            lag = time.monotonic() - tick_t
            self.max_lag_s = max(self.max_lag_s, lag)
            if lag > DT:
                self.overruns += 1
                if lag > 3 * DT:  # fell behind: jump to the wall clock (history is resampled by time anyway)
                    skip = int(lag / DT)
                    self.tick += skip
                    self.skipped += skip
            if odom is not None:
                odom.poll()
            snap = state.snapshot()
            hist = state.history(time.monotonic(), 6) if snap is not None else None
            if snap is None or hist is None:
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
            self.history = hist
            if not hasattr(self, "memory"):
                self.make_memory(q[:2])
            if getattr(self, "perc_cloud", None) is not cloud:
                # one perception thread per run(), bound to this run's lidar source; a previous run's thread is
                # stopped first (it would otherwise keep polling the old source)
                if getattr(self, "perc_thread", None) is not None:
                    self.perc_stop.set()
                    self.perc_thread.join(1.0)
                self.perc_stop, self.perc_cloud = threading.Event(), cloud
                self.perc_thread = threading.Thread(target=self.perception_loop, args=(cloud, state, t0, self.perc_stop),
                                                    daemon=True)
                self.perc_thread.start()
            cloud_age = time.monotonic() - cloud.last if cloud.last > 0 else 1e9

            if self.phase == "WAIT" and (start_event is None or start_event.is_set()):
                self.phase, prescan_until = "PRESCAN", self.tick + int(a.prescan_s / DT)
                emit("prescan")
            elif (self.phase == "PRESCAN" and self.tick >= prescan_until and cloud_age < a.cloud_timeout
                  and hasattr(self, "route")):
                self.collect(self.tick)  # drain results still in flight from a previous run (they are dropped)
                if self.pool.free():
                    self.phase = "RUN"
                    self.request(self.tick, self.pool.free()[0])
            elif self.phase == "RUN":
                self.collect(self.tick)
                dist = float(np.linalg.norm(q[:2] - self.goal))
                if dist <= a.goal_radius and self.active is not None:
                    self.phase = "ARRIVED"
                    emit("arrived", tick=self.tick, dist=dist)
                    self.stand(self.tick, "arrived")
                elif cloud_age > a.cloud_timeout:
                    self.phase = "STOPPED"
                    self.stand(self.tick, f"cloud stale {cloud_age:.2f}s")
                elif self.active is not None and self.pool.free() and self.tick - self.last_request >= self.replan:
                    self.request(self.tick, self.pool.free()[0])
                if self.active is not None and 0 <= self.stream_end - self.tick < LOOKAHEAD_FRAMES:
                    emit("stall_risk", ahead=self.stream_end - self.tick)
            if getattr(self, "pending_stand", None) is not None:
                self.stand(self.tick, self.pending_stand)
            if a.max_ticks and self.tick >= a.max_ticks:
                return "max_ticks"
            self.tick += 1
        return "stopped"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--goal", type=float, nargs=2, required=True, help="goal xy in the odometry frame")
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--odom", choices=("dds", "zmq", "leg"), default="dds",
                    help="dds: rt/odostate (simulator ground truth); zmq: LIO bridge topic `odom`; "
                         "leg: leg odometry from rt/lowstate only (fallback; ~12%% drift in MuJoCo)")
    ap.add_argument("--goal-frame", choices=("odom", "start"), default="odom",
                    help="start: --goal is (ahead, left) metres from the robot's start pose")
    ap.add_argument("--odom-connect", default="tcp://localhost:5561")
    ap.add_argument("--cloud", default="tcp://localhost:5560",
                    help="'dds': the G1's own rt/utlidar/cloud_livox_mid360; otherwise a ZMQ endpoint with world clouds")
    ap.add_argument("--mount-calib", default=None, help="calibrated MID-360 mount JSON (tools/calibrate_mount.py)")
    ap.add_argument("--lidar-transport-s", type=float, default=0.01, help="driver + network delay of the last point")
    ap.add_argument("--stream-bind", default="tcp://*:5556")
    ap.add_argument("--route", choices=("sensor", "privileged"), default="sensor")
    ap.add_argument("--boxes", default=None, help="JSON list of boxes for --route privileged (a measured course)")
    ap.add_argument("--goal-lookahead", type=float, default=None, help="override (sets walking speed; 1.5 evaluated)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inflight", type=int, default=1,
                    help="2+ is harmful: a second in-flight plan is predicted from the landed plan, not the one "
                         "running in between (G4: 32/98 vs 79/98)")
    ap.add_argument("--latency-guess-s", type=float, default=0.3)
    ap.add_argument("--prescan-s", type=float, default=0.5)
    ap.add_argument("--tilt-estop", type=float, default=0.75)
    ap.add_argument("--cloud-timeout", type=float, default=1.0,
                    help="no lidar frame for this long -> settle to stand (10 missed frames at 10 Hz)")
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
    ap.add_argument("--dump-memory", default=None, help="write the lidar memory layers + boxes here at exit (npz)")
    ap.add_argument("--start-file", default=None, help="wait until this file exists before the prescan")
    return ap.parse_args(argv)


def main(argv=None):
    import signal

    args = parse(argv)
    node = StudentNode(args)
    state = DdsState(args.iface, args.domain, args.odom)
    if args.odom == "leg":
        LegOdomSource(state)
    if args.goal_frame == "start":
        t0 = time.monotonic()
        while state.snapshot() is None and time.monotonic() - t0 < 10:
            time.sleep(0.02)
        snap = state.snapshot()
        if snap is None:
            raise RuntimeError("no robot state to anchor the goal")
        w, x, y, z = snap["qpos"][3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s_ = math.cos(yaw), math.sin(yaw)
        ahead, left = args.goal
        node.goal = snap["qpos"][:2] + np.array([c * ahead - s_ * left, s_ * ahead + c * left])
        emit("goal", frame="start", goal_odom=node.goal.tolist())
    cloud = DdsCloudInput() if args.cloud == "dds" else CloudInput(args.cloud)
    start = threading.Event()
    if args.start_file:
        def wait_file():
            while not Path(args.start_file).exists():
                time.sleep(0.05)
            start.set()

        threading.Thread(target=wait_file, daemon=True).start()
    else:
        start.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: node.stop_flag.set())
    result = "interrupted"
    try:
        result = node.run(state, cloud, start, OdomInput(state, args.odom_connect) if args.odom == "zmq" else None)
    finally:
        node.stop_flag.set()
        emit("node_exit", result=result, plans=len(node.log), overruns=node.overruns, skipped=node.skipped,
             ticks=node.tick, max_lag_s=round(node.max_lag_s, 4))
        if args.log:
            Path(args.log).write_text(json.dumps(dict(result=result, plans=node.log, overruns=node.overruns,
                                                      skipped=node.skipped, ticks=node.tick, max_lag_s=node.max_lag_s),
                                                 indent=1, default=float) + "\n")
        if args.dump_memory and hasattr(node, "memory"):
            with node.mem_lock:
                boxes = node.flow.sensor_map.memory_boxes(node.memory, grow_overhead_m=node.pcfg.guide_grow_overhead_m)
                np.savez_compressed(args.dump_memory, support=node.memory.support, ceiling=node.memory.ceiling,
                                    lo=node.memory.lo, res=node.memory.res, boxes=json.dumps(boxes),
                                    stats=json.dumps(node.memory.stats))
        node.pool.close()
        if node.streamer is not None:
            node.streamer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
