#!/usr/bin/env python3
"""Passive session recorder: everything the robot, the runner and the node exchange, on disk, for research.

Subscribe-only. It never publishes on DDS or ZMQ and never touches the node's commands (its only request is
`status`), so it cannot change what the robot does. Runs as its own process next to the runner and the node,
started by the console (`g1 sim` / `g1 robot`) or by hand:

    recorder.py --iface <nic> --out <session>/rec [--no-lidar] [--seconds S]

Streams (each one a directory of chunk files <out>/<stream>/NNNNNN.npz, written every --flush-s, so a crash or
power loss loses at most one chunk):
    lowstate     rt/lowstate 500 Hz: tick, modes, pelvis IMU, all 35 motors (q dq ddq tau_est temps vol state),
                 the wireless remote bytes
    lowcmd       rt/lowcmd: what the SONIC runner commands (q dq tau kp kd mode per motor)
    secondary_imu  rt/secondary_imu (torso IMU) when published
    odostate     rt/odostate (simulator ground truth; absent on the robot)
    lidar        rt/utlidar/cloud_livox_mid360 raw PointCloud2 frames, bytes untouched (decode with
                 g1deploy.recording.lidar_frames)
    lidar_imu    rt/utlidar/imu_livox_mid360
    zmq_pose     node -> runner reference stream (:5556), raw messages (g1deploy.zmq_stream.unpack)
    zmq_debug    runner g1_debug (:5557), raw messages
    zmq_odom     LIO bridge odometry (:5561) when running
    zmq_cloud    LIO bridge registered cloud (:5561, topic `cloud`) when running, compressed chunks
    node_status  node.py status (phase, goal, odometry xy, tilt, latency, ...) at 5 Hz, node_status.jsonl
Every row carries `t` (time.monotonic(), the clock of the console/node/runner logs) and `t_unix`.

Recording health (CONTRACT v3 G; read by g1deploy/evidence.py recording_health): `stats.json` is rewritten every second
(the recorder's heartbeat: pid, t_unix, disk free, per stream samples, rate over the last 5 s, last sample and last
chunk write times, gaps = silences longer than max(3 expected periods, 0.1 s), samples dropped and why, chunks, bytes,
write errors); `meta.json` describes the recording (lidar_fields as soon as the first frame arrives); `finalized.json`
is written only at a clean exit, after the last chunk: without it a reader knows the recording was cut (power loss,
kill) and that everything up to the last whole chunk is still usable.

Faults are reported, never allowed to stop the recording or anything else: a chunk that cannot be written (disk full,
I/O error) is dropped and counted; samples beyond a bounded queue (a writer that falls behind) are dropped and counted;
below --min-free-gb of free disk or past --quota-gb the recorder keeps its heartbeat but drops (and counts) samples
instead of filling the disk. If the console that started it disappears, it keeps recording --orphan-grace-s (the
aftermath) and then finalizes by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMAT = "g1deploy.recording.v1"  # the chunk files (unchanged)
STATS_FORMAT = "g1deploy.recording.stats.v2"
N_MOTORS = 35
# The rate each stream comes at (Hz; None: event-driven, e.g. the student stream flows only while walking). Shown in
# the stats and used for the gap threshold; g1deploy/evidence.py decides which streams a session requires.
NOMINAL_HZ = {"lowstate": 500.0, "lowcmd": 500.0, "secondary_imu": 500.0, "odostate": 200.0, "lidar": 10.0,
              "lidar_imu": 200.0, "zmq_pose": None, "zmq_debug": 50.0, "zmq_odom": 50.0, "zmq_cloud": 10.0}
SIM_HZ = {"lowstate": 200.0, "secondary_imu": 200.0, "odostate": 200.0}  # robot_sim's physics rate (iface lo)
GAP_FLOOR_S = 0.1  # a Python subscriber's receive times jitter by tens of ms: a shorter silence is never a gap
RATE_WINDOW_S = 5.0
PENDING_MB = {"lidar": 256, "zmq_cloud": 256}  # rows a stream may hold for the writer before it drops (default below)
DEFAULT_PENDING_MB = 64
disk_usage = shutil.disk_usage  # (tests replace it)


def _nbytes(v):
    if isinstance(v, (bytes, bytearray, memoryview)):
        return len(v)
    return getattr(v, "nbytes", 8)


class StreamStats:
    """Counters of one stream, updated by its receive thread and read by the stats loop (all under self.lock)."""

    def __init__(self, name, expected_hz=None):
        self.name, self.expected_hz = name, expected_hz
        self.gap_s = max(3.0 / expected_hz, GAP_FLOOR_S) if expected_hz else None
        self.lock = threading.Lock()
        self.count, self.bytes, self.first, self.last, self.max_gap = 0, 0, None, None, 0.0
        self.gaps, self.gap_total = 0, 0.0
        self.dropped, self.drop_reasons = 0, {}
        self.write_errors, self.last_error = 0, None
        self.last_sample_unix = self.last_write_unix = None
        self.paused = None  # why samples are not kept now ("quota", "disk low"): they are counted as dropped
        self.closed = False  # after the final flush: late samples are ignored (not a loss)
        self.pending = 0  # rows waiting for (or being written by) the writer
        self.hist = deque(maxlen=2 * int(RATE_WINDOW_S) + 4)  # (t, count) once per stats call: the 5 s rate

    def _seen(self, t):
        """A sample arrived at t (lock held): rate, gaps."""
        if self.last is not None:
            dt = t - self.last
            self.max_gap = max(self.max_gap, dt)
            if self.gap_s is not None and dt > self.gap_s:
                self.gaps += 1
                self.gap_total += dt
        self.first = t if self.first is None else self.first
        self.last = t
        self.last_sample_unix = time.time()
        self.count += 1

    def _drop(self, why, n=1):
        self.dropped += n
        self.drop_reasons[why] = self.drop_reasons.get(why, 0) + n

    def failed(self, exc, n):
        """A write of n rows failed: they are lost, counted."""
        with self.lock:
            self.write_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._drop("write error", n)

    def stats(self, now, now_unix=None):
        now_unix = time.time() if now_unix is None else now_unix
        with self.lock:
            self.hist.append((now, self.count))
            t0, c0 = next(((t, c) for t, c in self.hist if now - t <= RATE_WINDOW_S + 1e-3), (now, self.count))
            span = (self.last - self.first) if self.count > 1 else 0.0
            mean = round((self.count - 1) / span, 2) if span > 0 else None
            win = round((self.count - c0) / (now - t0), 2) if now - t0 >= 0.9 else None
            return {"count": self.count, "samples": self.count, "rate_hz": win if win is not None else mean,
                    "rate_mean_hz": mean, "expected_hz": self.expected_hz,
                    "age_s": round(now - self.last, 3) if self.last is not None else None,
                    "first_t": self.first, "last_t": self.last,
                    "last_sample_unix": self.last_sample_unix, "last_write_unix": self.last_write_unix,
                    "max_gap_s": round(self.max_gap, 4), "gap_s": self.gap_s, "gaps": self.gaps,
                    "gap_total_s": round(self.gap_total, 3), "dropped": self.dropped, "drop_reasons": dict(self.drop_reasons),
                    "pending": self.pending, "chunks": getattr(self, "seq", None), "bytes": self.bytes,
                    "write_errors": self.write_errors, "last_error": self.last_error, "paused": self.paused}


class ChunkWriter(StreamStats):
    """Rows of named fields -> <root>/<name>/NNNNNN.npz every flush_s, written atomically off the receive thread.

    A field is a scalar, a fixed-shape array, or `bytes` (variable length: stored concatenated as <key> with
    <key>_off offsets of length rows + 1). The rows waiting for the writer are bounded (max_pending_bytes): beyond
    that a sample is dropped and counted, so a slow disk never grows the recorder without limit."""

    def __init__(self, root: Path, name: str, compress=False, expected_hz=None, max_pending_bytes=DEFAULT_PENDING_MB * 1e6):
        super().__init__(name, expected_hz)
        self.dir = Path(root) / name
        self.compress, self.max_pending = compress, max_pending_bytes
        self.rows: dict[str, list] = {}
        self.pending_bytes = 0
        self.seq = 0

    def add(self, t, **fields):
        size = 16 + sum(_nbytes(v) for v in fields.values())
        with self.lock:
            if self.closed:
                return
            self._seen(t)
            if self.paused or self.pending_bytes + size > self.max_pending:
                self._drop(self.paused or "queue full")
                return
            self.pending += 1
            self.pending_bytes += size
            self.rows.setdefault("t", []).append(t)
            self.rows.setdefault("t_unix", []).append(self.last_sample_unix)
            for k, v in fields.items():
                self.rows.setdefault(k, []).append(v)

    def take(self, close=False):
        """The rows so far (the writer's next chunk); close=True: the last take, later samples are ignored."""
        with self.lock:
            rows, self.rows, self.pending_bytes = self.rows, {}, 0
            self.closed = self.closed or close
        return rows

    def write(self, rows):
        n = len(rows.get("t") or []) if rows else 0
        if not n:
            return
        out = {}
        for k, vals in rows.items():
            if isinstance(vals[0], (bytes, bytearray, memoryview)):
                lens = np.fromiter((len(v) for v in vals), dtype=np.int64, count=len(vals))
                out[k + "_off"] = np.r_[0, np.cumsum(lens)]
                out[k] = np.frombuffer(b"".join(bytes(v) for v in vals), dtype=np.uint8)
            else:
                out[k] = np.asarray(vals)
        path = self.dir / f"{self.seq:06d}.npz"
        tmp = path.with_suffix(".tmp.npz")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (np.savez_compressed if self.compress else np.savez)(tmp, **out)
            os.replace(tmp, path)
            size = path.stat().st_size
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)  # a half-written chunk never looks like a chunk
            except OSError:
                pass
            raise
        finally:
            with self.lock:
                self.pending = max(0, self.pending - n)
        with self.lock:
            self.bytes += size
            self.seq += 1
            self.last_write_unix = time.time()


class LineWriter(StreamStats):
    """A JSON-lines stream (node status), written line by line by its own thread; same counters as a ChunkWriter."""

    def __init__(self, path: Path, name: str, expected_hz=None):
        super().__init__(name, expected_hz)
        self.path, self.f = Path(path), None

    def add(self, t, row):
        with self.lock:
            if self.closed:
                return
            self._seen(t)
            if self.paused:
                self._drop(self.paused)
                return
        line = json.dumps(row) + "\n"
        try:
            if self.f is None:
                self.f = open(self.path, "a", buffering=1)
            self.f.write(line)
        except (OSError, ValueError) as exc:
            self.failed(exc, 1)
            return
        with self.lock:
            self.bytes += len(line)
            self.last_write_unix = time.time()

    def close(self):
        with self.lock:
            self.closed = True
        try:
            if self.f is not None:
                self.f.close()
        except OSError as exc:
            self.failed(exc, 0)


class Recorder:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.writers: dict[str, ChunkWriter] = {}
        self.writers_lock = threading.Lock()
        self.sim = getattr(args, "iface", None) == "lo"
        self.status = LineWriter(self.out / "node_status.jsonl", "node_status", getattr(args, "status_hz", None) or None)
        self.errors: dict[str, str] = {}  # where -> the last error there (kept: a fault seen stays visible)
        self.error_log = deque(maxlen=20)
        self.lidar_fields = None
        self.meta, self.meta_dirty = {}, False
        self.paused = None  # "quota" | "disk low": every stream drops (and counts) instead of writing
        self.quota = (getattr(args, "quota_gb", 0) or 0) * 1e9 or None
        self.min_free = (getattr(args, "min_free_gb", 0) or 0) * 1e9
        self.disk = {"free_bytes": None, "total_bytes": None}
        self.ppid0, self.orphan_t, self.stop_reason = os.getppid(), None, None
        self.flush_started = self.last_flush_unix = self.last_flush_s = None
        self.t0 = time.monotonic()

    def error(self, where, msg):
        self.errors[where] = msg
        self.error_log.append({"t_unix": round(time.time(), 3), "where": where, "error": msg})

    def request_stop(self, reason):
        self.stop_reason = self.stop_reason or reason
        self.stop.set()

    def writer(self, name, compress=False):
        w = self.writers.get(name)
        if w is None:
            with self.writers_lock:  # first sample of a stream: callbacks of several threads may race here
                w = self.writers.get(name)
                if w is None:
                    hz = (SIM_HZ.get(name) if self.sim else None) or NOMINAL_HZ.get(name)
                    mb = PENDING_MB.get(name, DEFAULT_PENDING_MB) * (getattr(self.args, "pending_scale", 1.0) or 1.0)
                    w = ChunkWriter(self.out, name, compress, hz, mb * 1e6)
                    w.paused = self.paused
                    self.writers[name] = w
        return w

    # -- DDS ------------------------------------------------------------------------------------------
    def start_dds(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_, OdoState_

        ChannelFactoryInitialize(self.args.domain, self.args.iface)
        self.subs = []

        def sub(topic, typ, cb, depth=10):
            s = ChannelSubscriber(topic, typ)
            s.Init(self._guard(topic, cb), depth)
            self.subs.append(s)

        w_low, w_cmd = self.writer("lowstate"), self.writer("lowcmd")

        def on_low(m):
            ms, imu = m.motor_state, m.imu_state
            w_low.add(time.monotonic(), tick=int(m.tick), mode_pr=int(m.mode_pr), mode_machine=int(m.mode_machine),
                      imu_quat=np.asarray(imu.quaternion, np.float32), imu_gyro=np.asarray(imu.gyroscope, np.float32),
                      imu_acc=np.asarray(imu.accelerometer, np.float32), imu_rpy=np.asarray(imu.rpy, np.float32),
                      imu_temp=int(imu.temperature),
                      q=np.fromiter((s.q for s in ms), np.float32, N_MOTORS),
                      dq=np.fromiter((s.dq for s in ms), np.float32, N_MOTORS),
                      ddq=np.fromiter((s.ddq for s in ms), np.float32, N_MOTORS),
                      tau=np.fromiter((s.tau_est for s in ms), np.float32, N_MOTORS),
                      temp=np.asarray([s.temperature for s in ms], np.int16),
                      vol=np.fromiter((s.vol for s in ms), np.float32, N_MOTORS),
                      motor_mode=np.fromiter((s.mode for s in ms), np.uint8, N_MOTORS),
                      motorstate=np.fromiter((s.motorstate for s in ms), np.uint32, N_MOTORS),
                      remote=np.frombuffer(bytes(m.wireless_remote), np.uint8))

        def on_cmd(m):
            mc = m.motor_cmd
            w_cmd.add(time.monotonic(), mode_pr=int(m.mode_pr), mode_machine=int(m.mode_machine),
                      q=np.fromiter((c.q for c in mc), np.float32, N_MOTORS),
                      dq=np.fromiter((c.dq for c in mc), np.float32, N_MOTORS),
                      tau=np.fromiter((c.tau for c in mc), np.float32, N_MOTORS),
                      kp=np.fromiter((c.kp for c in mc), np.float32, N_MOTORS),
                      kd=np.fromiter((c.kd for c in mc), np.float32, N_MOTORS),
                      motor_mode=np.fromiter((c.mode for c in mc), np.uint8, N_MOTORS))

        def on_imu2(m):
            self.writer("secondary_imu").add(time.monotonic(), quat=np.asarray(m.quaternion, np.float32),
                                             gyro=np.asarray(m.gyroscope, np.float32),
                                             acc=np.asarray(m.accelerometer, np.float32),
                                             rpy=np.asarray(m.rpy, np.float32))

        def on_odo(m):
            # gear_sonic_sim writes wxyz into `orientation` (despite the IDL comment); stored as published
            self.writer("odostate").add(time.monotonic(), tick=int(m.tick), position=np.asarray(m.position, np.float32),
                                        orientation=np.asarray(m.orientation, np.float32),
                                        linear_velocity=np.asarray(m.linear_velocity, np.float32),
                                        angular_velocity=np.asarray(m.angular_velocity, np.float32))

        sub("rt/lowstate", LowState_, on_low, 50)
        sub("rt/lowcmd", LowCmd_, on_cmd, 50)
        sub("rt/secondary_imu", IMUState_, on_imu2, 50)
        sub("rt/odostate", OdoState_, on_odo, 50)
        if self.args.lidar:
            from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

            from g1deploy.perception.lidar_imu_idl import Imu_
            from g1deploy.perception.unitree_lidar import TOPIC_CLOUD, TOPIC_IMU

            w_lidar = self.writer("lidar", compress=True)

            def on_cloud(m):
                t = time.monotonic()
                if self.lidar_fields is None:  # into meta.json now (main loop): a cut recording stays decodable
                    self.lidar_fields = [{"name": f.name, "offset": int(f.offset), "datatype": int(f.datatype),
                                          "count": int(f.count)} for f in m.fields]
                    self.meta_dirty = True
                w_lidar.add(t, stamp_sec=int(m.header.stamp.sec), stamp_nanosec=int(m.header.stamp.nanosec),
                            width=int(m.width), height=int(m.height), point_step=int(m.point_step),
                            row_step=int(m.row_step), is_bigendian=bool(m.is_bigendian), data=bytes(m.data))

            def on_lidar_imu(m):
                self.writer("lidar_imu").add(time.monotonic(), stamp_sec=int(m.header.stamp.sec),
                                             stamp_nanosec=int(m.header.stamp.nanosec),
                                             orientation=np.array([m.orientation.w, m.orientation.x, m.orientation.y,
                                                                   m.orientation.z], np.float64),
                                             angular_velocity=np.array([m.angular_velocity.x, m.angular_velocity.y,
                                                                        m.angular_velocity.z], np.float64),
                                             linear_acceleration=np.array([m.linear_acceleration.x,
                                                                           m.linear_acceleration.y,
                                                                           m.linear_acceleration.z], np.float64))

            # best effort, own reader: a reliable reader that falls behind on ~0.9 MB frames under full-chain load
            # stalled for a whole session (1 frame recorded in walled rooms); here a missed frame is just a gap
            self.lidar_reader = self.best_effort_reader(TOPIC_CLOUD, PointCloud2_, self._guard(TOPIC_CLOUD, on_cloud))
            sub(TOPIC_IMU, Imu_, on_lidar_imu, 50)

    def best_effort_reader(self, topic_name, typ, cb, depth=8):
        """A DataReader on the SDK's participant with BEST_EFFORT reliability, drained by its own thread."""
        from cyclonedds.core import WaitSet, ReadCondition, SampleState, ViewState, InstanceState
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from cyclonedds.util import duration
        from unitree_sdk2py.core.channel import ChannelFactory

        dp = ChannelFactory()._ChannelFactory__participant
        reader = DataReader(dp, Topic(dp, topic_name, typ),
                            Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(depth)))
        ws = WaitSet(dp)
        ws.attach(ReadCondition(reader, SampleState.Any | ViewState.Any | InstanceState.Any))

        def loop():
            while not self.stop.is_set():
                ws.wait(duration(milliseconds=100))
                for sample in reader.take(N=depth):
                    if type(sample).__name__ != "InvalidSample":
                        cb(sample)

        threading.Thread(target=loop, daemon=True).start()
        return reader

    def _guard(self, name, cb):
        def wrapped(msg):
            try:
                cb(msg)
            except Exception as exc:  # a bad sample is recorded as an error, never kills the recorder
                self.error(name, f"{type(exc).__name__}: {exc}")
        return wrapped

    # -- ZMQ ------------------------------------------------------------------------------------------
    def zmq_loop(self):
        if not self.args.zmq:
            return
        import zmq

        ctx = zmq.Context.instance()
        poller, socks = zmq.Poller(), {}
        for spec in self.args.zmq:
            name, topic, endpoint = spec.split("=", 2)
            s = ctx.socket(zmq.SUB)
            s.setsockopt(zmq.RCVHWM, 1000)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.SUBSCRIBE, topic.encode())
            s.connect(endpoint)
            poller.register(s, zmq.POLLIN)
            socks[s] = self.writer(f"zmq_{name}", compress=name == "cloud")
        while not self.stop.is_set():
            for s, _ in poller.poll(100):
                w = socks[s]
                while True:
                    try:
                        msg = s.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    w.add(time.monotonic(), msg=msg)
        for s in socks:
            s.close(0)

    def status_loop(self):
        if not self.args.status:
            return
        import zmq

        ctx = zmq.Context.instance()
        period = 1.0 / self.args.status_hz
        req = None
        while not self.stop.is_set():
            t_next = time.monotonic() + period
            if req is None:
                req = ctx.socket(zmq.REQ)
                req.setsockopt(zmq.LINGER, 0)
                req.connect(self.args.status)
            try:
                req.send_string('{"cmd": "status"}')
                if req.poll(300):
                    status = json.loads(req.recv())
                    t = time.monotonic()
                    self.status.add(t, {"t": t, "t_unix": time.time(), "status": status})
                else:  # no reply (node not up yet / restarting): lazy pirate, fresh socket
                    req.close(0)
                    req = None
            except (zmq.ZMQError, ValueError):
                if req is not None:
                    req.close(0)
                req = None
            time.sleep(max(0.0, t_next - time.monotonic()))
        if req is not None:
            req.close(0)

    # -- writing --------------------------------------------------------------------------------------
    def flush_all(self, close=False):
        """One writer round: every stream's rows so far -> one chunk each. A failed write loses that chunk's rows
        (counted, reported) and the round goes on with the next stream."""
        self.flush_started = time.monotonic()
        for w in list(self.writers.values()):
            self.check_quota()  # before each chunk: the quota overshoots by at most one chunk per stream
            rows = w.take(close)
            try:
                w.write(rows)
            except Exception as exc:
                w.failed(exc, len(rows.get("t") or []))
                self.error(f"write:{w.name}", f"{type(exc).__name__}: {exc}")
        self.last_flush_s = round(time.monotonic() - self.flush_started, 3)
        self.last_flush_unix, self.flush_started = time.time(), None

    def writer_loop(self):
        """Chunks off the receive threads and off the heartbeat: a slow disk shows as a growing queue and drops."""
        while True:
            stopping = self.stop.wait(self.args.flush_s)
            try:
                self.flush_all(close=stopping)
            except Exception as exc:  # never the end of the writer: the next round tries again
                self.error("writer", f"{type(exc).__name__}: {exc}")
            if stopping:
                return

    def set_paused(self, why):
        if why != self.paused:
            self.paused = why
            for w in list(self.writers.values()) + [self.status]:
                with w.lock:
                    w.paused = why

    def check_quota(self):
        """Past --quota-gb every stream drops (and counts) from now on."""
        if not self.quota or self.paused == "quota":
            return self.paused == "quota"
        total = sum(w.bytes for w in list(self.writers.values())) + self.status.bytes
        if total >= self.quota:
            self.error("quota", f"quota of {self.quota / 1e9:.3g} GB reached ({total / 1e9:.3g} GB): recording "
                                "stopped, samples dropped and counted")
            self.set_paused("quota")
        return self.paused == "quota"

    def check_disk(self):
        """Disk free and the quota, once a second: below --min-free-gb (or past the quota) every stream drops."""
        try:
            du = disk_usage(self.out)
            self.disk = {"free_bytes": du.free, "total_bytes": du.total}
        except OSError as exc:
            self.error("disk", f"cannot read free space: {exc}")
            return
        if self.check_quota():
            pass
        elif self.min_free and du.free < self.min_free:
            if self.paused != "disk low":
                self.error("disk", f"disk free {du.free / 1e9:.2f} GB < {self.min_free / 1e9:.2f} GB: recording paused, "
                                   "samples dropped and counted")
            self.set_paused("disk low")
        elif self.paused == "disk low" and du.free > self.min_free + 0.2e9:
            self.error("disk", f"disk free {du.free / 1e9:.2f} GB again: recording resumed (samples were dropped)")
            self.set_paused(None)

    def check_parent(self):
        """The console that started us is gone (killed, crashed): record the aftermath, then finalize by ourselves."""
        if self.ppid0 in (0, 1) or os.getppid() == self.ppid0:
            return
        now = time.monotonic()
        if self.orphan_t is None:
            self.orphan_t = now
            self.error("parent", f"the console (pid {self.ppid0}) exited without stopping the recorder: recording "
                                 f"{self.args.orphan_grace_s:.0f} s more, then finalizing")
        elif now - self.orphan_t >= self.args.orphan_grace_s:
            self.request_stop(f"the console (pid {self.ppid0}) is gone")

    # -- housekeeping ---------------------------------------------------------------------------------
    def stats(self, finalized=False):
        now, now_unix = time.monotonic(), time.time()
        out = {name: w.stats(now, now_unix) for name, w in sorted(self.writers.items())}
        out["node_status"] = self.status.stats(now, now_unix)
        state = ("finalized" if finalized else "finalizing" if self.stop.is_set() else
                 f"paused: {self.paused}" if self.paused else "orphan" if self.orphan_t is not None else "recording")
        return {"format": STATS_FORMAT, "t": now, "t_unix": now_unix, "elapsed_s": round(now - self.t0, 2),
                "pid": os.getpid(), "state": state, "finalized": finalized, "paused": self.paused,
                "flush_s": self.args.flush_s, "out": str(self.out),
                "writer": {"busy_s": None if self.flush_started is None else round(now - self.flush_started, 2),
                           "last_flush_unix": self.last_flush_unix, "last_flush_s": self.last_flush_s},
                "disk": dict(self.disk, min_free_bytes=self.min_free or None), "quota_bytes": self.quota,
                "bytes_total": sum(v["bytes"] or 0 for v in out.values()),
                "dropped_total": sum(v["dropped"] for v in out.values()),
                "streams": out, "errors": dict(self.errors), "error_log": list(self.error_log)}

    def write_json(self, name, doc):
        tmp = self.out / f".{name}.tmp"
        try:
            tmp.write_text(json.dumps(doc, indent=1, default=float) + "\n")
            os.replace(tmp, self.out / name)
            return True
        except OSError as exc:  # disk full: the heartbeat goes stale, the recording goes on
            if name != "stats.json" or "stats" not in self.errors:
                self.error(name.split(".")[0], f"cannot write {name}: {exc}")
            return False

    def run(self):
        self.t0 = time.monotonic()
        self.meta = meta = {"format": FORMAT, "started_unix": time.time(), "started_monotonic": self.t0,
                            "host": socket.gethostname(), "iface": self.args.iface, "domain": self.args.domain,
                            "lidar": self.args.lidar, "zmq": self.args.zmq, "status": self.args.status,
                            "status_hz": self.args.status_hz, "flush_s": self.args.flush_s, "clock": "time.monotonic()",
                            "pid": os.getpid(), "quota_gb": self.args.quota_gb, "min_free_gb": self.args.min_free_gb}
        self.write_json("meta.json", meta)
        self.check_disk()
        writer = threading.Thread(target=self.writer_loop, daemon=True)
        writer.start()
        threads = []
        last_stats = 0.0
        try:
            self.start_dds()
            threads = [threading.Thread(target=self.zmq_loop, daemon=True),
                       threading.Thread(target=self.status_loop, daemon=True)]
            for th in threads:
                th.start()
            print(json.dumps({"event": "recorder_up", "out": str(self.out), "pid": os.getpid()}), flush=True)
            while not self.stop.is_set():
                time.sleep(0.1)
                now = time.monotonic()
                if now - last_stats >= 1.0:
                    last_stats = now
                    self.check_disk()
                    self.check_parent()
                    if self.meta_dirty:
                        self.meta_dirty = False
                        self.write_json("meta.json", dict(meta, lidar_fields=self.lidar_fields))
                    self.write_json("stats.json", self.stats())
                if self.args.seconds and now - self.t0 >= self.args.seconds:
                    self.request_stop(f"--seconds {self.args.seconds:g} reached")
        finally:
            self.request_stop("interrupted")
            for th in threads:
                th.join(2.0)
            self.status.close()
            writer.join(self.args.final_timeout_s)  # its last round takes every stream's rows, then closes them
            stuck = writer.is_alive()
            if stuck:
                self.error("writer", f"the last chunks were not written within {self.args.final_timeout_s:g} s "
                                     "(disk stalled?)")
            meta.update(stopped_unix=time.time(), stopped_monotonic=time.monotonic(), lidar_fields=self.lidar_fields,
                        stop_reason=self.stop_reason)
            self.write_json("meta.json", meta)
            stats, done = self.stats(finalized=True), False
            if not stuck:  # the marker of a recording closed by the recorder itself: every chunk is on disk before it
                done = self.write_json("finalized.json", {
                    "format": STATS_FORMAT, "t_unix": time.time(), "t": time.monotonic(), "reason": self.stop_reason,
                    "elapsed_s": stats["elapsed_s"], "clean": not self.errors and not stats["dropped_total"],
                    "dropped_total": stats["dropped_total"], "bytes_total": stats["bytes_total"],
                    "streams": {k: {f: v[f] for f in ("samples", "chunks", "bytes", "dropped", "gaps", "first_t", "last_t")}
                                for k, v in stats["streams"].items()},
                    "errors": dict(self.errors)})
            self.write_json("stats.json", stats if done else self.stats(finalized=False))
            print(json.dumps({"event": "recorder_down", "reason": self.stop_reason, "finalized": done,
                              **{k: v["count"] for k, v in stats["streams"].items()}}), flush=True)


DEFAULT_ZMQ = ["pose=pose=tcp://127.0.0.1:5556", "debug=g1_debug=tcp://127.0.0.1:5557",
               "odom=odom=tcp://127.0.0.1:5561", "cloud=cloud=tcp://127.0.0.1:5561"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--no-lidar", dest="lidar", action="store_false", help="skip the raw lidar frames (~2 MB/s)")
    ap.add_argument("--zmq", action="append", default=None, metavar="NAME=TOPIC=ENDPOINT",
                    help=f"ZMQ streams to capture (default: {' '.join(DEFAULT_ZMQ)})")
    ap.add_argument("--status", default="tcp://127.0.0.1:5562", help="node control socket for status ('' = off)")
    ap.add_argument("--status-hz", type=float, default=5.0)
    ap.add_argument("--flush-s", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after S s (0 = until SIGINT/SIGTERM)")
    ap.add_argument("--quota-gb", type=float, default=0.0, help="stop keeping samples past this much data (0 = none)")
    ap.add_argument("--min-free-gb", type=float, default=1.0,
                    help="pause (drop and count samples) while the disk has less than this free")
    ap.add_argument("--pending-scale", type=float, default=1.0, help="scale of the per-stream writer queue bounds")
    ap.add_argument("--orphan-grace-s", type=float, default=60.0,
                    help="when the console that started the recorder disappears: record this long, then finalize")
    ap.add_argument("--final-timeout-s", type=float, default=6.0, help="wait this long for the last chunks at exit")
    args = ap.parse_args(argv)
    args.zmq = DEFAULT_ZMQ if args.zmq is None else [z for z in args.zmq if z]
    return args


def main(argv=None):
    rec = Recorder(parse_args(argv))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda n, _f: rec.request_stop(f"stopped ({signal.Signals(n).name})"))
    rec.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
