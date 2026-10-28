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
`stats.json` (rates, gaps, bytes) is rewritten every second; `meta.json` describes the recording.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMAT = "g1deploy.recording.v1"
N_MOTORS = 35


class ChunkWriter:
    """Rows of named fields -> <root>/<name>/NNNNNN.npz every flush_s, written atomically off the receive thread.

    A field is a scalar, a fixed-shape array, or `bytes` (variable length: stored concatenated as <key> with
    <key>_off offsets of length rows + 1)."""

    def __init__(self, root: Path, name: str, compress=False):
        self.dir = Path(root) / name
        self.name, self.compress = name, compress
        self.lock = threading.Lock()
        self.rows: dict[str, list] = {}
        self.seq = 0
        self.count, self.bytes, self.first, self.last, self.max_gap = 0, 0, None, None, 0.0

    def add(self, t, **fields):
        with self.lock:
            if self.last is not None:
                self.max_gap = max(self.max_gap, t - self.last)
            self.first = t if self.first is None else self.first
            self.last = t
            self.count += 1
            self.rows.setdefault("t", []).append(t)
            self.rows.setdefault("t_unix", []).append(time.time())
            for k, v in fields.items():
                self.rows.setdefault(k, []).append(v)

    def take(self):
        with self.lock:
            rows, self.rows = self.rows, {}
        return rows

    def write(self, rows):
        if not rows or not rows.get("t"):
            return
        out = {}
        for k, vals in rows.items():
            if isinstance(vals[0], (bytes, bytearray, memoryview)):
                lens = np.fromiter((len(v) for v in vals), dtype=np.int64, count=len(vals))
                out[k + "_off"] = np.r_[0, np.cumsum(lens)]
                out[k] = np.frombuffer(b"".join(bytes(v) for v in vals), dtype=np.uint8)
            else:
                out[k] = np.asarray(vals)
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self.seq:06d}.npz"
        tmp = path.with_suffix(".tmp.npz")
        (np.savez_compressed if self.compress else np.savez)(tmp, **out)
        os.replace(tmp, path)
        self.bytes += path.stat().st_size
        self.seq += 1

    def stats(self, now):
        span = (self.last - self.first) if self.count > 1 else 0.0
        return {"count": self.count, "rate_hz": round((self.count - 1) / span, 2) if span > 0 else None,
                "age_s": round(now - self.last, 3) if self.last is not None else None,
                "max_gap_s": round(self.max_gap, 4), "chunks": self.seq, "bytes": self.bytes}


class Recorder:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.writers: dict[str, ChunkWriter] = {}
        self.status_file = None
        self.status_count, self.status_last, self.status_ok = 0, None, 0
        self.errors: dict[str, str] = {}
        self.lidar_fields = None

    def writer(self, name, compress=False):
        if name not in self.writers:
            self.writers[name] = ChunkWriter(self.out, name, compress)
        return self.writers[name]

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
                if self.lidar_fields is None:
                    self.lidar_fields = [{"name": f.name, "offset": int(f.offset), "datatype": int(f.datatype),
                                          "count": int(f.count)} for f in m.fields]
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
                self.errors[name] = f"{type(exc).__name__}: {exc}"
        return wrapped

    # -- ZMQ ------------------------------------------------------------------------------------------
    def zmq_loop(self):
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
        self.status_file = open(self.out / "node_status.jsonl", "a", buffering=1)
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
                    self.status_file.write(json.dumps({"t": t, "t_unix": time.time(), "status": status}) + "\n")
                    self.status_count, self.status_last = self.status_count + 1, t
                else:  # no reply (node not up yet / restarting): lazy pirate, fresh socket
                    req.close(0)
                    req = None
            except zmq.ZMQError:
                req.close(0)
                req = None
            time.sleep(max(0.0, t_next - time.monotonic()))
        if req is not None:
            req.close(0)
        self.status_file.close()

    # -- housekeeping ---------------------------------------------------------------------------------
    def stats(self):
        now = time.monotonic()
        out = {name: w.stats(now) for name, w in sorted(self.writers.items())}
        out["node_status"] = {"count": self.status_count,
                              "age_s": round(now - self.status_last, 3) if self.status_last else None}
        return {"t": now, "elapsed_s": round(now - self.t0, 2), "streams": out, "errors": self.errors}

    def flush_all(self):
        for w in list(self.writers.values()):
            try:
                w.write(w.take())
            except Exception as exc:
                self.errors[f"write:{w.name}"] = f"{type(exc).__name__}: {exc}"

    def write_json(self, name, doc):
        tmp = self.out / f".{name}.tmp"
        tmp.write_text(json.dumps(doc, indent=1, default=float) + "\n")
        os.replace(tmp, self.out / name)

    def run(self):
        self.t0 = time.monotonic()
        meta = {"format": FORMAT, "started_unix": time.time(), "started_monotonic": self.t0,
                "host": socket.gethostname(), "iface": self.args.iface, "domain": self.args.domain,
                "lidar": self.args.lidar, "zmq": self.args.zmq, "status": self.args.status,
                "status_hz": self.args.status_hz, "flush_s": self.args.flush_s, "clock": "time.monotonic()"}
        self.write_json("meta.json", meta)
        self.start_dds()
        threads = [threading.Thread(target=self.zmq_loop, daemon=True),
                   threading.Thread(target=self.status_loop, daemon=True)]
        for th in threads:
            th.start()
        print(json.dumps({"event": "recorder_up", "out": str(self.out)}), flush=True)
        last_flush = last_stats = time.monotonic()
        try:
            while not self.stop.is_set():
                time.sleep(0.1)
                now = time.monotonic()
                if now - last_flush >= self.args.flush_s:
                    self.flush_all()
                    last_flush = now
                if now - last_stats >= 1.0:
                    self.write_json("stats.json", self.stats())
                    last_stats = now
                if self.args.seconds and now - self.t0 >= self.args.seconds:
                    break
        finally:
            self.stop.set()
            for th in threads:
                th.join(2.0)
            self.flush_all()
            meta.update(stopped_unix=time.time(), stopped_monotonic=time.monotonic(), lidar_fields=self.lidar_fields)
            self.write_json("meta.json", meta)
            self.write_json("stats.json", self.stats())
            print(json.dumps({"event": "recorder_down", **{k: v["count"] for k, v in self.stats()["streams"].items()}}),
                  flush=True)


DEFAULT_ZMQ = ["pose=pose=tcp://127.0.0.1:5556", "debug=g1_debug=tcp://127.0.0.1:5557",
               "odom=odom=tcp://127.0.0.1:5561", "cloud=cloud=tcp://127.0.0.1:5561"]


def main(argv=None):
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
    args = ap.parse_args(argv)
    args.zmq = args.zmq or DEFAULT_ZMQ
    rec = Recorder(args)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: rec.stop.set())
    rec.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
