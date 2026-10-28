#!/usr/bin/env python3
"""LIO odometry (FAST-LIO2 / Point-LIO running under ROS 2) -> the node's `odom` (and `cloud`) ZMQ topics, read
straight from DDS: no rclpy in this process.

Same frame math and 50 Hz pose filter as adapters/ros2_lio_bridge.py (LioFrames + PoseFilter, tested), for machines
where the ROS python cannot import the deploy stack (ros2_lio_bridge needs rclpy *and* unitree_sdk2py/zmq/numpy in
one interpreter). A ROS 2 topic /X is the DDS topic rt/X, so with rmw_cyclonedds on the robot NIC:

  in   rt/Odometry          nav_msgs Odometry_ (FAST-LIO2 /Odometry, or Point-LIO /aft_mapped_to_init via --odom-topic)
       rt/cloud_registered  sensor_msgs PointCloud2_ (with --forward-cloud)
       rt/lowstate          joints + pelvis IMU (gravity, roll/pitch, the 50 Hz filter)
  out  ZMQ PUB --bind (default tcp://*:5561): `odom` pos/quat_wxyz/lin_w/ang_w/stamp at 50 Hz; `cloud` per scan

Start it with the robot standing still: the first LIO pose defines the world (gravity from the pelvis IMU, floor from
FK, yaw 0 = start heading). Node: `--odom zmq` (console: `g1 robot --odom zmq`).

usage: dds_lio_bridge.py --iface eno1 [--odom-topic rt/Odometry] [--forward-cloud]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from g1deploy.adapters.lio import LioFrames, PoseFilter, fix_time, hom, quat_to_rot  # noqa: E402
from g1deploy.zmq_stream import pack  # noqa: E402


def cloud_xyz(msg):
    """PointCloud2_ -> (N, 3) float64 x y z, fields read by name."""
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step, n = int(msg.point_step), int(msg.width) * int(msg.height)
    buf = raw[: step * n].reshape(n, step)
    off = {f.name: int(f.offset) for f in msg.fields}
    cols = [buf[:, off[k]:off[k] + 4].copy().view("<f4").reshape(-1) for k in ("x", "y", "z")]
    pts = np.stack(cols, 1).astype(np.float64)
    return pts[np.isfinite(pts).all(1)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", required=True)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--bind", default="tcp://*:5561")
    ap.add_argument("--odom-topic", default="rt/Odometry", help="FAST-LIO2: rt/Odometry; Point-LIO: rt/aft_mapped_to_init")
    ap.add_argument("--cloud-topic", default="rt/cloud_registered")
    ap.add_argument("--forward-cloud", action="store_true")
    ap.add_argument("--stats-s", type=float, default=5.0)
    args = ap.parse_args(argv)
    import zmq
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

    from g1deploy.node import DdsState

    state = DdsState(args.iface, args.domain, odom="none")
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.bind(args.bind)
    send_lock = threading.Lock()
    frames, filt = LioFrames(), PoseFilter()
    last = {"t_cb": None, "odom": 0, "cloud": 0, "pub": 0, "lio_xyz": None}

    def low():
        with state.lock:
            return state.low

    def on_odom(msg):
        lo = low()
        if lo is None:
            return
        q_mj, _, imu, _ = lo
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        t_cb = hom(quat_to_rot([o.w, o.x, o.y, o.z]), np.array([p.x, p.y, p.z]))
        if frames.t_world_caminit is None:
            frames.initialise(imu, q_mj, t_cb)
            print("[dds_lio] world initialised from the first LIO pose (robot must stand still now)", flush=True)
        last["t_cb"], last["lio_xyz"] = t_cb, (p.x, p.y, p.z)
        last["odom"] += 1
        pos, quat = frames.pelvis_pose(t_cb, q_mj)
        t_fix, age = fix_time(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        if age is not None:
            last["age_sum"] = last.get("age_sum", 0.0) + age
            last["age_n"] = last.get("age_n", 0) + 1
        filt.fix(t_fix, pos, quat, imu)

    def on_cloud(msg):
        if frames.t_world_caminit is None or last["t_cb"] is None:
            return
        pts = cloud_xyz(msg)
        if not len(pts):
            return
        with send_lock:
            sock.send(pack([("points", frames.points_world(pts).astype(np.float32)),
                            ("origin", frames.lidar_origin_world(last["t_cb"])),
                            ("stamp", np.array([time.monotonic()]))], topic=b"cloud"))
        last["cloud"] += 1

    subs = [ChannelSubscriber(args.odom_topic, Odometry_)]
    subs[0].Init(on_odom, 10)
    if args.forward_cloud:
        subs.append(ChannelSubscriber(args.cloud_topic, PointCloud2_))
        subs[1].Init(on_cloud, 2)
    print(f"[dds_lio] {args.odom_topic}" + (f" + {args.cloud_topic}" if args.forward_cloud else "")
          + f" -> {args.bind} (odom 50 Hz" + (", cloud)" if args.forward_cloud else ")"), flush=True)
    t_last, t_stat = time.monotonic(), time.monotonic()
    try:
        while True:
            t_next = t_last + 0.02
            if (rest := t_next - time.monotonic()) > 0:
                time.sleep(rest)
            t_last = time.monotonic()
            lo = low()
            if lo is None:
                continue
            out = filt.pose(time.monotonic(), lo[2])
            if out is None:
                continue
            pos, quat, vel = out
            with send_lock:
                sock.send(pack([("pos", pos), ("quat_wxyz", quat), ("lin_w", vel), ("ang_w", np.zeros(3)),
                                ("stamp", np.array([time.monotonic()]))], topic=b"odom"))
            last["pub"] += 1
            if args.stats_s and t_last - t_stat >= args.stats_s:
                age = last.get("age_sum", 0.0) / max(last.get("age_n", 0), 1) if last.get("age_n") else None
                print(f"[dds_lio] LIO odom in {last['odom']}, clouds {last['cloud']}, odom out {last['pub']}, "
                      f"pelvis {np.round(pos, 3).tolist()}, LIO pose age "
                      + (f"{age * 1000:.0f} ms" if age is not None else "unknown (clocks apart)"), flush=True)
                t_stat = t_last
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
