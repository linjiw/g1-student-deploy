#!/usr/bin/env python3
"""ROS 2 bridge: LIO odometry on the head MID-360 -> the node's `odom` ZMQ topic (and optionally `cloud`).

Recommended LIO (docs/G1_PERCEPTION.md): Point-LIO on the robot's own DDS lidar topics, seen by ROS 2 as
/utlidar/cloud_livox_mid360 + /utlidar/imu_livox_mid360 (lidar_type 2, timestamp_unit 3, acc_norm 1.0,
extrinsic_T [-0.011, -0.02329, 0.04412], extrinsic_est_en false); its high-rate odometry is
/aft_mapped_to_init (the default here). With FAST-LIO2 use --odom-topic /Odometry (10 Hz; our 50 Hz pose
filter smooths it). The node reads the lidar itself (`--cloud dds`), so cloud forwarding is off by default
(`--forward-cloud` re-enables it for setups that only have the registered cloud).

UNTESTED: there is no ROS 2 on the development machine. The frame math it calls
(`g1deploy/adapters/lio.py`) is unit-tested (tests/test_lio.py); this file only moves messages.

Inputs   /Odometry          nav_msgs/Odometry      FAST-LIO body (sensor IMU) pose in camera_init
         /cloud_registered  sensor_msgs/PointCloud2 deskewed points in camera_init
         rt/lowstate        DDS (unitree_sdk2py)    joints (waist) + pelvis IMU
Outputs  ZMQ PUB --bind (default tcp://*:5561):
         `odom`  pos [3] f64, quat_wxyz [4] f64, lin_w [3] f64, ang_w [3] f64, stamp [1] f64   at 50 Hz
         `cloud` points [N,3] f32 (world, floor z = 0), origin [3] f64, stamp [1] f64            per scan
Run the node with `--odom zmq --odom-connect tcp://<bridge>:5561 --cloud tcp://<bridge>:5561`.

Start with the robot standing still (in the gantry, feet on the floor): the first LIO pose defines
the world (gravity from the pelvis IMU, floor from FK, yaw 0 = start heading).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from g1deploy.adapters.lio import LioFrames, PoseFilter, hom, quat_to_rot  # noqa: E402
from g1deploy.zmq_stream import pack  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", required=True, help="DDS interface on the robot network (e.g. eth0)")
    ap.add_argument("--bind", default="tcp://*:5561")
    ap.add_argument("--odom-topic", default="/aft_mapped_to_init", help="Point-LIO; FAST-LIO2: /Odometry")
    ap.add_argument("--cloud-topic", default="/cloud_registered")
    ap.add_argument("--forward-cloud", action="store_true", help="also forward the LIO registered cloud as `cloud`")
    args = ap.parse_args(argv)

    import rclpy
    import zmq
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2

    from g1deploy.node import DdsState

    state = DdsState(args.iface, odom="none")
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.bind(args.bind)
    frames, filt = LioFrames(), PoseFilter()
    last = {"t_caminit_body": None}

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
        last["t_caminit_body"] = t_cb
        pos, quat = frames.pelvis_pose(t_cb, q_mj)
        filt.fix(time.monotonic(), pos, quat, imu)

    def on_cloud(msg):
        if frames.t_world_caminit is None or last["t_caminit_body"] is None:
            return
        pts = np.array([[x, y, z] for x, y, z in point_cloud2.read_points(msg, ("x", "y", "z"), skip_nans=True)])
        if not len(pts):
            return
        sock.send(pack([("points", frames.points_world(pts).astype(np.float32)),
                        ("origin", frames.lidar_origin_world(last["t_caminit_body"])),
                        ("stamp", np.array([time.monotonic()]))], topic=b"cloud"))

    def tick():
        lo = low()
        if lo is None:
            return
        out = filt.pose(time.monotonic(), lo[2])
        if out is None:
            return
        pos, quat, vel = out
        sock.send(pack([("pos", pos), ("quat_wxyz", quat), ("lin_w", vel), ("ang_w", np.zeros(3)),
                        ("stamp", np.array([time.monotonic()]))], topic=b"odom"))

    rclpy.init()
    node = rclpy.create_node("g1_student_lio_bridge")
    node.create_subscription(Odometry, args.odom_topic, on_odom, 10)
    if args.forward_cloud:
        node.create_subscription(PointCloud2, args.cloud_topic, on_cloud, 2)
    node.create_timer(0.02, tick)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
