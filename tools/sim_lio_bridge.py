#!/usr/bin/env python3
"""Simulation stand-in for Point-LIO + g1deploy/adapters/ros2_lio_bridge.py: the same ZMQ `odom` and `cloud` topics
on :5561, from the MuJoCo G1 (`g1 sim --odom lio`).

  odom   pos [3] f64, quat_wxyz [4] f64, lin_w [3] f64, ang_w [3] f64, stamp [1] f64 (time.monotonic())  at 50 Hz,
         from rt/odostate (simulator truth) plus an optional random-walk drift (--noise '{"xy_rw_m": 0.02,
         "yaw_rw_rad": 0.01}', per sqrt(s)), the way a LIO estimate drifts, and an optional odometry error
         (--noise '{"scale": 0.9, "rotate_deg": -4}': each step's xy displacement scaled and turned, the way leg
         odometry under-counts; the node registers the raw lidar frames with this pose, so its map smears as on the
         robot - an odometry error on top of true odometry, for tolerance sweeps)
  cloud  points [N,3] f32 in the odometry frame (floor z = 0), origin [3] f64, stamp [1] f64, per lidar frame: the
         robot's raw rt/utlidar frames deskewed with the true pose and expressed in the drifted frame, so map and
         pose agree as they do in a real LIO

With it the node runs `--odom zmq` exactly as on the robot with LIO, and Studio's robot link (odom + cloud + pose +
control) can be rehearsed against the full simulated chain. Subscribe-only on DDS.

usage: sim_lio_bridge.py [--iface lo] [--bind tcp://*:5561] [--noise JSON]
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

from g1deploy.node import DdsCloudInput, DdsState  # noqa: E402
from g1deploy.zmq_stream import pack  # noqa: E402


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--bind", default="tcp://*:5561")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    noise = json.loads(args.noise) if args.noise else {}
    rng = np.random.default_rng(args.seed)
    import zmq

    from g1deploy.perception.cloud_memory import BodyFilter
    from g1deploy.perception.unitree_lidar import CloudFromSensorFrame, Mount

    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 8)
    sock.bind(args.bind)
    state, cloud = DdsState(args.iface, 0, "dds"), DdsCloudInput()
    conv = CloudFromSensorFrame(BodyFilter(), Mount.nominal())
    drift_xy, drift_yaw = np.zeros(2), 0.0
    a = math.radians(float(noise.get("rotate_deg", 0.0)))
    err = float(noise.get("scale", 1.0)) * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    stepped = "scale" in noise or "rotate_deg" in noise
    true_prev, odo_xy = None, None
    period, last = 1.0 / args.hz, time.monotonic()
    n_odom = n_cloud = 0
    print(f"[sim_lio] odom {args.hz:.0f} Hz + registered cloud on {args.bind}, noise {noise or 'none'}", flush=True)

    def to_odom(p):
        c, s = math.cos(drift_yaw), math.sin(drift_yaw)
        out = p.copy()
        out[..., 0] = c * p[..., 0] - s * p[..., 1] + drift_xy[0]
        out[..., 1] = s * p[..., 0] + c * p[..., 1] + drift_xy[1]
        return out

    try:
        while True:
            t_next = last + period
            if (rest := t_next - time.monotonic()) > 0:
                time.sleep(rest)
            now = time.monotonic()
            dt, last = now - last, now
            if noise:
                drift_xy += rng.normal(0.0, noise.get("xy_rw_m", 0.0) * math.sqrt(dt), 2)
                drift_yaw += float(rng.normal(0.0, noise.get("yaw_rw_rad", 0.0) * math.sqrt(dt)))
            snap = state.snapshot()
            if snap is None:
                continue
            q = np.array(snap["qpos"], float)
            if stepped:  # integrate the erroneous displacement
                odo_xy = q[:2].copy() if odo_xy is None else odo_xy + err @ (q[:2] - true_prev)
                true_prev = q[:2].copy()
                q[:2] = odo_xy
            c, s = math.cos(drift_yaw), math.sin(drift_yaw)
            lin = snap["lin_w"].copy()
            lin[:2] = [c * lin[0] - s * lin[1], s * lin[0] + c * lin[1]]
            sock.send(pack([("pos", to_odom(np.asarray(q[:3], float))), ("quat_wxyz", quat_mul(yaw_quat(drift_yaw), q[3:7])),
                            ("lin_w", lin), ("ang_w", np.asarray(snap["ang_w"], float)), ("stamp", np.array([now]))],
                           topic=b"odom"))
            n_odom += 1
            for f in cloud.poll():
                pts, org = conv(f["points"], f["t_s"], f["wall"], state.qpos_at)
                if not len(pts):
                    continue
                sock.send(pack([("points", to_odom(pts).astype(np.float32)), ("origin", to_odom(org[-1])),
                                ("stamp", np.array([time.monotonic()]))], topic=b"cloud"))
                n_cloud += 1
            if n_odom % int(10 * args.hz) == 0:
                print(f"[sim_lio] odom {n_odom}, clouds {n_cloud}, drift xy {drift_xy.round(3).tolist()} "
                      f"yaw {drift_yaw:+.3f}", flush=True)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
