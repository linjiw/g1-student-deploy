#!/usr/bin/env python3
"""FAST-LIO on recorded G1 lidar: replay a session's lidar + lidar IMU into FAST-LIO and score its odometry, or (with
--probe-only) score the live odometry of a robot standing still. Behind `g1 lio replay` and `g1 lio check`.

Runs in the ROS 2 environment (system python + rclpy + the livox_ros_driver2 messages). `g1 lio replay` puts it and
FAST-LIO on a private DDS domain on loopback, so nothing from a replay reaches the robot NIC.

  in   SESSION/rec/lidar       rt/utlidar/cloud_livox_mid360 frames as recorded (G1 raw layout, 10 Hz, lidar clock)
       SESSION/rec/lidar_imu   rt/utlidar/imu_livox_mid360 (sensor_msgs/Imu, accel in g, lidar clock)
  out  /livox/lidar_10hz       CustomMsg, one message per frame (what livox_to_utlidar.py --custom-out publishes)
       /livox/imu              sensor_msgs/Imu
  read /Odometry               FAST-LIO's pose -> report (JSON): rate, drift from the first pose, largest step

--loops N plays the recording N times back to back, stamps shifted by its span plus one frame: a robot that stood
still stays still, so a short recording gives a longer stationary test (each seam is a one-frame gap).

usage: lio_replay.py SESSION [--loops 6] [--rate 1.0] --out report.json
       lio_replay.py --probe-only --seconds 30 --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy.adapters.livox_to_utlidar import CUSTOM_DT, POINT_DT, serialize_custom_msg  # noqa: E402

# a standing robot: the LIO pose must stay inside these (HARDWARE.md §3 asks < 5 cm, < 1 deg after a 5 m loop)
STILL_DRIFT_M = 0.03
STILL_YAW_DEG = 0.5
STEP_M = 0.05  # one odometry step larger than this on a standing robot is a jump (divergence, a bad scan match)


def _chunks(rec, name):
    return sorted(p for p in (rec / name).glob("[0-9]*.npz") if ".tmp" not in p.name)


def load_frames(rec):
    """[(stamp_ns, serialized CustomMsg)] from the recorded rt/utlidar frames (header.stamp = frame start)."""
    out = []
    for path in _chunks(rec, "lidar"):
        with np.load(path) as z:
            off = z["data_off"]
            for j in range(len(z["t"])):
                if int(z["point_step"][j]) != POINT_DT.itemsize:
                    raise ValueError(f"{path.name}: point_step {int(z['point_step'][j])}, expected the adapter's 22")
                frame = np.frombuffer(z["data"][off[j]:off[j + 1]].tobytes(), POINT_DT)
                xyz = np.stack([frame["x"], frame["y"], frame["z"]], 1)
                frame = frame[np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)]
                t0 = int(z["stamp_sec"][j]) * 1_000_000_000 + int(z["stamp_nanosec"][j])
                pts = np.zeros(len(frame), CUSTOM_DT)
                for k in ("x", "y", "z"):
                    pts[k] = frame[k]
                pts["reflectivity"] = np.clip(frame["intensity"], 0, 255).astype(np.uint8)
                pts["line"] = frame["ring"].astype(np.uint8)
                out.append((t0, t0 + frame["time"].astype(np.int64), pts))
    return out


def load_imu(rec):
    """(stamp_ns (N,), angular_velocity (N, 3), linear_acceleration (N, 3)) from the recorded lidar IMU."""
    parts = []
    for path in _chunks(rec, "lidar_imu"):
        with np.load(path) as z:
            parts.append((z["stamp_sec"] * 1_000_000_000 + z["stamp_nanosec"], z["angular_velocity"],
                          z["linear_acceleration"]))
    if not parts:
        return np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def yaw_of(q):
    """q = (w, x, y, z) -> yaw (rad)."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def score(poses, seconds=None):
    """poses: [(t_wall, stamp_s, pos (3,), quat wxyz (4,))] -> drift numbers and a standing-still verdict."""
    if len(poses) < 2:
        return {"odom": len(poses), "verdict": "NO-ODOM", "reason": "FAST-LIO published fewer than 2 poses"}
    pos = np.array([p[2] for p in poses])
    stamp = np.array([p[1] for p in poses])
    yaw = np.unwrap([yaw_of(p[3]) for p in poses])
    dist = np.linalg.norm(pos - pos[0], axis=1)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    span = float(stamp[-1] - stamp[0])
    out = {
        "odom": len(poses),
        "span_s": round(span, 2),
        "rate_hz": round((len(poses) - 1) / span, 2) if span > 0 else None,
        "drift_max_m": round(float(dist.max()), 4),
        "drift_end_m": round(float(dist[-1]), 4),
        "drift_end_xyz_m": np.round(pos[-1] - pos[0], 4).tolist(),
        "yaw_drift_max_deg": round(float(np.degrees(np.abs(yaw - yaw[0]).max())), 3),
        "step_max_m": round(float(steps.max()), 4),
        "stamp_gaps_over_0p25s": int((np.diff(stamp) > 0.25).sum()),
    }
    bad = []
    if out["drift_max_m"] > STILL_DRIFT_M:
        bad.append(f"drift {out['drift_max_m']:.3f} m > {STILL_DRIFT_M}")
    if out["yaw_drift_max_deg"] > STILL_YAW_DEG:
        bad.append(f"yaw {out['yaw_drift_max_deg']:.2f} deg > {STILL_YAW_DEG}")
    if out["step_max_m"] > STEP_M:
        bad.append(f"jump {out['step_max_m']:.3f} m > {STEP_M}")
    if out["rate_hz"] is not None and out["rate_hz"] < 8:
        bad.append(f"rate {out['rate_hz']} Hz < 8")
    out["verdict"] = "FAIL" if bad else "PASS"
    out["reason"] = "; ".join(bad) or f"standing still within {STILL_DRIFT_M * 100:.0f} cm / {STILL_YAW_DEG} deg"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", nargs="?", help="a recorded session (dir with rec/, or the rec/ dir itself)")
    ap.add_argument("--out", required=True, help="report JSON (poses go next to it as .npz)")
    ap.add_argument("--loops", type=int, default=6)
    ap.add_argument("--rate", type=float, default=1.0, help="playback speed (1 = recorded time)")
    ap.add_argument("--probe-only", action="store_true", help="no replay: score live /Odometry for --seconds")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--lidar-topic", default="/livox/lidar_10hz")
    ap.add_argument("--imu-topic", default="/livox/imu")
    ap.add_argument("--odom-topic", default="/Odometry")
    ap.add_argument("--wait-s", type=float, default=30.0, help="wait this long for FAST-LIO to subscribe")
    args = ap.parse_args(argv)

    import rclpy
    from builtin_interfaces.msg import Time
    from livox_ros_driver2.msg import CustomMsg
    from nav_msgs.msg import Odometry
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu

    rclpy.init()
    node = rclpy.create_node("g1_lio_replay")
    poses = []

    def on_odom(m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        poses.append((time.monotonic(), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                      np.array([p.x, p.y, p.z]), np.array([o.w, o.x, o.y, o.z])))

    node.create_subscription(Odometry, args.odom_topic, on_odom, 50)
    report = {"mode": "probe" if args.probe_only else "replay", "odom_topic": args.odom_topic}

    if args.probe_only:
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
    else:
        if not args.session:
            raise SystemExit("give a SESSION (or --probe-only)")
        rec = Path(args.session)
        rec = rec / "rec" if (rec / "rec").is_dir() else rec
        frames, (imu_t, gyr, acc) = load_frames(rec), load_imu(rec)
        if not frames or not len(imu_t):
            raise SystemExit(f"{rec}: needs recorded lidar and lidar_imu streams (lidar {len(frames)}, imu {len(imu_t)})")
        t_first = min(frames[0][0], int(imu_t[0]))
        span = max(int(frames[-1][1][-1]) if len(frames[-1][1]) else frames[-1][0], int(imu_t[-1])) - t_first
        shift = span + 100_000_000
        report.update(session=str(rec), frames=len(frames), imu=len(imu_t), loops=args.loops,
                      recorded_s=round(span * 1e-9, 2), points_per_frame=int(np.mean([len(f[2]) for f in frames])))
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        pub_lidar = node.create_publisher(CustomMsg, args.lidar_topic, qos)
        pub_imu = node.create_publisher(Imu, args.imu_topic, QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                                                                        history=HistoryPolicy.KEEP_LAST))
        t_wait = time.monotonic() + args.wait_s
        while (pub_lidar.get_subscription_count() == 0 or pub_imu.get_subscription_count() == 0) and time.monotonic() < t_wait:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub_lidar.get_subscription_count() == 0:
            raise SystemExit(f"nobody subscribes to {args.lidar_topic} after {args.wait_s:.0f} s: is FAST-LIO running "
                             "on this domain?")
        # one timeline: (stamp_ns, kind, index) for every loop, played in stamp order at --rate
        events = []
        for k in range(args.loops):
            events += [(f[0] + k * shift, 0, i) for i, f in enumerate(frames)]
            events += [(int(t) + k * shift, 1, i) for i, t in enumerate(imu_t)]
        events.sort()
        wall0, stamp0 = time.monotonic() + 0.2, events[0][0]
        late = 0
        for stamp, kind, i in events:
            due = wall0 + (stamp - stamp0) * 1e-9 / args.rate
            while (rest := due - time.monotonic()) > 0:
                rclpy.spin_once(node, timeout_sec=min(rest, 0.01))
            late += time.monotonic() - due > 0.02
            if kind == 0:
                t0, t_ns, pts = frames[i]
                k = (stamp - t0) // shift if shift else 0
                pub_lidar.publish(serialize_custom_msg(stamp, t_ns + k * shift, pts))
            else:
                m = Imu()
                m.header.stamp = Time(sec=int(stamp // 1_000_000_000), nanosec=int(stamp % 1_000_000_000))
                m.header.frame_id = "livox_frame"
                m.orientation.w = 1.0
                m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (float(v) for v in gyr[i])
                m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (float(v) for v in acc[i])
                pub_imu.publish(m)
        t_end = time.monotonic() + 1.5  # the last scans are still in FAST-LIO's queue
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
        report["late_publishes"] = int(late)
    node.destroy_node()
    rclpy.try_shutdown()

    report.update(score(poses))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    if poses:
        np.savez_compressed(out.with_suffix(".npz"), t=np.array([p[0] for p in poses]),
                            stamp=np.array([p[1] for p in poses]), pos=np.array([p[2] for p in poses]),
                            quat_wxyz=np.array([p[3] for p in poses]))
    if "drift_max_m" in report:
        print(f"{report['verdict']}: {report['reason']} | {report['odom']} poses at {report['rate_hz']} Hz over "
              f"{report['span_s']} s, drift max {report['drift_max_m'] * 100:.2f} cm (end {report['drift_end_m'] * 100:.2f}), "
              f"yaw {report['yaw_drift_max_deg']:.2f} deg, largest step {report['step_max_m'] * 100:.2f} cm -> {out}")
    else:
        print(f"{report['verdict']}: {report['reason']} -> {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
