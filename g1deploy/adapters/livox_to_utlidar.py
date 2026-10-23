#!/usr/bin/env python3
"""livox_ros_driver2 (/livox/lidar CustomMsg) -> rt/utlidar/cloud_livox_mid360 in the G1 raw layout, for G1s without
Unitree's lidar service.

Some G1 firmware has no `lidar_driver` service: nothing publishes rt/utlidar/*, and the MID-360 (192.168.123.120) is
read on a host with Livox-SDK2 / livox_ros_driver2 instead (Unitree's G1 lidar instructions; GalaxyGeneral's
Click-and-Traverse runs it this way, host_net_info = the host's IP). This adapter turns that driver's output into the
topic the node already consumes, so `node.py --cloud dds` (deskew per 5 ms slice with the node's own odometry, mesh
self filter, evidence memory) runs unchanged:

  in   /livox/lidar  livox_ros_driver2/CustomMsg (xfer_format 1), any publish rate: one message per UDP packet
       (96 points) or per frame; points in the MID-360's own frame, offset_time = ns from the message timebase
       /livox/imu    sensor_msgs/Imu (accel in g)
  out  /utlidar/cloud_livox_mid360  sensor_msgs/PointCloud2 = DDS rt/utlidar/cloud_livox_mid360, `--frame-s` frames
       (default 0.1 s = 10 Hz) cut on lidar time; fields x y z intensity f32 @0/4/8/12, ring u16 @16,
       time f32 @18 = ns from frame start (point_step 22, g1deploy/perception/unitree_lidar.py LAYOUT);
       header.stamp = frame start on the lidar clock (the node ignores it and times points on arrival)
       /utlidar/imu_livox_mid360   sensor_msgs/Imu, relayed unchanged
  filter  Livox tag: keep a point only when the low four tag bits are 0 (Unitree's G1 lidar instructions: noise /
          multi-echo marks), drop (0,0,0) and points closer than --min-range-m

Runs in the ROS 2 environment (system python + rclpy + the livox_ros_driver2 messages), with rmw_cyclonedds on the
robot NIC so the DDS topic is the one the node subscribes to:

    source /opt/ros/humble/setup.bash; source <lidar_ws>/install/setup.bash
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=<cyclonedds.xml with NetworkInterface eno1> \
      python3 g1deploy/adapters/livox_to_utlidar.py [--frame-s 0.1] [--stats-s 5]

It only subscribes to the driver and publishes the two utlidar topics; it never talks to the robot's controller.
"""

from __future__ import annotations

import argparse
import array
import struct
import sys
import time

import numpy as np

POINT_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("ring", "<u2"), ("time", "<f4")])
assert POINT_DT.itemsize == 22
# CustomPoint in CDR: uint32 offset_time, float32 x y z, uint8 reflectivity tag line -> 19 bytes, next element's
# uint32 aligned to 4 -> 20-byte stride
CUSTOM_DT = np.dtype({"names": ["offset_time", "x", "y", "z", "reflectivity", "tag", "line"],
                      "formats": ["<u4", "<f4", "<f4", "<f4", "u1", "u1", "u1"],
                      "offsets": [0, 4, 8, 12, 16, 17, 18], "itemsize": 20})


def _align(off, n):
    return off + (-off) % n


def parse_custom_msg(raw: bytes):
    """Serialized CustomMsg (CDR, little endian) -> (timebase_ns, points structured array). Offsets are relative to the
    payload after the 4-byte encapsulation header, as CDR alignment is."""
    if raw[1] != 1:
        raise ValueError("expected little-endian CDR")
    p = memoryview(raw)[4:]
    off = 8                                             # header.stamp: int32 sec, uint32 nanosec
    n = struct.unpack_from("<I", p, off)[0]             # header.frame_id: uint32 length (incl. NUL) + chars
    off = _align(off + 4 + n, 8)
    timebase = struct.unpack_from("<Q", p, off)[0]      # uint64 timebase
    off += 8
    off += 4 + 1 + 3                                    # uint32 point_num, uint8 lidar_id, uint8[3] rsvd
    off = _align(off, 4)
    count = struct.unpack_from("<I", p, off)[0]         # sequence<CustomPoint> length
    off += 4
    if count == 0:
        return timebase, np.zeros(0, CUSTOM_DT)
    need = (count - 1) * 20 + 19
    pts = np.frombuffer(p[off:off + need].tobytes() + b"\0", dtype=CUSTOM_DT, count=count)
    return timebase, pts


def serialize_custom_msg(t0_ns, t_ns, pts, frame_id=b"livox_frame", lidar_id=0):
    """One frame as a serialized CustomMsg (CDR little endian): timebase = frame start, offset_time = ns from it.
    For LIO filters that treat every message as one scan (FAST-LIO lidar_type 1)."""
    head = bytearray(struct.pack("<iI", int(t0_ns // 1_000_000_000), int(t0_ns % 1_000_000_000)))
    s = frame_id + b"\0"
    head += struct.pack("<I", len(s)) + s
    head += b"\0" * ((-len(head)) % 8)
    head += struct.pack("<QIB3x", int(t0_ns), len(pts), lidar_id)
    head += b"\0" * ((-len(head)) % 4)
    head += struct.pack("<I", len(pts))
    out = np.zeros(len(pts), CUSTOM_DT)
    out["offset_time"] = (t_ns - t0_ns).astype(np.uint32)
    for k in ("x", "y", "z", "reflectivity", "tag", "line"):
        out[k] = pts[k]
    body = out.tobytes()[:-1] if len(pts) else b""  # the last element has no trailing pad byte
    return b"\x00\x01\x00\x00" + bytes(head) + body


class FrameBuilder:
    """Accumulate per-packet points on lidar time and cut frames of frame_s."""

    def __init__(self, frame_s=0.1, min_range_m=0.1):
        self.frame_ns = int(frame_s * 1e9)
        self.min_range = min_range_m
        self.t0 = None
        self.parts = []
        self.stats = {"packets": 0, "frames": 0, "points_in": 0, "points_out": 0, "tag_dropped": 0, "zero_dropped": 0,
                      "late_packets": 0}

    def add(self, timebase, pts):
        """The frames this message finishes: [(t0_ns, points POINT_DT, (t_ns, raw CustomPoint))], often empty."""
        self.stats["packets"] += 1
        self.stats["points_in"] += len(pts)
        if not len(pts):
            return []
        t = timebase + pts["offset_time"].astype(np.int64)
        keep = (pts["tag"] & 0x0F) == 0
        self.stats["tag_dropped"] += int((~keep).sum())
        r2 = pts["x"].astype(np.float64) ** 2 + pts["y"] ** 2 + pts["z"] ** 2
        nz = r2 > self.min_range ** 2
        self.stats["zero_dropped"] += int((keep & ~nz).sum())
        keep &= nz
        out = []
        if t[-1] - t[0] >= self.frame_ns // 2:
            # the driver already sends whole frames (publish_freq honoured): one message = one frame. Cutting them on
            # a fixed grid would merge two whenever a message starts just before the boundary.
            if self.parts:
                out.append(self.flush())
            self.t0 = int(t[0])
            self.parts.append((t[keep], pts[keep]))
            out.append(self.flush())
            self.t0 = None
            return out
        if self.t0 is None:
            self.t0 = int(t[0])
        if t[0] < self.t0:
            self.stats["late_packets"] += 1
        if t[0] >= self.t0 + self.frame_ns and self.parts:
            out.append(self.flush())
            self.t0 = int(t[0]) if t[0] >= self.t0 + 2 * self.frame_ns else self.t0 + self.frame_ns
        self.parts.append((t[keep], pts[keep]))
        return out

    def flush(self):
        t = np.concatenate([a for a, _ in self.parts])
        p = np.concatenate([b for _, b in self.parts])
        self.parts = []
        t0 = self.t0
        frame = np.zeros(len(p), POINT_DT)
        frame["x"], frame["y"], frame["z"] = p["x"], p["y"], p["z"]
        frame["intensity"] = p["reflectivity"].astype(np.float32)
        frame["ring"] = p["line"].astype(np.uint16)
        frame["time"] = (t - t0).astype(np.float32)
        self.stats["frames"] += 1
        self.stats["points_out"] += len(frame)
        return t0, frame, (t, p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-cloud", default="/livox/lidar")
    ap.add_argument("--in-imu", default="/livox/imu")
    ap.add_argument("--out-cloud", default="utlidar/cloud_livox_mid360")
    ap.add_argument("--out-imu", default="utlidar/imu_livox_mid360")
    ap.add_argument("--frame-s", type=float, default=0.1)
    ap.add_argument("--min-range-m", type=float, default=0.1)
    ap.add_argument("--frame-id", default="livox_frame")
    ap.add_argument("--custom-out", default=None, help="also publish each frame as a 10 Hz CustomMsg on this topic "
                                                         "(e.g. /livox/lidar_10hz for FAST-LIO: one message = one scan)")
    ap.add_argument("--stats-s", type=float, default=5.0, help="print a stats line every S s (0 = off)")
    ap.add_argument("--seconds", type=float, default=0.0)
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
    from rclpy.serialization import serialize_message  # noqa: F401  (keeps rclpy's serialization loaded)
    from builtin_interfaces.msg import Time
    from livox_ros_driver2.msg import CustomMsg
    from sensor_msgs.msg import Imu, PointCloud2, PointField

    rclpy.init()
    node = rclpy.create_node("g1_livox_to_utlidar")
    out_qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    pub_cloud = node.create_publisher(PointCloud2, args.out_cloud, out_qos)
    pub_imu = node.create_publisher(Imu, args.out_imu, QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                                                                   history=HistoryPolicy.KEEP_LAST))
    fields = [PointField(name=n, offset=o, datatype=d, count=1)
              for n, o, d in (("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7), ("ring", 16, 4), ("time", 18, 7))]
    pub_custom = node.create_publisher(CustomMsg, args.custom_out, out_qos) if args.custom_out else None
    fb = FrameBuilder(args.frame_s, args.min_range_m)
    imu_n = [0]

    def publish(t0, frame):
        msg = PointCloud2()
        msg.header.stamp = Time(sec=int(t0 // 1_000_000_000), nanosec=int(t0 % 1_000_000_000))
        msg.header.frame_id = args.frame_id
        msg.height, msg.width = 1, len(frame)
        msg.fields = fields
        msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 22, 22 * len(frame), True
        msg.data = array.array("B", frame.tobytes())
        pub_cloud.publish(msg)

    def on_cloud(raw):
        try:
            timebase, pts = parse_custom_msg(raw)
        except Exception as exc:  # malformed sample: count it, keep running
            fb.stats["parse_errors"] = fb.stats.get("parse_errors", 0) + 1
            fb.stats["last_error"] = str(exc)
            return
        for t0, frame, (t_raw, p_raw) in fb.add(timebase, pts):
            publish(t0, frame)
            if pub_custom is not None:
                pub_custom.publish(serialize_custom_msg(t0, t_raw, p_raw, frame_id=args.frame_id.encode()))

    def on_imu(msg):
        imu_n[0] += 1
        pub_imu.publish(msg)

    node.create_subscription(CustomMsg, args.in_cloud, on_cloud, qos_profile_sensor_data, raw=True)
    node.create_subscription(Imu, args.in_imu, on_imu, qos_profile_sensor_data)
    print(f"[livox_to_utlidar] {args.in_cloud} -> {args.out_cloud} ({1 / args.frame_s:.0f} Hz frames), "
          f"{args.in_imu} -> {args.out_imu}", flush=True)
    t0 = last = time.monotonic()
    prev = dict(fb.stats)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if args.stats_s and now - last >= args.stats_s:
                dt = now - last
                s = fb.stats
                print(f"[livox_to_utlidar] {(s['packets'] - prev['packets']) / dt:.0f} packets/s -> "
                      f"{(s['frames'] - prev['frames']) / dt:.1f} frames/s, "
                      f"{(s['points_out'] - prev['points_out']) / max(s['frames'] - prev['frames'], 1):.0f} points/frame, "
                      f"tag-dropped {s['tag_dropped'] - prev['tag_dropped']}, imu {imu_n[0] / (now - t0):.0f} Hz"
                      + (f", parse errors {s.get('parse_errors')}: {s.get('last_error')}" if s.get("parse_errors") else ""),
                      flush=True)
                prev, last = dict(s), now
            if args.seconds and now - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
