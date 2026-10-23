"""Deployment kit pieces that need no robot: the Livox adapter's parser and framing, the recorder's chunk files and
loader, the course library."""

import json
import struct
from pathlib import Path

import numpy as np

from g1deploy.adapters.livox_to_utlidar import CUSTOM_DT, POINT_DT, FrameBuilder, parse_custom_msg

ROOT = Path(__file__).resolve().parents[1]


def custom_msg_cdr(timebase, pts, frame_id=b"livox_frame"):
    """livox_ros_driver2/CustomMsg serialized as ROS 2 CDR (little endian), the way rclpy hands it to raw callbacks."""
    body = bytearray(struct.pack("<iI", 12, 34))
    s = frame_id + b"\0"
    body += struct.pack("<I", len(s)) + s
    body += b"\0" * ((-len(body)) % 8)
    body += struct.pack("<QIB3x", timebase, len(pts), 0)
    body += b"\0" * ((-len(body)) % 4)
    body += struct.pack("<I", len(pts))
    for i, p in enumerate(pts):
        body += struct.pack("<IfffBBB", *p)
        if i < len(pts) - 1:
            body += b"\0"  # next element's uint32 aligned to 4
    return bytes(b"\x00\x01\x00\x00" + body)


def test_custom_msg_parse_matches_fields():
    pts = [(10 * i, 1.0 + i, -2.0, 0.5 * i, 7, 0, i % 4) for i in range(96)]
    tb, arr = parse_custom_msg(custom_msg_cdr(123456789, pts))
    assert tb == 123456789 and len(arr) == 96 and arr.dtype == CUSTOM_DT
    assert np.allclose(arr["x"], [p[1] for p in pts]) and np.array_equal(arr["offset_time"], [p[0] for p in pts])
    assert np.array_equal(arr["line"], [p[6] for p in pts])


def test_frame_builder_cuts_frames_and_filters():
    fb = FrameBuilder(frame_s=0.1)
    frames = []
    for k in range(25):  # 25 packets of 96 points, 10 ms apart -> two full 0.1 s frames
        pts = np.zeros(96, CUSTOM_DT)
        pts["x"], pts["y"], pts["z"] = 2.0, 0.5, -1.0
        pts["offset_time"] = np.arange(96) * 100
        pts["tag"][0] = 0x01      # noise mark: dropped
        pts["x"][1] = 0.0         # (0,0,0) placeholder: dropped
        pts["y"][1] = pts["z"][1] = 0.0
        frames += fb.add(1_000_000_000 + k * 10_000_000, pts)
    assert len(frames) == 2
    t0, f, _ = frames[0]
    assert f.dtype == POINT_DT and f.dtype.itemsize == 22
    assert len(f) == 10 * 94  # 10 packets, 2 points dropped from each
    assert 0 <= f["time"].min() < 1e3 and f["time"].max() < 1e8  # ns from frame start (first kept point), inside 0.1 s
    assert fb.stats["tag_dropped"] == 25 and fb.stats["zero_dropped"] == 25


def test_recorder_chunks_roundtrip(tmp_path):
    from g1deploy.recorder import ChunkWriter
    from g1deploy.recording import open_session

    w = ChunkWriter(tmp_path / "rec", "lowstate")
    for i in range(10):
        w.add(float(i), tick=i, q=np.full(35, i, np.float32))
        if i == 4:
            w.write(w.take())
    w.write(w.take())
    z = ChunkWriter(tmp_path / "rec", "zmq_pose")
    z.add(0.5, msg=b"pose" + b"\x01\x02")
    z.add(0.6, msg=b"pose")
    z.write(z.take())
    (tmp_path / "rec" / "meta.json").write_text(json.dumps({"format": "g1deploy.recording.v1"}))
    rec = open_session(tmp_path)
    low = rec.stream("lowstate")
    assert rec.streams() == ["lowstate", "zmq_pose"] and np.array_equal(low["tick"], np.arange(10))
    assert low["q"].shape == (10, 35) and w.stats(10.0)["chunks"] == 2
    assert rec.stream("zmq_pose")["msg"] == [b"pose\x01\x02", b"pose"]


def test_course_library_is_valid():
    keys = {"scene_boxes_world", "initial_state_record", "goal_world_xy", "goal_radius_m", "deadline_s", "goal_hold_s",
            "fall_height_m", "fall_tilt_rad", "obstacle_contact_threshold_n", "goal_speed_m_s"}
    courses = sorted((ROOT / "sim/courses").glob("*/task.json"))
    assert len(courses) >= 6
    for t in courses:
        doc = json.loads(t.read_text())
        assert keys <= set(doc), t
        assert (ROOT / doc["initial_state_record"]).exists()
        for b in doc["scene_boxes_world"]:
            assert b["z_top"] > b["z_bottom"] and min(b["half_xy"]) > 0


def test_custom_msg_serializer_roundtrip():
    from g1deploy.adapters.livox_to_utlidar import serialize_custom_msg

    pts = np.zeros(500, CUSTOM_DT)
    pts["x"], pts["y"], pts["z"] = np.linspace(1, 2, 500), 0.5, -1.0
    pts["reflectivity"], pts["line"] = 9, np.arange(500) % 4
    t0 = 1_700_000_000_123_456_789
    t = t0 + np.arange(500, dtype=np.int64) * 200_000
    tb, back = parse_custom_msg(serialize_custom_msg(t0, t, pts))
    assert tb == t0 and len(back) == 500
    assert np.array_equal(back["offset_time"], (t - t0).astype(np.uint32)) and np.allclose(back["x"], pts["x"])


def test_frame_builder_passes_whole_frames_through():
    """A driver that honours publish_freq sends 0.1 s messages: each must come out as exactly one frame, even when a
    message starts a little before the 0.1 s grid line (that used to merge two)."""
    fb = FrameBuilder(frame_s=0.1)
    frames = []
    for k in range(20):
        pts = np.zeros(2000, CUSTOM_DT)
        pts["x"] = 3.0
        pts["offset_time"] = np.linspace(0, 99_000_000, 2000).astype(np.uint32)
        jitter = (-300_000 if k % 3 == 0 else 200_000)
        frames += fb.add(5_000_000_000 + k * 100_000_000 + jitter, pts)
    assert len(frames) == 20 and all(len(f) == 2000 for _, f, _ in frames)
