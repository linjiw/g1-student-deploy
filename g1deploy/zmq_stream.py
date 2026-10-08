"""SONIC ZMQ streaming protocol v1 (gear_sonic_deploy `--input-type zmq`), plan -> stream frames.

Wire format (`zmq_packed_message_subscriber.hpp`): one single-part ZMQ message
    [topic bytes "pose"][1280-byte null-padded JSON header][fields, little endian, header order]
header = {"v": 1, "endian": "le", "count": N, "fields": [{"name", "dtype", "shape"}, ...]}.
Protocol v1 fields: joint_pos [N,29] and joint_vel [N,29] in IsaacLab order, body_quat [N,4]
(w,x,y,z, pelvis, any yaw frame: the runner re-captures heading at stream start),
frame_index [N] (global 50 Hz index; one stored frame is consumed per control tick); optional
`timestamp_monotonic` (lateness alarm), `catch_up` (bool).

Runner semantics this module relies on (`streamed_motion_merger.hpp`):
- the first chunk resets playback to its first frame_index and auto-plays;
- a chunk is merged if it starts after the window start (playback - 5 frames) and ends after the
  window end; it overwrites from its first frame_index, older frames are kept in front;
- otherwise (stale or rewinding chunk) playback resets to the chunk start and heading is re-captured;
- playback stalls when fewer than 46 frames are ahead.

Framing: the streamed frame at global tick tau is plan frame (tau - anchor_tick - 1), the anchor
itself at tau = anchor_tick, clamped to the last plan frame beyond the plan (= the Isaac feed's
clamp). At playback tick g the runner then reads frames g + 5k, which are exactly the rows
`feed.target` built for SONIC in Isaac.
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import contract as C

HEADER_SIZE = 1280
TOPIC = b"pose"
_DTYPES = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "u8": np.uint8, "bool": np.bool_}
_NAMES = {np.dtype(v).str[1:]: k for k, v in _DTYPES.items()}
LOOKAHEAD_FRAMES = 46  # runner stalls with fewer frames ahead (observation reads up to +45)


def pack(fields, topic=TOPIC, version=1):
    """fields: list of (name, array). Returns the single-part message bytes."""
    header_fields, blobs, count = [], [], None
    for name, arr in fields:
        a = np.ascontiguousarray(arr)
        dt = a.dtype.newbyteorder("<") if a.dtype.byteorder == ">" else a.dtype
        key = {np.dtype(np.float32): "f32", np.dtype(np.float64): "f64", np.dtype(np.int32): "i32",
               np.dtype(np.int64): "i64", np.dtype(np.uint8): "u8", np.dtype(np.bool_): "bool"}[np.dtype(dt)]
        header_fields.append({"name": name, "dtype": key, "shape": list(a.shape)})
        blobs.append(a.astype(dt.newbyteorder("<"), copy=False).tobytes())
        if count is None and a.ndim >= 1:
            count = int(a.shape[0])
    header = json.dumps({"v": version, "endian": "le", "count": count or 0, "fields": header_fields},
                        separators=(",", ":")).encode()
    if len(header) > HEADER_SIZE:
        raise ValueError(f"header {len(header)} bytes exceeds {HEADER_SIZE}")
    return topic + header.ljust(HEADER_SIZE, b"\0") + b"".join(blobs)


def unpack(message, topic=TOPIC):
    if not message.startswith(topic):
        raise ValueError("topic mismatch")
    body = message[len(topic):]
    header = json.loads(body[:HEADER_SIZE].rstrip(b"\0"))
    off, out = HEADER_SIZE, {}
    for f in header["fields"]:
        dt = np.dtype(_DTYPES[f["dtype"]]).newbyteorder("<")
        n = int(np.prod(f["shape"])) * dt.itemsize
        out[f["name"]] = np.frombuffer(body[off:off + n], dtype=dt).reshape(f["shape"])
        off += n
    if off != len(body):
        raise ValueError(f"payload size mismatch: {off} vs {len(body)}")
    return header, out


def rotation_to_quat_wxyz(r):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(r)).as_quat()[..., [3, 0, 1, 2]]


def stream_frames(plan, anchor_tick, anchor_q, anchor_dq, start_tick, end_tick, joint_limits=None):
    """Frames for global ticks [start_tick, end_tick] (inclusive): MuJoCo-order q, dq, pelvis quat."""
    ticks = np.arange(start_tick, end_tick + 1)
    idx = np.clip(ticks - anchor_tick - 1, -1, len(plan["qpos"]) - 1)
    q = np.where(idx[:, None] < 0, np.asarray(anchor_q)[None, 7:], plan["qpos"][np.maximum(idx, 0), 7:])
    dq = np.where(idx[:, None] < 0, np.asarray(anchor_dq)[None], plan["joint_velocity"][np.maximum(idx, 0)])
    rot = np.where(idx[:, None, None] < 0, _quat_matrix(anchor_q[3:7])[None], plan["rotation"][np.maximum(idx, 0)])
    if joint_limits is not None:
        q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])
    return ticks, q, dq, rotation_to_quat_wxyz(rot)


def _quat_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_message(ticks, q_mj, dq_mj, quat_wxyz, timestamp=None, catch_up=True):
    fields = [
        ("joint_pos", C.mj_to_isaac(np.asarray(q_mj, dtype=np.float32))),
        ("joint_vel", C.mj_to_isaac(np.asarray(dq_mj, dtype=np.float32))),
        ("body_quat", np.asarray(quat_wxyz, dtype=np.float32)),
        ("frame_index", np.asarray(ticks, dtype=np.int64)),
        ("catch_up", np.asarray([catch_up], dtype=np.bool_)),
    ]
    if timestamp is not None:
        fields.append(("timestamp_monotonic", np.asarray([timestamp], dtype=np.float64)))
    return pack(fields)


class PoseStreamer:
    """PUB socket the runner SUB-connects to (`--zmq-host <this host> --zmq-port 5556 --zmq-topic pose`)."""

    def __init__(self, bind="tcp://*:5556"):
        import zmq

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 4)
        self.sock.bind(bind)
        self.sent = 0

    def send_plan(self, plan, anchor_tick, anchor_q, anchor_dq, start_tick, joint_limits=None, horizon_pad=LOOKAHEAD_FRAMES):
        """Stream from `start_tick` to the plan end + `horizon_pad` frames (clamped = Isaac semantics)."""
        end_tick = anchor_tick + len(plan["qpos"]) + horizon_pad
        ticks, q, dq, quat = stream_frames(plan, anchor_tick, anchor_q, anchor_dq, start_tick, end_tick, joint_limits)
        self.sock.send(pose_message(ticks, q, dq, quat, timestamp=time.monotonic()))
        self.sent += 1
        return ticks[0], ticks[-1]

    def close(self):
        self.sock.close(0)
