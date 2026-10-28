"""Read a session recorded by g1deploy/recorder.py (format g1deploy.recording.v1) back into arrays.

    from g1deploy import recording as R
    rec = R.open_session("runs/console/robot-20261024-101500")     # the session dir, or its rec/ dir
    low = rec.stream("lowstate")          # dict of arrays, chunks concatenated: t, q (N, 35), imu_quat, ...
    for t, pts, t_s in rec.lidar_frames():   # raw sensor-frame points (placeholders dropped), point times (s)
        ...
    for t, fields in rec.zmq("pose"):     # decoded ZMQ messages (g1deploy.zmq_stream format)
        ...
    status = rec.node_status()            # list of {t, t_unix, status}

All times `t` are time.monotonic() of the deploy PC, the same clock as the console, node and runner logs.
A chunk that cannot be read (cut by a power loss or a kill while it was written) is skipped and listed in
`rec.skipped` [(stream, file, error)]; everything before it reads as usual.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ZMQ_TOPICS = {"pose": b"pose", "odom": b"odom", "cloud": b"cloud", "debug": b"g1_debug"}


class Session:
    def __init__(self, path):
        path = Path(path)
        self.dir = path / "rec" if (path / "rec").is_dir() else path
        self.root = self.dir.parent if self.dir.name == "rec" else self.dir
        meta = self.dir / "meta.json"
        try:
            self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        except ValueError:
            self.meta = {}
        self.skipped: list[tuple[str, str, str]] = []

    def _load(self, name, path, keys=None):
        """The chunk's arrays (those in `keys` with their _off and t, t_unix; all when None), or None (skipped, listed
        in self.skipped) when it cannot be read."""
        try:
            with np.load(path) as z:
                want = [k for k in z.files if keys is None or k in ("t", "t_unix") or k in keys
                        or (k.endswith("_off") and k[:-4] in keys)]
                return {k: z[k] for k in want}
        except Exception as exc:  # BadZipFile, EOFError, ValueError, OSError: a cut chunk
            item = (name, path.name, f"{type(exc).__name__}: {exc}")
            if item not in self.skipped:
                self.skipped.append(item)
            return None

    def streams(self):
        return sorted(p.name for p in self.dir.iterdir() if p.is_dir() and any(p.glob("*.npz")))

    def chunks(self, name):
        return sorted(p for p in (self.dir / name).glob("[0-9]*.npz") if ".tmp" not in p.name)

    def stream(self, name, keys=None):
        """All chunks of a fixed-shape stream, concatenated. Variable-length fields come back as a list of bytes."""
        parts: dict[str, list] = {}
        for path in self.chunks(name):
            z = self._load(name, path, keys)
            if z is None:
                continue
            for k in z:
                if k.endswith("_off") or (keys and k not in keys and k not in ("t", "t_unix")):
                    continue
                if k + "_off" in z:
                    blob, off = z[k], z[k + "_off"]
                    parts.setdefault(k, []).append([blob[off[i]:off[i + 1]].tobytes() for i in range(len(off) - 1)])
                else:
                    parts.setdefault(k, []).append(z[k])
        out = {}
        for k, v in parts.items():
            out[k] = sum(v, []) if isinstance(v[0], list) else np.concatenate(v)
        return out

    def lidar_frames(self, every=1):
        """Yield (t_recv, points (N, 3) raw sensor frame, t_s (N,) seconds from frame start) per recorded frame."""
        from g1deploy.perception.unitree_lidar import decode_cloud

        fields = [SimpleNamespace(**f) for f in (self.meta.get("lidar_fields") or [])]
        if not fields:
            raise ValueError("meta.json has no lidar_fields (recording still running, or no lidar frame seen)")
        i = 0
        for path in self.chunks("lidar"):
            z = self._load("lidar", path)
            if z is None:
                continue
            off = z["data_off"]
            for j in range(len(z["t"])):
                if i % every == 0:
                    stamp = SimpleNamespace(sec=int(z["stamp_sec"][j]), nanosec=int(z["stamp_nanosec"][j]))
                    msg = SimpleNamespace(data=z["data"][off[j]:off[j + 1]].tobytes(), fields=fields,
                                          point_step=int(z["point_step"][j]), width=int(z["width"][j]),
                                          height=int(z["height"][j]), is_bigendian=bool(z["is_bigendian"][j]),
                                          header=SimpleNamespace(stamp=stamp))
                    pts, t_s, _ = decode_cloud(msg)
                    yield float(z["t"][j]), pts, t_s
                i += 1

    def zmq(self, name):
        """Yield (t_recv, decoded fields) for a recorded ZMQ stream (pose, odom, cloud) in protocol v1."""
        from g1deploy.zmq_stream import unpack

        topic = ZMQ_TOPICS.get(name, name.encode())
        data = self.stream(f"zmq_{name}")
        for t, msg in zip(data.get("t", []), data.get("msg", [])):
            try:
                yield float(t), unpack(msg, topic=topic)[1]
            except ValueError:
                continue  # another protocol (g1_debug is the runner's own format): read the raw bytes instead

    def node_status(self):
        path = self.dir / "node_status.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut by power loss
        return out


def open_session(path):
    return Session(path)
