#!/usr/bin/env python3
"""Hardware ladder H4: stream a recorded reference sequence into the SONIC runner, no planner.

Reconstructs the per-tick stream frames of a recorded episode (lock-step MuJoCo or Isaac:
`closed_loop.npz` with `references` (T, 10, 67) and `pre_action_qpos` (T, 36)): row 0 of each tick's
target is the reference frame for that tick; its absolute pelvis rotation is R_measured · R6.
Optionally time-scales it (`--slow 2` = half speed) and prefixes a settle-from-stand segment, then
sends it as ZMQ protocol v1 chunks at real-time pace, exactly as the node would.

usage: stream_replay.py <episode dir> [--slow 1.0] [--bind tcp://*:5556] [--dry-run]
Start the runner first (`--input-type zmq`), press `]`, then Enter, then run this.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1deploy import contract as C  # noqa: E402
from g1deploy.zmq_stream import LOOKAHEAD_FRAMES, PoseStreamer, pose_message, rotation_to_quat_wxyz  # noqa: E402
from tools.replay_isaac import absolute_reference_rotations  # noqa: E402


def frames_from_episode(ep: Path, slow: float = 1.0, settle_s: float = 1.0):
    cb = ep / "callback" if (ep / "callback").is_dir() else ep
    z = np.load(cb / "closed_loop.npz")
    refs, pre_q = z["references"], z["pre_action_qpos"]
    q, dq = refs[:, 0, :29].astype(np.float64), refs[:, 0, 29:58].astype(np.float64)
    rot = absolute_reference_rotations(refs, pre_q)[:, 0]
    if slow != 1.0:  # resample in time; velocities scale by 1/slow
        t_src = np.arange(len(q))
        t_new = np.arange(0, len(q) - 1, 1.0 / slow)
        q = np.stack([np.interp(t_new, t_src, q[:, j]) for j in range(29)], 1)
        dq = np.stack([np.interp(t_new, t_src, dq[:, j]) for j in range(29)], 1) / slow
        from scipy.spatial.transform import Rotation, Slerp

        rot = Slerp(t_src, Rotation.from_matrix(rot))(t_new).as_matrix()
    n = int(settle_s / C.DT)  # stand -> first frame
    a = (np.arange(1, n + 1) / n)[:, None]
    lead_q = (1 - a) * C.DEFAULT_MJ + a * q[0]
    q = np.vstack([lead_q, q])
    dq = np.vstack([np.diff(np.vstack([C.DEFAULT_MJ, lead_q]), axis=0) / C.DT, dq])
    rot = np.concatenate([np.repeat(rot[:1], n, 0), rot])
    return q, dq, rotation_to_quat_wxyz(rot)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episode")
    ap.add_argument("--slow", type=float, default=1.0)
    ap.add_argument("--bind", default="tcp://*:5556")
    ap.add_argument("--chunk-s", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    q, dq, quat = frames_from_episode(Path(args.episode), args.slow)
    n = len(q)
    print(f"{n} frames ({n * C.DT:.1f} s), joint range check: "
          f"{float(np.abs(dq).max()):.1f} rad/s max reference velocity", flush=True)
    if args.dry_run:
        msg = pose_message(np.arange(n), q, dq, quat)
        print(f"one-chunk message {len(msg)} bytes")
        return 0
    s = PoseStreamer(args.bind)
    time.sleep(0.5)  # let the runner's SUB connect
    t0, step = time.monotonic(), int(args.chunk_s / C.DT)
    sent_end = -1
    for start in range(0, n, step):
        end = min(n - 1, start + step + LOOKAHEAD_FRAMES + 50)
        idx = np.arange(start, end + 1)
        pad = np.full(LOOKAHEAD_FRAMES, end)  # hold the last frame at the end (Isaac clamp)
        sel = np.r_[idx, pad] if end == n - 1 else idx
        if start + len(sel) - 1 <= sent_end:
            break  # a chunk must extend the runner's window, or it resets playback (sequence fully sent)
        sent_end = start + len(sel) - 1
        s.sock.send(pose_message(np.arange(start, start + len(sel)), q[sel], dq[sel], quat[sel], timestamp=time.monotonic()))
        if (rest := t0 + (start + step) * C.DT - time.monotonic()) > 0:
            time.sleep(rest)
    print("replay sent", flush=True)
    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
