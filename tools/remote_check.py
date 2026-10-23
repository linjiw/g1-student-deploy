#!/usr/bin/env python3
"""Read-only check of the Unitree wireless remote through rt/lowstate (the runner's gamepad_manager reads the same bytes).

Press each operator button once while it runs; it reports which were seen. Sends nothing to the robot.
The buttons the deploy procedure relies on (docs/OPERATOR_GUIDE.md):
    Start   SONIC takes control          F1      hand the runner to the student's stream
    Up      (D-up) SONIC planner takeover Select  E-STOP: damping (the robot drops)

usage: remote_check.py --iface <nic> [--seconds 30] [--need Start,Select,F1,Up]
"""

from __future__ import annotations

import argparse
import struct
import sys
import threading
import time

BUTTONS = ["R1", "L1", "Start", "Select", "R2", "L2", "F1", "F2", "A", "B", "X", "Y", "Up", "Right", "Down", "Left"]


def decode(raw: bytes):
    """wireless_remote[40]: 2 header bytes, uint16 button bits, then lx rx ry L2 ly as float32."""
    keys, = struct.unpack("<H", raw[2:4])
    lx, rx, ry, l2, ly = struct.unpack("<5f", raw[4:24])
    return {n for i, n in enumerate(BUTTONS) if keys >> i & 1}, (lx, ly, rx, ry, l2), raw[:2] != b"\0\0"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", required=True)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--need", default="Start,Select,F1,Up")
    args = ap.parse_args(argv)
    need = [n.strip() for n in args.need.split(",") if n.strip()]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ChannelFactoryInitialize(args.domain, args.iface)
    lock, latest = threading.Lock(), {"raw": None, "n": 0}

    def on_low(m):
        with lock:
            latest["raw"], latest["n"] = bytes(m.wireless_remote), latest["n"] + 1

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_low, 1)
    seen, stick_max, header_ok, last = set(), 0.0, False, set()
    print(f"press each of: {', '.join(need)}  ({args.seconds:.0f} s)", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        time.sleep(0.01)
        with lock:
            raw = latest["raw"]
        if raw is None:
            continue
        pressed, sticks, header = decode(raw)
        header_ok |= header
        stick_max = max(stick_max, max(abs(v) for v in sticks))
        for b in sorted(pressed - last):
            print(f"  {time.monotonic() - t0:5.1f} s  {b}" + ("   <- needed" if b in need and b not in seen else ""), flush=True)
        seen |= pressed
        last = pressed
        if all(n in seen for n in need):
            break
    missing = [n for n in need if n not in seen]
    print(f"\nlowstate samples {latest['n']}, remote header present {header_ok}, stick deflection max {stick_max:.2f}")
    print("REMOTE OK: all needed buttons seen" if not missing else f"REMOTE NOT CONFIRMED: never saw {', '.join(missing)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
