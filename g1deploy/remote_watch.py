#!/usr/bin/env python3
"""Wireless-remote watcher for the console: the remote's E-stop and takeover in every runner input mode.

The runner's interface manager evaluates only the ACTIVE input interface. While the student streams (ZMQ active) the
gamepad's Select (stop) and D-up are never read, so the remote alone cannot stop the robot mid-walk. This process
reads the same wireless_remote bytes from rt/lowstate (subscribe only, it sends nothing to the robot) and prints one
JSON line per button press; the console turns Select into the runner's global `o` stop (damping, any interface) and
D-up into a takeover.

  {"remote": "press", "button": "Select", "t": ...}     on every press edge (any button)
  {"remote": "status", "connected": true, "samples": N, "t": ...}   every --status-s, and when the link changes

usage: remote_watch.py --iface eno1 [--status-s 1.0]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.remote_check import decode  # noqa: E402

STALE_S = 0.5  # no lowstate for this long: the remote link counts as lost (the console shows it)


def emit(**kw):
    print(json.dumps({**kw, "t": round(time.monotonic(), 3)}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", required=True)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--status-s", type=float, default=1.0)
    args = ap.parse_args(argv)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ChannelFactoryInitialize(args.domain, args.iface)
    lock, latest = threading.Lock(), {"raw": None, "n": 0, "t": 0.0}

    def on_low(m):
        with lock:
            latest["raw"], latest["n"], latest["t"] = bytes(m.wireless_remote), latest["n"] + 1, time.monotonic()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_low, 1)
    last, connected, t_status = set(), None, 0.0
    emit(remote="status", connected=False, samples=0, note="watching rt/lowstate wireless_remote")
    while True:
        time.sleep(0.005)
        with lock:
            raw, n, t_raw = latest["raw"], latest["n"], latest["t"]
        now = time.monotonic()
        live = raw is not None and now - t_raw < STALE_S
        header = live and decode(raw)[2]
        if header != connected or now - t_status >= args.status_s:
            connected, t_status = header, now
            emit(remote="status", connected=bool(header), samples=n, lowstate=bool(live))
        if not live:
            last = set()
            continue
        pressed = decode(raw)[0]
        for b in sorted(pressed - last):
            emit(remote="press", button=b)
        last = pressed


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
