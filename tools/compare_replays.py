#!/usr/bin/env python3
"""Table of tools/replay_plans.py results: one row per (session, bundle, lidar), markdown on stdout.

usage: compare_replays.py runs/replays/*.json [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def row(path):
    d = json.loads(Path(path).read_text())
    s, rows = d["summary"], d["rows"]
    n = len(rows)
    walk = [r for r in rows if r.get("advance_m") is not None]
    calm = sum(1 for r in rows if str(r.get("guard", "")).startswith("calm") or str(r.get("rule", "")).startswith("calm"))
    stop = sum(1 for r in rows if "stop" in str(r.get("rule", "")) or "fallback" in str(r.get("rule", ""))
               or str(r.get("rule", "")).startswith("calm"))
    lat = np.array([r["latency_s"] for r in rows[1:]] or [np.nan])
    return {
        "session": Path(s["session"]).name, "bundle": Path(s["bundle"]).name,
        "lidar": "near-rear" if s["rear_blind"][1] else ("rear half" if s["rear_blind"][0] else "full"),
        "plans": n, "no_safe": f"{s['no_safe']}/{n}", "fallback/calm": f"{stop}/{n}", "calm": calm,
        "advance_p50_m": round(float(np.nanmedian([r["advance_m"] for r in walk])), 2) if walk else None,
        "min_pelvis_m": round(float(np.nanmin([r["min_pelvis_z"] for r in walk])), 2) if walk else None,
        "lat_p50_s": round(float(np.nanmedian(lat)), 2), "lat_p95_s": round(float(np.nanpercentile(lat, 95)), 2),
        "rebuild_iou_min": s.get("rebuild_occupied_iou_min"),
        "rules_flags": " ".join(f"{k}:{v}" for k, v in (s.get("rules_flags") or {}).items()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)
    rows = sorted((row(f) for f in args.files), key=lambda r: (r["session"], r["lidar"], r["bundle"]))
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
