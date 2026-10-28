"""What the deploy service sends: strict JSON (no NaN / Infinity: non-finite floats are null) and the compact layers of
the live view (docs: the node `view`, the hub's scan and scene endpoints).

memory  {lo: [x, y] (centre of cell [0][0]), res, n: [nx, ny], order: "C, first index x",
         support_b64 (uint8: 255 never seen, else round(min(h, 5.08) / 0.02), 0 = floor),
         ceiling_b64 (uint8: 255 no overhead return, else round(min(z, 5.08) / 0.02)),
         near_b64 (uint8 0/1: cells seen only from up close) | null when the source has no such layer}
cloud   {n, origin: [x, y, z], points_b64 (int16 LE xyz in cm relative to origin; finite points only, clipped to
         +-327 m), class_b64 (uint8 per point: 0 fused, 1 self-filtered, 2 rear-blind dropped)}
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np

HEIGHT_STEP_M, HEIGHT_MAX_M = 0.02, 5.08
CLASS_FUSED, CLASS_SELF, CLASS_REAR = 0, 1, 2


def clean(x):
    """A JSON-safe copy: numpy scalars and arrays as Python numbers and lists, non-finite floats as None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, Path):
        return str(x)
    if x is None or isinstance(x, str):
        return x
    return str(x)


def dumps(x) -> str:
    return json.dumps(clean(x), allow_nan=False, separators=(",", ":"))


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def unb64(s, dtype):
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


def height_code(h):
    """Heights (m; NaN = none) -> uint8 (255 none, else round(clip(h, 0, 5.08) / 0.02))."""
    h = np.asarray(h, dtype=float)
    out = np.full(h.shape, 255, dtype=np.uint8)
    ok = np.isfinite(h)
    out[ok] = np.round(np.clip(h[ok], 0.0, HEIGHT_MAX_M) / HEIGHT_STEP_M).astype(np.uint8)
    return out


def height_decode(code):
    code = np.asarray(code)
    return np.where(code == 255, np.nan, code.astype(float) * HEIGHT_STEP_M)


def encode_memory(support, ceiling, lo, res, near=None, **extra):
    """A memory grid (support / ceiling heights, index [x][y], cell (i, j) centred at lo + (i, j) res)."""
    support = np.asarray(support, dtype=float)
    out = dict(extra)
    out.update(lo=[float(v) for v in np.asarray(lo, dtype=float)[:2]], res=float(res), n=[int(v) for v in support.shape],
               order="C, first index x", support_b64=b64(height_code(support)), ceiling_b64=b64(height_code(ceiling)),
               near_b64=None if near is None else b64(np.asarray(near, dtype=bool).astype(np.uint8)))
    return out


def decode_memory(m):
    """(support, ceiling, near | None) arrays from encode_memory's dict."""
    n = tuple(m["n"])
    sup = height_decode(unb64(m["support_b64"], np.uint8).reshape(n))
    ceil = height_decode(unb64(m["ceiling_b64"], np.uint8).reshape(n))
    near = unb64(m["near_b64"], np.uint8).reshape(n).astype(bool) if m.get("near_b64") else None
    return sup, ceil, near


def encode_cloud(points, classes=None, origin=None, max_points=4000, rng=None, **extra):
    """World points [N, 3] (and a class per point) -> the cloud dict, each class decimated on its own to its share of
    `max_points`."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    c = np.zeros(len(p), dtype=np.uint8) if classes is None else np.asarray(classes, dtype=np.uint8).reshape(-1)
    ok = np.isfinite(p).all(axis=1)
    p, c = p[ok], c[ok]
    if len(p) > max_points:
        rng = rng or np.random.default_rng(0)
        keep = []
        for k in np.unique(c):
            idx = np.flatnonzero(c == k)
            share = max(1, int(round(max_points * len(idx) / len(p))))
            keep.append(idx if len(idx) <= share else np.sort(rng.choice(idx, share, replace=False)))
        sel = np.sort(np.concatenate(keep))
        p, c = p[sel], c[sel]
    o = np.asarray(origin if origin is not None else (p.mean(axis=0) if len(p) else np.zeros(3)), dtype=float)[:3]
    rel = np.clip(np.round((p - o) * 100.0), -32700, 32700).astype("<i2")
    out = dict(extra)
    out.update(n=int(len(p)), origin=[float(v) for v in o], points_b64=b64(rel), class_b64=b64(c))
    return out


def decode_cloud(cl):
    """(points [N, 3] m, classes [N]) from encode_cloud's dict."""
    rel = unb64(cl["points_b64"], "<i2").reshape(-1, 3).astype(float) / 100.0
    return rel + np.asarray(cl["origin"], dtype=float), unb64(cl["class_b64"], np.uint8)
