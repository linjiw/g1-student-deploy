"""Where the robot stands now relative to where it stood for the scan: the live map registered to the scan's map.

Goals are picked on a scan (ahead / left of the robot as it stood for the scan) but sent to a live session relative to
the robot now. On 2026-10-27 the robot turned 17 deg right and moved 0.2 m between the scan and GO (it hangs in the
gantry until the runner takes over), so an unconverted goal pointed somewhere else in the room and the rehearsal ran
from a start the robot never had. `start_check` registers the obstacle cells of the live map (StudentNode.map_snapshot,
robot frame now) to the scan's (Hub.scan_grid, robot frame at the scan): a heading search with an FFT cross-correlation
over translation, then a finer heading pass around the best. It reports the robot's move since the scan and the scan's
goal in the robot's frame now; `repose_task` moves a scan's sim course into that frame (the rehearsal from the real
start).
"""

from __future__ import annotations

import math

import numpy as np

from g1deploy import frames

OBSTACLE_CODES = "23"  # workspace.MAP_CODES: obstacles >= 0.3 m (the floor and low clutter do not register well)
RADIUS_M = 4.0  # cells this far from the robot take part
RES_M = 0.05  # correlation raster
MIN_OVERLAP = 0.6  # below this the match is not trusted (too little in common, or the scene changed)
MOVED_WARN_M, TURNED_WARN_DEG = 0.10, 5.0


def grid_points(grid, codes=OBSTACLE_CODES):
    """World xy of the cells of a coded grid (workspace.code_grid: rows, res, x0, y0) whose code is in `codes`."""
    pts = [(grid["x0"] + (i + 0.5) * grid["res"], grid["y0"] + (j + 0.5) * grid["res"])
           for i, row in enumerate(grid["rows"]) for j, ch in enumerate(row) if ch in codes]
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def to_frame(pts, pose):
    """World points into the frame of `pose` (x, y, yaw): (ahead, left)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(pts, dtype=float) - [x, y]
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def rotate(pts, theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(pts, dtype=float) @ np.array([[c, s], [-s, c]])


def _raster(pts, half, res):
    n = int(round(2 * half / res))
    g = np.zeros((n, n))
    i = np.floor((pts + half) / res).astype(int)
    ok = (i >= 0).all(1) & (i < n).all(1)
    g[i[ok, 0], i[ok, 1]] = 1.0
    # a 3-cell box blur: tolerant to the 10 cm cells and to small shape changes
    k = np.ones(3) / 3
    g = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 0, g)
    return np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, g)


def register(src, dst, radius=RADIUS_M, max_deg=30.0, res=RES_M):
    """Rigid 2D transform taking `src` points onto `dst` points (both robot-centred): dst ~ R(theta) src + t.
    Returns (theta_rad, t, overlap); overlap = the smaller of the shares of each map's points that land on the other's
    cells (0..1)."""
    src = np.asarray(src, float)[np.hypot(*np.asarray(src, float).T) < radius] if len(src) else np.zeros((0, 2))
    dst = np.asarray(dst, float)[np.hypot(*np.asarray(dst, float).T) < radius] if len(dst) else np.zeros((0, 2))
    if len(src) < 10 or len(dst) < 10:
        return 0.0, np.zeros(2), 0.0
    half = radius + 1.0
    D = np.fft.rfft2(_raster(dst, half, res))
    n = int(round(2 * half / res))

    def score(theta):
        S = _raster(rotate(src, theta), half, res)
        cc = np.fft.irfft2(D * np.conj(np.fft.rfft2(S)), s=(n, n))
        k = np.unravel_index(int(np.argmax(cc)), cc.shape)
        shift = np.array([(k[0] + n // 2) % n - n // 2, (k[1] + n // 2) % n - n // 2], float) * res
        return float(cc[k]) / max(float(S.sum()), 1e-9), shift

    best = max(((score(th), th) for th in np.radians(np.arange(-max_deg, max_deg + 1e-6, 1.0))), key=lambda r: r[0][0])
    for th in best[1] + np.radians(np.arange(-1.0, 1.01, 0.25)):
        r = score(th)
        if r[0] > best[0][0]:
            best = (r, th)
    (_, t), theta = best
    moved = rotate(src, theta) + t
    # both ways (a dense or a partly matching map must not pass): the share of each map's cells within one 10 cm
    # cell of the other's, where both maps could see (inside the smaller of the two extents)
    r = min(np.hypot(*moved.T).max(), np.hypot(*dst.T).max())

    def share(a, b):
        cells = {(int(x), int(y)) for x, y in np.floor(b / 0.1)}
        a = a[np.hypot(*a.T) < r]
        hit = sum(any((int(math.floor(x / 0.1)) + di, int(math.floor(y / 0.1)) + dj) in cells
                      for di in (-1, 0, 1) for dj in (-1, 0, 1)) for x, y in a)
        return hit / max(len(a), 1)
    return float(theta), t, min(share(moved, dst), share(dst, moved))


def start_check(scan_grid, live_map, goal_scan=None):
    """The robot now vs at the scan. `scan_grid`: Hub.scan_grid (coded grid + start [x, y, yaw]); `live_map`: the node's
    map_snapshot (coded grid + robot [x, y, yaw]); `goal_scan`: (ahead, left) of the robot at the scan.
    Returns moved_m / turned_deg (the robot's move since the scan; turned > 0 = to the left), the goal in the robot's
    frame now, overlap, the transform scan frame -> robot frame now (theta = theta_rad, t: p_robot_now = R(theta)
    p_scene + t), the node's pose when the live map was taken (P_c, odometry frame) and from both scene_from_odom
    (g1deploy/frames.py). `t` is the translation, never a time."""
    if not live_map or not live_map.get("ok"):
        return {"ok": False, "error": "no live map yet"}
    src = to_frame(grid_points(scan_grid), scan_grid["start"])
    dst = to_frame(grid_points(live_map), live_map["robot"])
    theta, t, overlap = register(src, dst)
    # the robot now in the scan's frame: the inverse transform of its origin
    pos = rotate(-np.asarray(t)[None], -theta)[0]
    p_c = frames.pose(live_map["robot"])
    out = {"ok": overlap >= MIN_OVERLAP, "overlap": round(overlap, 2), "theta": float(theta), "theta_rad": float(theta),
           "t": [float(v) for v in t], "P_c": p_c, "scene_from_odom": frames.scene_from_odom(theta, t, p_c),
           "moved_m": round(float(np.hypot(*pos)), 3), "turned_deg": round(-math.degrees(theta), 1),
           "robot_in_scan": [round(float(pos[0]), 3), round(float(pos[1]), 3)], "cells": [len(src), len(dst)]}
    if goal_scan is not None:
        g = rotate(np.asarray(goal_scan, float)[None], theta)[0] + t
        out["goal_now"] = [round(float(g[0]), 2), round(float(g[1]), 2)]
    out["warn"] = bool(out["ok"] and (out["moved_m"] > MOVED_WARN_M or abs(out["turned_deg"]) > TURNED_WARN_DEG))
    if not out["ok"]:
        out["error"] = f"the live map matches the scan poorly ({overlap:.0%}): rescan from here"
    return out


def repose_task(task, theta, t):
    """A sim course (scene_boxes_world in the scan's start frame) moved into the robot's frame now."""
    c, s = math.cos(theta), math.sin(theta)
    out = dict(task, scene_boxes_world=[])
    for b in task["scene_boxes_world"]:
        x, y = b["center_xy"]
        out["scene_boxes_world"].append(dict(b, center_xy=[c * x - s * y + t[0], s * x + c * y + t[1]],
                                             yaw=b.get("yaw", 0.0) + theta))
    if task.get("goal_world_xy") is not None:
        x, y = task["goal_world_xy"][:2]
        out["goal_world_xy"] = [c * x - s * y + t[0], s * x + c * y + t[1]]
    out["reposed"] = {"theta_deg": round(math.degrees(theta), 2), "t": [round(float(v), 3) for v in t]}
    return out
