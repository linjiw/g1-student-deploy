"""Planner latency compensation: plan from the state the robot will be in when the plan lands.

The Isaac evaluation ran lock-step (physics paused while the planner ran), so every plan was
anchored at the state it was executed from. On hardware a plan requested at tick t lands at
t + L (L ≈ 15 ticks at 0.3 s): anchored at t, its first L frames are already in the past and the
robot runs L ticks further on the previous plan, clamped at its end.

Predictive anchoring: the request at t is made from a *predicted* history ending at t + L — the
measured history up to t, continued by the active plan's frames t+1 .. t+L — re-posed by the planar
rigid transform that maps the active plan's pose at t onto the measured pose at t (SONIC executes
plans relative to the measured pelvis, F28). The new plan is then timestamped from t + L.
"""

from __future__ import annotations

import math

import numpy as np

DT = 0.02


def _yaw(q_wxyz):
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _yaw_quat(dyaw):
    return np.array([math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def plan_frame(plan, anchor_tick, anchor_q, tick):
    """World qpos (36) of `plan` at control tick `tick` (frame i is at anchor + (i+1) dt; clamped)."""
    i = tick - anchor_tick - 1
    if i < 0:
        return np.asarray(anchor_q, dtype=np.float64)
    q = plan["qpos"]
    return np.asarray(q[min(i, len(q) - 1)], dtype=np.float64)


def predicted_history(measured_hist, active, tick, lead_ticks, n_frames):
    """`n_frames` world qpos frames ending at tick + lead_ticks.

    measured_hist: list of measured world qpos, last = tick. active: (plan, anchor_tick, anchor_q, _).
    """
    plan, anchor_tick, anchor_q, _ = active
    meas_now = np.asarray(measured_hist[-1], dtype=np.float64)
    plan_now = plan_frame(plan, anchor_tick, anchor_q, tick)
    dyaw = _yaw(meas_now[3:7]) - _yaw(plan_now[3:7])
    c, s = math.cos(dyaw), math.sin(dyaw)
    rot = np.array([[c, -s], [s, c]])
    out = []
    for tau in range(tick + lead_ticks - n_frames + 1, tick + lead_ticks + 1):
        if tau <= tick:
            out.append(np.asarray(measured_hist[-1 - (tick - tau)], dtype=np.float64))
            continue
        q = plan_frame(plan, anchor_tick, anchor_q, tau).copy()
        q[:2] = meas_now[:2] + rot @ (q[:2] - plan_now[:2])
        q[3:7] = _qmul(_yaw_quat(dyaw), q[3:7])
        out.append(q)
    return [q.astype(np.float32) for q in out]
