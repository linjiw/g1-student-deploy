"""Leg-odometry-like pose error for simulation: tied to each step, not a random walk.

A random walk on (x, y, yaw) drifts while the robot stands and does not grow with walking. The error of the G1's leg
odometry (g1deploy/leg_odometry.py against FAST-LIO replays of three lab-g1 walks, 2026-10-27) is the other way round:
  - the stride is under-counted: per-1 s window scale 0.82-0.94 (walk mean), +-0.07-0.09 from window to window;
  - each step's displacement is turned: per-window heading bias -1..-2 deg on average (1 cm contact threshold; the
    3 cm threshold gave -5..-11 deg), +-3-9 deg from window to window; the IMU's yaw itself stays within ~0.6 deg;
  - nothing while standing (0.0-0.5 cm in 20 s);
  - in-place turns make it worse (the 3 cm threshold lost 80 % of the distance during a spin).
`MotionOdomError` integrates the true pelvis displacement through a per-window scale and rotation,
    p_est += scale_w * R(bias_w) @ (p_true(t) - p_true(t - dt)),   yaw_est = yaw_true + an optional yaw random walk,
and exposes `err` = (dx, dy, dyaw) = estimate - truth: the `pose_error` convention of the planner's lidar memory, so
lidar frames written with the estimate smear the memory as the robot's did (a 0.69 m gap closed to 0.45 m).

Presets (per-episode ranges; each episode draws its means once, each window draws around them):
  lab-g1-leg   what was measured (leg odometry, 1 cm contact)
  training     wider, the target a robust planner should hold: stride 0.70-1.05, heading bias +-8 deg

Used by sim/lockstep.py (`--odom-noise '{"motion": "lab-g1-leg"}'` or a dict of ranges) and the sim LIO stand-in
(tools/sim_lio_bridge.py `--noise '{"motion": ...}'`, the node registering raw lidar frames with that pose).
"""

from __future__ import annotations

import math

import numpy as np

WINDOW_S = 1.0
PRESETS = {
    "lab-g1-leg": {"scale": (0.82, 0.94), "bias_deg": (-2.0, -1.0), "scale_jitter": (0.07, 0.09),
                   "bias_jitter_deg": (2.6, 9.1), "turn_slip": (0.0, 0.0), "yaw_rw_deg_sqrt_s": (0.0, 0.0)},
    "training": {"scale": (0.70, 1.05), "bias_deg": (-8.0, 8.0), "scale_jitter": (0.05, 0.10),
                 "bias_jitter_deg": (2.0, 9.0), "turn_slip": (0.0, 0.3), "yaw_rw_deg_sqrt_s": (0.0, 0.2)},
}


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def ranges(spec):
    """A preset name, or a dict of (lo, hi) ranges / single values over the lab-g1-leg preset."""
    if isinstance(spec, str):
        return dict(PRESETS[spec])
    out = dict(PRESETS["lab-g1-leg"])
    for k, v in (spec or {}).items():
        if k not in out:
            raise ValueError(f"unknown odometry error key {k!r} (known: {', '.join(out)})")
        out[k] = tuple(v) if isinstance(v, (list, tuple)) else (float(v), float(v))
    return out


class MotionOdomError:
    """Callable like flow's StateEstimate: `q_est = model(q_true, dt)`; `err` = (dx, dy, dyaw); `cfg` is truthy."""

    def __init__(self, spec="lab-g1-leg", rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.cfg = {"motion": spec}
        self.p = {k: float(self.rng.uniform(*v)) for k, v in ranges(spec).items()}
        self.err = np.zeros(3)
        self.p_true = self.p_est = self.yaw_prev = None
        self.t_win, self.win, self.yaw_err = WINDOW_S, None, 0.0

    def _new_window(self):
        p = self.p
        self.win = (max(0.3, self.rng.normal(p["scale"], p["scale_jitter"])),
                    math.radians(self.rng.normal(p["bias_deg"], p["bias_jitter_deg"])))

    def step_xy(self, xy, yaw, dt):
        """The estimated xy for true `xy` / `yaw` after `dt` (the core, without a quaternion)."""
        xy = np.asarray(xy, dtype=np.float64)[:2]
        if self.p_true is None:
            self.p_true, self.p_est, self.yaw_prev = xy.copy(), xy.copy(), yaw
            self._new_window()
        self.t_win -= dt
        if self.t_win <= 0:
            self.t_win += WINDOW_S
            self._new_window()
        scale, bias = self.win
        yaw_rate = abs((yaw - self.yaw_prev + math.pi) % (2 * math.pi) - math.pi) / max(dt, 1e-6)
        scale = max(0.0, scale - self.p["turn_slip"] * yaw_rate)
        c, s = math.cos(bias), math.sin(bias)
        d = xy - self.p_true
        self.p_est = self.p_est + scale * np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        self.p_true, self.yaw_prev = xy.copy(), yaw
        self.yaw_err += math.radians(self.p["yaw_rw_deg_sqrt_s"]) * math.sqrt(dt) * float(self.rng.normal())
        self.err = np.array([self.p_est[0] - xy[0], self.p_est[1] - xy[1], self.yaw_err])
        return self.p_est.copy()

    def __call__(self, q, dt):
        q = np.asarray(q)
        e = np.array(q, dtype=np.float64)
        e[:2] = self.step_xy(e[:2], _yaw(e[3:7]), dt)
        h = 0.5 * self.yaw_err
        e[3:7] = _quat_mul([math.cos(h), 0.0, 0.0, math.sin(h)], e[3:7])
        return e.astype(q.dtype)
