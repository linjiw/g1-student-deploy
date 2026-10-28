"""Leg odometry + IMU + LIO in one small EKF: LIO keeps the pose honest, legs bridge its gaps and expose its faults.

State (LIO's world frame): x, y, yaw, and the correction of the leg odometry - `s` multiplies its displacement and
`beta` turns it (true step = s R(beta) leg step). On lab-g1 (2026-10-27) leg odometry counted 0.82-0.94 of the
distance FAST-LIO did (s ~ 1.06-1.22) and turned its steps -1..-2 deg on average (beta ~ +1..+2 deg; 1 cm contact),
with +-7-9 % and +-3-9 deg from second to second, so the correction is a slow random walk and the step itself is noisy (VILENS estimates a velocity bias the
same way; Cerberus calibrates the kinematics online).

  predict(d_body, dyaw, dt)  a leg-odometry displacement in the body frame since the last call (frame-free: rotated
                          by the IMU's yaw at the time), the IMU's yaw change and the time step: the world step is
                          s R(yaw + beta) d_body
  update_lio(xy, yaw, t)  a LIO pose; rejected when it implies a speed above `max_speed_m_s` since the last accepted one
                          (a jump or a runaway: LIO's own plausibility, legs not needed), or its Mahalanobis distance
                          exceeds `gate_chi2` (3 dof). Rejections for longer than `lost_after_s` mark LIO lost
                          (`lio_lost` is the time it started): the filter keeps going on the calibrated legs and the
                          caller should stop the robot within its bridge time

Process noise grows with the distance walked (none while standing), as the leg error does: `step_sigma` 0.15 covers
the measured per-second jitter (stride +-7-9 %, heading +-3-9 deg = +-5-16 % sideways). Tuned on the three walks:
tighter leg noise rejected good LIO poses on the walk with the largest heading jitter.
"""

from __future__ import annotations

import math

import numpy as np


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class LegLioFilter:
    def __init__(self, step_sigma=0.15, turn_sigma_m_per_rad=1.2, yaw_sigma_deg_sqrt_s=0.2, s_sigma=0.03,
                 beta_sigma_deg=1.0, lio_xy_sigma=0.03, lio_yaw_sigma_deg=3.0, gate_chi2=16.3, lost_after_s=0.5,
                 max_speed_m_s=1.5, jump_m=0.25):
        """`step_sigma`: the 1-sigma error of a step as a fraction of its length; `yaw_sigma_deg_sqrt_s`: IMU yaw noise
        density; `s_sigma`, `beta_sigma_deg`: correction random walk per sqrt(metre); `gate_chi2` 16.3 = 99.9 % (3 dof);
        `max_speed_m_s`: LIO implying more than this since its last accepted pose is a fault. `jump_m`: LIO moving
        this much more (or less) than the filter's own prediction between two consecutive LIO poses is a
        discontinuity, latched in `discontinuity` (a jump; between two LIO poses the legs differ from LIO by up to 13 cm
        on the real walks, 21 cm in sim turns).
        `turn_sigma_m_per_rad`: extra step error while turning (leg odometry breaks in turns: on one walk legs and LIO
        parted 0.3 m sideways over ~0.6 rad of turning; tuned on 3 real + 8 sim walks to give no false refusal, so
        the gate is a slow consistency check - jumps and runaways are caught by `jump_m` and the caller's speed check). LIO's yaw only corrects slow drift (3 deg): the pelvis yaw it gives wobbles +-1.5 deg
        with the gait, while the IMU's yaw changes are smooth."""
        self.turn_var = turn_sigma_m_per_rad ** 2
        self.step_sigma, self.yaw_var = step_sigma, math.radians(yaw_sigma_deg_sqrt_s) ** 2
        self.max_speed, self.last_lio, self.jump_m = max_speed_m_s, None, jump_m
        self.prev_lio = self.discontinuity = None  # (xy, filter xy) at the previous LIO pose; time of a jump
        self.s_var, self.b_var = s_sigma ** 2, math.radians(beta_sigma_deg) ** 2
        self.R = np.diag([lio_xy_sigma ** 2, lio_xy_sigma ** 2, math.radians(lio_yaw_sigma_deg) ** 2])
        self.gate, self.lost_after = gate_chi2, lost_after_s
        self.x = self.P = None
        self.bad_since = self.lio_lost = None
        self.accepted = self.rejected = 0

    def init(self, xy, yaw, s=1.1, beta=0.0):
        self.x = np.array([xy[0], xy[1], yaw, s, beta], dtype=float)
        self.P = np.diag([1e-4, 1e-4, math.radians(0.5) ** 2, 0.1 ** 2, math.radians(5.0) ** 2])

    @property
    def pose(self):
        return self.x[:3].copy()

    def predict(self, d_body, dyaw, dt=0.01):
        x, y, th, s, b = self.x
        d = np.asarray(d_body, dtype=float)[:2]
        c, sn = math.cos(th + b), math.sin(th + b)
        dx, dy = s * (c * d[0] - sn * d[1]), s * (sn * d[0] + c * d[1])
        self.x = np.array([x + dx, y + dy, _wrap(th + dyaw), s, b])
        F = np.eye(5)
        F[0, 2] = F[0, 4] = -dy
        F[1, 2] = F[1, 4] = dx
        if s > 1e-6:
            F[0, 3], F[1, 3] = dx / s, dy / s
        dist = float(np.hypot(*d))
        q_xy = (self.step_sigma * dist) ** 2 + self.turn_var * dyaw ** 2 + 1e-8
        Q = np.diag([q_xy, q_xy,
                     self.yaw_var * dt + 1e-10, self.s_var * dist + 1e-10, self.b_var * dist + 1e-10])
        self.P = F @ self.P @ F.T + Q

    def update_lio(self, xy, yaw, t):
        """Returns (accepted, mahalanobis^2)."""
        H = np.zeros((3, 5))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        r = np.array([xy[0] - self.x[0], xy[1] - self.x[1], _wrap(yaw - self.x[2])])
        S = H @ self.P @ H.T + self.R
        d2 = float(r @ np.linalg.solve(S, r))
        jump = self.last_lio is not None and (
            np.hypot(xy[0] - self.last_lio[0], xy[1] - self.last_lio[1]) > self.max_speed * max(t - self.last_lio[2], 0.05))
        if self.prev_lio is not None:  # LIO's own step against the filter's step since the previous LIO pose
            d_lio = np.asarray(xy[:2], float) - self.prev_lio[0]
            d_flt = self.x[:2] - self.prev_lio[1]
            if np.hypot(*(d_lio - d_flt)) > self.jump_m:
                jump = True
                self.discontinuity = self.discontinuity or t
        if jump or d2 > self.gate:
            self.prev_lio = (np.asarray(xy[:2], float).copy(), self.x[:2].copy())
            self.rejected += 1
            self.bad_since = t if self.bad_since is None else self.bad_since
            if self.lio_lost is None and t - self.bad_since >= self.lost_after:
                self.lio_lost = self.bad_since
            return False, d2
        self.bad_since = None  # (lio_lost stays set: whether to trust a LIO that came back is the caller's call)
        self.last_lio = (float(xy[0]), float(xy[1]), float(t))
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ r
        self.x[2] = _wrap(self.x[2])
        self.x[3] = float(np.clip(self.x[3], 0.7, 1.6))
        self.P = (np.eye(5) - K @ H) @ self.P
        self.accepted += 1
        self.prev_lio = (np.asarray(xy[:2], float).copy(), self.x[:2].copy())  # after the correction
        return True, d2
