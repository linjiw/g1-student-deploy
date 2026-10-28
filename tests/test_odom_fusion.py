"""g1deploy/odom_fusion.py: legs + IMU + LIO in one EKF."""
import math

import numpy as np

from g1deploy.odom_fusion import LegLioFilter


def walk(f, n_s=8.0, v=0.5, scale=0.85, beta_deg=-3.0, lio=lambda t, p: p, outage=(None, None), rng=None, dt=0.01):
    """A straight walk along +x at `v`; legs report `scale` R(beta) of each true step (body frame), LIO the truth
    every 0.1 s (through `lio`, e.g. a jump), except during `outage`. Returns (true xy, fused xy, raw-leg xy at end)."""
    rng = rng or np.random.default_rng(0)
    b = math.radians(beta_deg)
    true, raw = np.zeros(2), np.zeros(2)
    f.init([0.0, 0.0], 0.0)
    for k in range(int(n_s / dt)):
        t = k * dt
        step = np.array([v * dt, 0.0])
        true = true + step
        noisy = scale * (1 + 0.08 * rng.normal()) * np.array([math.cos(b) * step[0], math.sin(b) * step[0]])
        raw = raw + noisy
        f.predict(noisy, 0.0)
        if k % 10 == 0 and not (outage[0] is not None and outage[0] <= t < outage[1]):
            f.update_lio(lio(t, true), 0.0, t)
    return true, f.pose[:2], raw


def test_calibration_converges_and_tracks():
    f = LegLioFilter()
    true, fused, _ = walk(f)
    assert np.linalg.norm(fused - true) < 0.03
    assert abs(f.x[3] - 1 / 0.85) < 0.05 and abs(math.degrees(f.x[4]) - 3.0) < 1.5  # the correction undoes the legs


def test_an_outage_is_bridged_better_than_raw_legs():
    f = LegLioFilter()
    true, fused, _ = walk(f, outage=(6.0, 8.1))  # the last ~2 s without LIO
    raw_err = 0.15 * 0.5 * 2.0  # the uncalibrated legs lose ~15 % of the 1 m walked in the outage
    assert np.linalg.norm(fused - true) < 0.5 * raw_err


def test_a_lio_jump_is_rejected_and_marks_lio_lost():
    f = LegLioFilter()
    true, fused, _ = walk(f, lio=lambda t, p: p + ([0.5, 0.0] if t >= 5.0 else [0.0, 0.0]))
    assert f.lio_lost is not None and 5.0 <= f.lio_lost < 5.2
    assert np.linalg.norm(fused - true) < 0.1  # the fused pose did not follow the jump
