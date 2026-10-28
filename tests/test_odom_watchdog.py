"""FAST-LIO cross-checked by leg odometry (node.OdomWatchdog)."""
import numpy as np

from g1deploy.node import OdomWatchdog


def run(wd, v_lio, v_leg, t0=0.0, t1=2.0, dt=0.02, p0=(np.zeros(2), np.zeros(2))):
    out, a, b = None, p0[0].copy(), p0[1].copy()
    for t in np.arange(t0, t1, dt):
        a, b = a + [v_lio(t) * dt, 0.0], b + [0.0, v_leg(t) * dt]  # the frames need not agree: speeds only
        out = out or wd.update(t, a, b)
    return out


def test_agreeing_odometries_pass_even_with_a_leg_under_count():
    assert run(OdomWatchdog(), lambda t: 0.6, lambda t: 0.45) is None


def test_a_lio_jump_or_a_large_disagreement_stops():
    assert "LIO 2.00" in run(OdomWatchdog(), lambda t: 2.0, lambda t: 0.5)
    assert "disagrees" in run(OdomWatchdog(), lambda t: 0.0, lambda t: 0.6)  # LIO frozen while the legs walk


def test_a_short_spike_does_not_stop():
    assert run(OdomWatchdog(), lambda t: 3.0 if 1.0 <= t < 1.1 else 0.5, lambda t: 0.5) is None
