"""The node's safety reactions, driven with fake robot inputs (no DDS, no runner, CPU planner)."""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from g1deploy import contract as C


class FakeState:
    """Stands in for DdsState: a standing robot whose pose / tilt / freshness the test controls."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tilt = 0.0
        self.stale = False
        self.xy = np.zeros(2)

    def _q(self):
        h = 0.5 * self.tilt
        return np.r_[self.xy, 0.79, np.cos(h), 0.0, np.sin(h), 0.0, C.DEFAULT_MJ]

    def snapshot(self):
        q = self._q()
        age = 1.0 if self.stale else 0.001
        return dict(qpos=q, dq=np.zeros(29), imu=q[3:7], gyro=np.zeros(3), lin_w=np.zeros(3), ang_w=np.zeros(3),
                    age_low=age, age_odo=age)

    def history(self, t_end, n):
        return [self._q().astype(np.float32) for _ in range(n)]

    def qpos_at(self, t):
        return self._q()


class FakeCloud:
    """Feeds empty-floor world clouds until `stop` is set."""

    def __init__(self):
        self.last = -1.0
        self.stop = False

    def poll(self):
        if self.stop:
            return []
        self.last = time.monotonic()
        g = np.random.default_rng(0).uniform(-3, 3, (2000, 2))
        pts = np.c_[g, np.zeros(len(g))]
        return [dict(points=pts, origin=np.array([0.0, 0.0, 1.25]), origins=None, dt=None, wall=self.last)]


@pytest.fixture(scope="module")
def node():
    from g1deploy.node import StudentNode, parse

    args = parse(["--goal", "3.0", "0.0", "--device", "cpu", "--dry-run", "--prescan-s", "0.1",
                  "--route", "privileged", "--latency-guess-s", "0.8"])
    n = StudentNode(args)
    yield n
    n.pool.close()


def _run(node, state, cloud, until, timeout=20.0):
    node.stop_flag.clear()  # run() resets its own per-run state (that is part of what these tests check)
    events = []
    import g1deploy.node as N

    orig = N.emit
    N.emit = lambda e, **kw: (events.append(e), orig(e, **kw))
    result = {}
    th = threading.Thread(target=lambda: result.setdefault("r", node.run(state, cloud)), daemon=True)
    th.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout and not until(events):
        time.sleep(0.05)
    node.stop_flag.set()
    th.join(5)
    N.emit = orig
    return events, result.get("r")


def test_tilt_triggers_estop(node):
    st, cl = FakeState(), FakeCloud()
    threading.Timer(4.0, lambda: setattr(st, "tilt", 0.9)).start()
    ev, r = _run(node, st, cl, lambda e: "estop" in e)
    assert "estop" in ev and r == "estop_tilt"


def test_stale_state_triggers_estop(node):
    st, cl = FakeState(), FakeCloud()
    threading.Timer(1.0, lambda: setattr(st, "stale", True)).start()
    ev, r = _run(node, st, cl, lambda e: "estop" in e)
    assert r == "stale_state"


def test_lidar_loss_settles_to_stand(node):
    st, cl = FakeState(), FakeCloud()
    threading.Timer(4.0, lambda: setattr(cl, "stop", True)).start()
    ev, _ = _run(node, st, cl, lambda e: "stand" in e)
    assert "stand" in ev and "estop" not in ev


def test_arrival_settles_to_stand(node):
    st, cl = FakeState(), FakeCloud()
    threading.Timer(4.0, lambda: setattr(st, "xy", np.array([3.0, 0.0]))).start()
    ev, _ = _run(node, st, cl, lambda e: "arrived" in e and "stand" in e)
    assert "arrived" in ev and "stand" in ev


class BeamCloud(FakeCloud):
    """Floor + the underside of a beam (1.15 m) just ahead of the robot, seen from the head."""

    def poll(self):
        if self.stop:
            return []
        self.last = time.monotonic()
        rng = np.random.default_rng(1)
        g = np.c_[rng.uniform(0.3, 2.5, 3000), rng.uniform(-1, 1, 3000)]
        floor = np.c_[g, np.zeros(len(g))]
        b = np.c_[rng.uniform(0.35, 0.55, 600), rng.uniform(-0.7, 0.7, 600), np.full(600, 1.15)]
        return [dict(points=np.vstack([floor, b]), origin=np.array([0.0, 0.0, 1.25]), origins=None, dt=None,
                     wall=self.last)]


def test_soft_stop_is_deferred_under_a_beam(node):
    st, cl = FakeState(), BeamCloud()
    threading.Timer(4.0, lambda: setattr(cl, "stop", True)).start()
    threading.Timer(8.0, lambda: setattr(st, "xy", np.array([-2.0, 0.0]))).start()  # walks clear of the beam
    ev, _ = _run(node, st, cl, lambda e: "stand" in e, timeout=25.0)
    assert "stand_deferred" in ev and "stand" in ev and ev.index("stand_deferred") < ev.index("stand")


def test_rejected_first_plan_is_replanned(node):
    """A first plan dropped by the sanity check (e.g. a crouch under a phantom ceiling) must not deadlock RUN: with no
    active plan the node has to ask again (rehearsal 2026-10-23: one dropped first plan left the robot standing in
    RUN for 90 s)."""
    st, cl = FakeState(), FakeCloud()
    orig, calls = node.sane, []

    def reject_first(plan):
        calls.append(1)
        return "test: rejected first plan" if len(calls) == 1 else orig(plan)

    node.sane = reject_first
    try:
        ev, _ = _run(node, st, cl, lambda e: "run" in e, timeout=30.0)
    finally:
        node.sane = orig
    assert "plan_dropped" in ev and "run" in ev and ev.index("plan_dropped") < ev.index("run")


def test_clear_map_only_while_standing(node):
    """`clear_map` (operator: a spotter stood close) is refused while the walk plans on the map, and after the stand
    replaces the memory; the next walk then waits for fresh frames."""
    st, cl = FakeState(), FakeCloud()
    threading.Timer(1.0, lambda: node.commands.append({"cmd": "clear_map"})).start()
    threading.Timer(4.0, lambda: setattr(st, "xy", np.array([3.0, 0.0]))).start()
    seen = {}

    def until(ev):
        if "stand" in ev and "sent" not in seen:
            seen["sent"], seen["memory"] = True, node.memory
            node.commands.append({"cmd": "clear_map"})
        return "map_cleared" in ev

    ev, _ = _run(node, st, cl, until)
    assert "command_refused" in ev and "map_cleared" in ev
    assert node.memory is not seen["memory"] and node.map_ready_tick > node.tick - int(10.0 / 0.02) - 5


def test_lio_lost_in_fused_odometry_stands_the_robot(node):
    """`--odom fused`: once the filter calls LIO lost (rejected or silent), the walk stops and no GO walks on legs."""
    st, cl = FakeState(), FakeCloud()
    node.fused = SimpleNamespace(lost=None, filter=SimpleNamespace(x=np.array([0.0, 0.0, 0.0, 1.05, 0.01]), bad_since=None, discontinuity=None),
                                 speed_alarm=None)
    threading.Timer(4.0, lambda: setattr(node.fused, "lost", time.monotonic())).start()
    try:
        ev, _ = _run(node, st, cl, lambda e: "lio_lost" in e and "stand" in e)
    finally:
        del node.fused
        node.lio_lost_handled = False
    assert "lio_lost" in ev and "stand" in ev and "estop" not in ev
