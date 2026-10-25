"""The node with `--guards deploy`: operator goal in the odometry frame, heading/guard status, measured anchoring."""

import numpy as np
import pytest

from tests.test_node_supervisor import FakeCloud, FakeState, _run


@pytest.fixture(scope="module")
def node():
    from g1deploy.node import StudentNode, parse

    args = parse(["--goal", "3.0", "0.0", "--device", "cpu", "--dry-run", "--prescan-s", "0.1",
                  "--route", "privileged", "--latency-guess-s", "0.8", "--guards", "deploy"])
    n = StudentNode(args)
    yield n
    n.pool.close()


def test_goal_in_the_odometry_frame_is_echoed(node):
    import g1deploy.node as N

    events = []
    orig = N.emit
    N.emit = lambda e, **kw: (events.append((e, kw)), orig(e, **kw))
    try:
        q = FakeState()._q()
        node.commands = [{"cmd": "goal", "frame": "odom", "x": 1.5, "y": -0.5, "seq": 7}]
        node.handle_commands(q, None)
        assert node.goal.tolist() == [1.5, -0.5] and node.goal_seq == 7
        node.commands = [{"cmd": "goal", "frame": "odom", "x": "far"}, {"cmd": "goal", "frame": "odom", "x": float("nan"), "y": 0}]
        node.handle_commands(q, None)
        assert node.goal.tolist() == [1.5, -0.5]
        assert [e for e, _ in events].count("command_refused") == 2
        node.commands = [{"cmd": "goal", "ahead": 1.0, "left": 0.0}]  # the relative form still works
        node.handle_commands(q, None)
        assert node.goal.tolist() == pytest.approx([1.0, 0.0]) and node.goal_seq is None
    finally:
        N.emit = orig
        node.goal = np.array([3.0, 0.0])


def test_guarded_node_plans_and_reports_heading(node):
    st, cl = FakeState(), FakeCloud()
    ev, _ = _run(node, st, cl, lambda e: "run" in e)
    assert "run" in ev and "estop" not in ev
    s = node.status
    assert s["guards"] == "deploy" and abs(s["yaw_rad"]) < 1e-6 and "guard_counts" in s


def test_a_fallback_plan_is_never_extrapolated(node):
    from g1deploy.guards import stand_plan

    q = FakeState()._q()
    node.history = [q.astype(np.float32) for _ in range(6)]
    node.active = (dict(stand_plan(q, 32), rule="calm_stand"), 0, q, np.zeros(29))
    node.route = getattr(node, "route", None) or __import__("types").SimpleNamespace()
    if not hasattr(node, "memory"):
        node.make_memory(q[:2])
    node.route_now = lambda xy, t: None
    for _ in range(400):  # the previous test's last request may still be in flight
        node.pool.poll()
        if node.pool.free():
            break
        __import__("time").sleep(0.05)
    worker = node.pool.free()[0]
    node.request(100, worker)
    jid = max(node.jobs)
    anchor_tick = node.jobs[jid][0]
    assert anchor_tick == 100  # measured: no latency lead after a calm plan
    for _ in range(200):  # let the worker finish before the module's other tests use it
        if node.pool.poll():
            break
        __import__("time").sleep(0.05)
    node.jobs.clear()


def test_measured_plan_after_a_predicted_one_is_not_stale(node):
    """A measured anchor (its request tick) is older than the predicted anchor of the plan it replaces."""
    from g1deploy.guards import stand_plan

    q = FakeState()._q()
    plan = dict(stand_plan(q, 32), rule="safe_min_goal_distance", chosen_clearance_m=0.08)
    real_poll = node.pool.poll
    try:
        for blocked, expect in ((False, True), (True, False)):
            node.active = (dict(plan), 120, q, np.zeros(29))
            node.last_streamed_anchor = 128 + 10**6 if blocked else 120  # a stand blocks every later plan
            node.last_streamed_req = 104
            node.jobs = {7: (112, q, np.zeros(29), 112, False)}
            node.measured_jobs = {7: False}
            node.pool.poll = lambda: [(7, dict(plan), 0.1)]
            node.collect(128)
            assert (node.active[1] == 112) is expect
            assert node.measured_jobs == {}
    finally:
        node.pool.poll = real_poll
        node.jobs = {}


def test_an_older_predicted_plan_never_replaces_a_newer_measured_one(node):
    """inflight >= 2: J2 (requested at 108, predicted anchor 125) lands after J3 (measured, requested at 116)."""
    from g1deploy.guards import stand_plan

    q = FakeState()._q()
    plan = dict(stand_plan(q, 32), rule="safe_min_goal_distance", chosen_clearance_m=0.08)
    real_poll = node.pool.poll
    try:
        node.active = (dict(plan), 117, q, np.zeros(29))
        node.last_streamed_anchor, node.last_streamed_req = 117, 100
        node.jobs = {3: (116, q, np.zeros(29), 116, False), 2: (125, q, np.zeros(29), 108, False)}
        node.measured_jobs = {3: False}
        node.pool.poll = lambda: [(3, dict(plan), 0.1)]
        node.collect(128)
        assert node.active[1] == 116 and node.last_streamed_req == 116
        node.pool.poll = lambda: [(2, dict(plan), 0.1)]
        node.collect(130)
        assert node.active[1] == 116  # J2 is stale by request order
        assert node.last_streamed_req == 116  # never moves backwards
    finally:
        node.pool.poll = real_poll
        node.jobs = {}


def test_a_measured_plan_from_a_standstill_starts_when_it_lands(node):
    from g1deploy.guards import stand_plan

    q = FakeState()._q()
    plan = dict(stand_plan(q, 32), rule="safe_min_goal_distance", chosen_clearance_m=0.08)
    real_poll = node.pool.poll
    try:
        node.active = (dict(plan), 100, q, np.zeros(29))
        node.last_streamed_anchor, node.last_streamed_req = 100, 90
        node.jobs = {9: (112, q, np.zeros(29), 112, False)}
        node.measured_jobs = {9: True}  # requested while standing
        node.pool.poll = lambda: [(9, dict(plan), 0.3)]
        node.collect(127)
        assert node.active[1] == 127  # not 112: no 0.3 s jump into the plan
    finally:
        node.pool.poll = real_poll
        node.jobs = {}


def test_walking_drops_are_bounded(node):
    """Each drop costs a planner latency of stream: after max_drops in a row, or with little stream left, the job
    asks the guards for the calm path instead."""
    from g1deploy.guards import stand_plan

    q = FakeState()._q()
    node.history = [q.astype(np.float32) for _ in range(6)]
    node.active = (dict(stand_plan(q, 32), rule="safe_min_goal_distance"), 0, q, np.zeros(29))
    if not hasattr(node, "memory"):
        node.make_memory(q[:2])
    node.route_now = lambda xy, t: None
    sent = []
    real_submit = node.pool.submit
    node.pool.submit = lambda worker, job: sent.append(job)
    try:
        node.latency_est = 0.1
        for drops, ahead, expect in ((0, 200, True), (node.guards.max_drops, 200, False), (0, 30, False)):
            node.consecutive_drops, node.stream_end = drops, 100 + ahead
            node.request(100, None)
            assert sent[-1]["allow_drop"] is expect, (drops, ahead)
    finally:
        node.pool.submit = real_submit
        node.jobs.clear()
        node.measured_jobs.clear()
        node.anchor_checks.clear()


def test_a_walk_that_cannot_start_says_why(node):
    """No lidar at all: the node stays in PRESCAN and reports it, instead of a robot that silently stands."""
    st, cl = FakeState(), FakeCloud()
    cl.stop = True
    ev, _ = _run(node, st, cl, lambda e: "prescan_blocked" in e, timeout=10.0)
    assert "prescan_blocked" in ev and "run" not in ev


def test_a_calm_streak_keeps_the_robot_standing_for_the_guards(node):
    """After a calm plan with nothing safe, the next request is judged as from a standstill (no walking drops), with
    the streak's remaining travel, even if the robot moved; a safe plan releases it."""
    from g1deploy.guards import CalmLatch, stand_plan

    q = FakeState()._q()
    moved = q.copy()
    moved[0] += 0.4  # well past the standing speed over the 6-frame window
    node.history = [q.astype(np.float32)] * 3 + [moved.astype(np.float32)] * 3
    node.active = (dict(stand_plan(q, 32), rule="calm_stop", no_safe_candidate=True), 0, q, np.zeros(29))
    node.calm_latch = CalmLatch(node.guards)
    node.calm_latch.update({"rule": "calm_stand", "guard": {"no_safe": True}}, (0.0, 0.0))
    if not hasattr(node, "memory"):
        node.make_memory(q[:2])
    node.route_now = lambda xy, t: None
    sent = []
    real_submit = node.pool.submit
    node.pool.submit = lambda worker, job: sent.append(job)
    try:
        node.stream_end, node.consecutive_drops = 400, 0
        node.request(100, None)
        assert sent[-1]["standing"] is True and sent[-1]["allow_drop"] is False
        assert sent[-1]["calm_move_max_m"] == 0.0  # 0.4 m from where the streak began: settle only
    finally:
        node.pool.submit = real_submit
        node.jobs.clear()
        node.measured_jobs.clear()
        node.anchor_checks.clear()
        node.calm_latch = CalmLatch(node.guards)
