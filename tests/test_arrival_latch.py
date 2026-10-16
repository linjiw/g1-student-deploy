"""Observed-anchor arrival state, independently from autoregressive candidates."""

from dataclasses import dataclass
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from g1deploy import bundle
from g1deploy.goal_guard import update_arrival_latch
from sim.lockstep import make_goal_fn


def anchor(xy, yaw):
    return np.r_[xy, .8, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2), np.zeros(29)]


def test_candidate_queries_cannot_enter_arrival_before_actual_anchor():
    f = bundle.load_flow()
    route = f.route.Route([1, 0], [0, 0], [])
    goal = make_goal_fn(route, 1.5, "bearing_near", .5, None,
                        f.callback.goal_vector, f.callback.route_goal_heading,
                        arrival_radius_m=.3, arrival_scope="run-a")
    assert goal.set_anchor(anchor([1, 0], 0), None) is None
    # A sampled future crosses the tolerance, but cannot latch its own yaw.
    out = goal([.1, 0], 0).numpy()
    assert goal.diagnostics["arrival_heading_held"] == 0
    assert out[2] < -.99
    for xy, yaw in (([0, .2], 2.7), ([-.1, 0], -2.9), ([0, 0], 1.2)):
        goal(xy, yaw)
    assert goal.diagnostics["arrival_heading_held"] == 0


def test_tolerance_entry_hysteresis_goal_and_run_reset():
    fn = lambda state, xy, yaw, goal=(0, 0), scope="a": update_arrival_latch(
        state, anchor(xy, yaw), goal, .3, .75, scope, yaw - .1)
    assert fn(None, [.301, 0], .2) is None
    state = fn(None, [.3, 0], .2)
    assert state["heading_rad"] == pytest.approx(.1)
    assert state["entry_measured_yaw_rad"] == pytest.approx(.2)
    # Hold a fixed WORLD yaw while drifting just beyond position tolerance.
    assert fn(state, [.5, 0], 1.1) is state
    assert fn(state, [.751, 0], 1.1) is None
    assert fn(state, [.2, 0], -.4, scope="b")["heading_rad"] == pytest.approx(-.5)
    assert fn(state, [.2, 0], -.4, goal=[2, 0]) is None


def test_fresh_worker_goal_closures_share_only_explicit_planner_state():
    f = bundle.load_flow()
    route = f.route.Route([.2, 0], [0, 0], [])
    def build(scope):
        return make_goal_fn(route, 1.5, "bearing_near", .5, None,
                            f.callback.goal_vector, f.callback.route_goal_heading,
                            arrival_radius_m=.3, arrival_scope=scope)
    first, second = build(1), build(1)
    state = first.set_anchor(anchor([.2, 0], .4), None)
    second.set_anchor(anchor([.2, 0], .9), state)
    for yaw in (-2., .1, 1.7):
        output = second([.1, 0], yaw).numpy()
        np.testing.assert_allclose(output[2:], [np.cos(np.pi-yaw), np.sin(np.pi-yaw)], atol=1e-7)
    third = build(2)
    fresh = third.set_anchor(anchor([.2, 0], -.8), state)
    assert fresh["heading_rad"] == pytest.approx(np.pi)
    assert fresh["entry_measured_yaw_rad"] == pytest.approx(-.8)


def test_arrival_entry_preserves_requested_approach_turn_despite_execution_lag():
    # Actual corridor-intermediate job-a82988440555 anchor at t=4.8s.
    # This fixture rebuilds only a straight final segment; no simulation/model.
    f = bundle.load_flow()
    xy, destination, measured_yaw = [2.906034765757067, -.016377092397980388], [3.1, -.2], .25838052505376224
    route = f.route.Route(xy, destination, [])
    goal = make_goal_fn(route, 1.5, "bearing_near", .5, None,
                        f.callback.goal_vector, f.callback.route_goal_heading,
                        arrival_radius_m=.3, arrival_scope="recorded-fixture")
    original = goal(xy, measured_yaw).numpy()
    previous_heading = goal.diagnostics["first"]["heading_rad"]
    assert previous_heading == pytest.approx(-.7580145137218708)
    state = goal.set_anchor(anchor(xy, measured_yaw), None)
    goal.reset_diagnostics()
    np.testing.assert_array_equal(goal(xy, measured_yaw).numpy(), original)
    assert state["heading_rad"] == previous_heading
    assert state["heading_basis"] == "source_route_goal_at_observed_anchor"
    # Route refresh after crossing the point cannot flip the arrival heading.
    route.start = np.array([3.2, -.2])
    route.path = route._plan()
    route.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route.path, axis=0), axis=1))]
    assert goal.set_anchor(anchor([3.2, -.2], -.1), state) is state
    for candidate_yaw in (-2.7, .1, 2.8):
        result = goal([3.2, -.2], candidate_yaw).numpy()
        np.testing.assert_allclose(result[2:], [np.cos(previous_heading-candidate_yaw),
                                               np.sin(previous_heading-candidate_yaw)], atol=1e-7)


def test_cached_planner_resets_state_even_for_identical_repeated_episode(monkeypatch):
    from sim import lockstep
    from g1deploy import fastplan
    @dataclass
    class Config:
        seed: int = 0
        device: str = "cpu"
    p = SimpleNamespace(generator=SimpleNamespace(manual_seed=lambda seed: None))
    monkeypatch.setattr(lockstep, "_PLANNERS", {})
    monkeypatch.setattr(fastplan, "build_planner", lambda *args: p)
    assert lockstep.cached_planner(None, Config(), "eager") is p
    p._g1_arrival_heading = {"heading_rad": 1.2}
    assert lockstep.cached_planner(None, Config(), "eager") is p
    assert p._g1_arrival_heading is None


@pytest.mark.parametrize("radius", [-.1, float("nan"), float("inf")])
def test_invalid_arrival_tolerance_is_rejected(radius):
    f = bundle.load_flow()
    route = f.route.Route([0, 0], [2, 0], [])
    with pytest.raises(ValueError, match="arrival_radius_m"):
        make_goal_fn(route, 1.5, "bearing_near", .5, None, f.callback.goal_vector,
                     f.callback.route_goal_heading, arrival_radius_m=radius)


def test_node_navigation_restarts_and_goal_commands_reset_worker_scope(monkeypatch):
    from g1deploy import node as module
    monkeypatch.setattr(module, "emit", lambda *args, **kwargs: None)
    n = object.__new__(module.StudentNode)
    n.navigation_generation = 0
    n.streamer = None
    n.pool = SimpleNamespace(poll=lambda: [])
    n.reset_run_state()
    n.reset_run_state()
    assert n.navigation_generation == 2
    n.phase = "STOPPED"
    n.commands = [dict(cmd="goal", ahead=1, left=0), dict(cmd="go")]
    n.go_event = threading.Event()
    n.args = SimpleNamespace(prescan_s=.3)
    n.handle_commands(anchor([0, 0], 0), None)
    assert n.navigation_generation == 4 and n.phase == "PRESCAN"


def test_node_job_carries_observed_anchor_separately_from_predictive_history(monkeypatch):
    from g1deploy import node as module
    n = object.__new__(module.StudentNode)
    n.active, n.latency_est = ("existing-reference",), .2
    n.history = [anchor([0, 0], .3), anchor([.1, 0], .3)]
    predicted = [q.copy() for q in n.history]
    predicted[-1][0] += 1
    monkeypatch.setattr(module, "predicted_history", lambda *args: predicted)
    monkeypatch.setattr(module, "memory_view", lambda mem: {})
    n.route_now = lambda *args: "configured-route"
    n.mem_lock, n.memory = threading.Lock(), None
    n.next_id, n.navigation_generation, n.jobs = 0, 7, {}
    sent = []
    n.pool = SimpleNamespace(submit=lambda worker, job: sent.append(job))
    n.flow = SimpleNamespace(feed=SimpleNamespace(finite_difference_dq=lambda hist: np.zeros(29)))
    n.request(10, 0)
    assert sent[0]["navigation_generation"] == 7
    np.testing.assert_array_equal(sent[0]["arrival_anchor"], n.history[-1])
    np.testing.assert_array_equal(sent[0]["hist"][-1], predicted[-1])
    assert sent[0]["arrival_anchor"][0] != sent[0]["hist"][-1][0]
