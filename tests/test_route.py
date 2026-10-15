"""CPU routing regressions using the actual frozen source, no policy or robot.

The recorded failing coordinates are public development-run evidence. The
all-free memory below is an explicit algorithm fixture, not a recovered scan.
"""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from g1deploy import bundle
from g1deploy.route_compat import install

START = np.array([3.7853962291291454, -0.5635527642023187])
GOAL = np.array([3.8247117178139156, -0.5738279709187756])


def source_route():
    path = bundle.ROOT / "flow/src/motion2scene_flow/closed_loop/route.py"
    spec = importlib.util.spec_from_file_location("route_regression_source", path)
    module = importlib.util.module_from_spec(spec)
    # Compile directly: never create bytecode in the frozen tree.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def memory():
    return SimpleNamespace(res=0.05, lo=np.array([-6.0, -6.0]), n=241,
                           support=np.zeros((241, 241)))


def test_recorded_same_cell_failure_and_repair():
    module = source_route()
    broken = module.SensorRoute(START, GOAL, memory())
    assert broken.cell(START) == broken.cell(GOAL) == (196, 109)
    assert broken.path.shape == (1, 2)
    with pytest.raises(ValueError, match="argmin"):
        broken.waypoint(START, 0.8)

    install(module)
    route = module.SensorRoute(START, GOAL, memory())
    np.testing.assert_array_equal(route.path, [START, GOAL])
    point, heading, remaining = route.waypoint(START, 0.8)
    np.testing.assert_allclose(point, GOAL)
    assert heading == pytest.approx(np.arctan2(*(GOAL - START)[::-1]))
    assert remaining == pytest.approx(0.040636037252520255)
    assert route.found and remaining > 0  # no fabricated arrival

    route.waypoint((START + GOAL) / 2, 0.8)
    progress = route.progress
    route.waypoint(START, 0.8)
    assert route.progress == progress  # backward motion cannot rewind a route
    route.waypoint(GOAL, 0.8, update=False)
    assert route.progress == progress


@pytest.mark.parametrize("sensor", [False, True])
@pytest.mark.parametrize("delta", [(0.0, 0.0), (0.01, -0.01)])
def test_same_cell_endpoint_queries_are_finite(sensor, delta):
    module = source_route()
    install(module)
    goal = START + delta
    route = (module.SensorRoute(START, goal, memory()) if sensor
             else module.Route(START, goal, []))
    assert route.cell(START) == route.cell(goal)
    np.testing.assert_array_equal(route.path, [START, goal])
    for xy in (START, goal, START - 0.1):
        point, heading, remaining = route.waypoint(xy, 0.8)
        assert np.isfinite([*point, heading, remaining]).all()
        np.testing.assert_allclose(point, goal)
    if not np.any(delta):
        assert remaining == 0


def test_sensor_refresh_keeps_period_and_resets_progress_at_goal():
    module = source_route()
    install(module)
    route = module.SensorRoute(START, GOAL, memory(), period_s=0.5)
    route.waypoint((START + GOAL) / 2, 0.8)
    assert route.progress > 0
    before = route.path.copy()
    assert not route.refresh(GOAL, 0.49)
    np.testing.assert_array_equal(route.path, before)
    assert route.refresh(GOAL, 0.5)
    assert route.last_refresh == 0.5 and route.progress == 0
    np.testing.assert_array_equal(route.path, [GOAL, GOAL])
    point, heading, remaining = route.waypoint(GOAL, 0.8)
    np.testing.assert_array_equal(point, GOAL)
    assert heading == remaining == 0


@pytest.mark.parametrize("boxes", [[], [dict(center_xy=[1.0, 0.0], half_xy=[0.1, 0.6],
                                           yaw=0.1, z_bottom=0.0, z_top=1.8)]])
def test_ordinary_route_matches_frozen_source_exactly(boxes):
    module = source_route()
    original = module.Route([0.0, 0.0], [2.0, 0.0], boxes)
    assert len(original.path) >= 2
    install(module)
    fixed = module.Route([0.0, 0.0], [2.0, 0.0], boxes)
    np.testing.assert_array_equal(fixed.path, original.path)
    np.testing.assert_array_equal(fixed.arc, original.arc)
    assert fixed.found == original.found
    for xy in ([0, 0], [0.9, -0.7], [0.8, -0.5], [2, 0]):
        a, b = fixed.waypoint(xy, 0.8), original.waypoint(xy, 0.8)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]


def test_install_is_idempotent_and_preserves_existing_class_references():
    module = source_route()
    existing_class = module.Route
    install(module)
    method = existing_class._plan
    install(module)
    assert module.Route is existing_class and existing_class._plan is method
    assert existing_class(START, START, []).waypoint(START, 0.8)[2] == 0


def test_deployment_loader_applies_fix_to_sensor_route_and_callback():
    flow = bundle.load_flow()
    route = flow.route.SensorRoute(START, GOAL, memory())
    assert route.waypoint(START, 0.8)[2] > 0
    assert flow.callback.Route is flow.route.Route
    assert flow.callback.Route(GOAL, GOAL, []).waypoint(GOAL, 0.8)[2] == 0


def test_near_goal_still_requires_low_speed_and_continuous_hold():
    from g1deploy.scoring import TaskScorer

    scorer = TaskScorer(dict(goal_world_xy=GOAL, goal_radius_m=0.3, goal_speed_m_s=0.25,
                            goal_hold_s=0.4, deadline_s=10.0, fall_height_m=0.35,
                            fall_tilt_rad=1.0, obstacle_contact_threshold_n=100.0))
    pos = np.r_[START, 0.79]

    def step(speed):
        return scorer.update(pos, [speed, 0, 0], physics_root_positions=[pos],
                             physics_root_quaternions=[[1, 0, 0, 0]], physics_contact_force_n=[0])

    for _ in range(25):
        assert step(0.26) is None  # close enough, but moving too fast
    for _ in range(19):
        assert step(0.0) is None
    assert step(0.26) is None and scorer.hold_steps == 0
    for _ in range(19):
        assert step(0.0) is None
    assert step(0.0) == "goal_completion"
