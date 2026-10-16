"""Route-grid behavior fixtures, plus the shared node/rehearsal goal function."""

from types import SimpleNamespace

import numpy as np
import pytest

from g1deploy import bundle
from g1deploy.goal_guard import limit_goal
from sim.lockstep import make_goal_fn


def grid():
    return SimpleNamespace(res=0.05, lo=np.array([-1., -1.]), shape=(81, 81),
                           occupied=np.zeros((81, 81), dtype=bool))


def test_free_ray_preserves_target_exactly():
    target = np.array([1.3556067943572998, -0.6421295404434204])
    got, status = limit_goal(grid(), [0, 0], target)
    np.testing.assert_array_equal(got, target)
    assert status == "grid_line_free"


def test_blocked_pursuit_stops_before_first_cell_and_cannot_jump_past_it():
    route = grid()
    route.occupied[40, :] = True  # x=1 m; a thin wall then free space
    got, status = limit_goal(route, [0, 0], [1.5, 0])
    assert status == "limited_at_occupied_grid" and 0.9 <= got[0] < 0.975
    n = int(np.ceil(np.linalg.norm(got) / (route.res / 2))) + 1
    ij = np.round((np.linspace([0, 0], got, n) - route.lo) / route.res).astype(int)
    assert not route.occupied[ij[:, 0], ij[:, 1]].any()


def test_occupied_start_is_explicitly_uncertified_and_allows_existing_escape():
    route = grid()
    route.occupied[20, 20] = True
    got, status = limit_goal(route, [0, 0], [0.5, 0])
    assert status == "occupied_start_uncertified"
    np.testing.assert_array_equal(got, [0.5, 0])


def test_zero_distance_and_out_of_grid_are_finite():
    route = grid()
    got, status = limit_goal(route, [0, 0], [0, 0])
    np.testing.assert_array_equal(got, [0, 0])
    assert status == "grid_line_free"
    got, status = limit_goal(route, [0, 0], [5, 0])
    assert status == "limited_at_occupied_grid" and got[0] <= 3.025


def test_shared_goal_builder_retains_4vector_frame_convention_and_diagnostics():
    flow = bundle.load_flow()
    route = flow.route.Route([0, 0], [2, 0], [])
    goal = make_goal_fn(route, 1.5, "bearing_near", .5, None,
                        flow.callback.goal_vector, flow.callback.route_goal_heading)
    np.testing.assert_allclose(goal([0, 0], np.pi / 2).numpy(), [0, -1.5, 0, -1], atol=1e-7)
    assert goal.diagnostics["first"]["status"] == "grid_line_free"
    assert goal.diagnostics["queries"] == 1
    goal.reset_diagnostics()
    assert goal.diagnostics["queries"] == 0 and goal.diagnostics["first"] is None


def test_fixture_route_bend_does_not_send_goal_through_occupied_ray():
    flow = bundle.load_flow()
    # The real failure uses the same extrapolation at a bend; this explicit
    # fixture isolates it without pretending to contain a full captured scan.
    route = flow.route.Route([0, 0], [2, 0], [])
    route.occupied[route.cell([1, 0])] = True
    goal = make_goal_fn(route, 1.5, "bearing_near", .5, None,
                        flow.callback.goal_vector, flow.callback.route_goal_heading)
    out = goal([0, 0], 0).numpy()
    assert 0.9 <= out[0] < 0.975 and out[1] == 0
    assert goal.diagnostics["redirected"] == 1
    assert goal.diagnostics["first"]["requested_xy"] == [1.5, 0.0]
    assert route.free_line(np.array([0, 0]), out[:2])


def test_visible_route_redirect_advances_around_wall_within_existing_lookahead():
    flow = bundle.load_flow()
    boxes = [dict(center_xy=[1., 0.], half_xy=[0.15, 0.4], yaw=0., z_bottom=0., z_top=1.9)]
    route = flow.route.Route([0, 0], [2, 0], boxes)
    xy = np.array([0., 0.])
    before, _ = limit_goal(route, xy, [1.5, 0])
    goal, status = limit_goal(route, xy, [1.5, 0], lookahead=1.5)
    assert status == "redirected_to_visible_route"
    assert abs(goal[1]) > 0.2  # follows the actual detour, not wall-facing creep
    assert np.linalg.norm(goal - xy) > np.linalg.norm(before - xy)
    assert route.free_line(xy, goal)
    assert 0 <= route.project(goal) <= 1.5 + 1e-8
    assert route.progress == 0  # candidate queries cannot advance measured progress


def test_zero_length_route_and_occupied_start_do_not_invent_progress():
    flow = bundle.load_flow()
    route = flow.route.Route([0, 0], [0, 0], [])
    goal, status = limit_goal(route, [0, 0], [0, 0], lookahead=1.5)
    np.testing.assert_array_equal(goal, [0, 0])
    assert status == "grid_line_free" and route.progress == 0
    route.occupied[route.cell([0, 0])] = True
    goal, status = limit_goal(route, [0, 0], [.5, 0], lookahead=1.5)
    np.testing.assert_array_equal(goal, [.5, 0])
    assert status == "occupied_start_uncertified" and route.progress == 0


@pytest.mark.parametrize("xy", [[.2, 0], [-.2, 0], [0, .2], [0, -.2]])
@pytest.mark.parametrize("yaw", [-3.13, .9, 3.13])
def test_position_only_arrival_keeps_candidate_yaw_on_every_side(xy, yaw):
    flow = bundle.load_flow()
    mem = SimpleNamespace(support=np.zeros((241, 241)), n=241, lo=np.array([-6., -6.]), res=.05)
    route = flow.route.SensorRoute(xy, [0, 0], mem)
    goal = make_goal_fn(route, 1.5, "bearing_near", .5, None,
                        flow.callback.goal_vector, flow.callback.route_goal_heading)
    expected = flow.callback.goal_vector([0, 0], yaw, np.array(xy), yaw).numpy()
    np.testing.assert_array_equal(goal(xy, yaw).numpy(), expected)
    assert goal.diagnostics["arrival_heading_held"] == 1
    assert goal.diagnostics["first"]["heading_policy"] == "hold_candidate_yaw"
    # Refresh can flip the route's final segment; each predicted candidate yaw
    # remains its own reference, including across quaternion/yaw wrap.
    other = -np.array(xy)
    route.refresh(other, .2, force=True)
    np.testing.assert_array_equal(goal(other, -yaw).numpy()[2:], [1, 0])


def test_arrival_heading_condition_preserves_non_near_and_explicit_path_modes():
    flow = bundle.load_flow()
    route = flow.route.Route([0, 0], [2, 0], [])
    goal = make_goal_fn(route, 1.5, "bearing_near", .5, None,
                        flow.callback.goal_vector, flow.callback.route_goal_heading)
    np.testing.assert_allclose(goal([0, 0], .9).numpy(),
                               flow.callback.goal_vector([1.5, 0], 0., np.array([0, 0]), .9).numpy())
    assert goal.diagnostics["arrival_heading_held"] == 0
    near = flow.route.Route([.2, 0], [0, 0], [])
    explicit = make_goal_fn(near, 1.5, "path", .5, None,
                            flow.callback.goal_vector, flow.callback.route_goal_heading)
    np.testing.assert_allclose(explicit([.2, 0], 0).numpy()[2:], [-1, 0], atol=1e-7)
    assert explicit.diagnostics["arrival_heading_held"] == 0


def test_explicit_path_heading_survives_visible_route_redirection():
    flow = bundle.load_flow()
    boxes = [dict(center_xy=[1., 0.], half_xy=[.15, .4], yaw=0., z_bottom=0., z_top=1.9)]
    route = flow.route.Route([0, 0], [2, 0], boxes)
    _, heading, _ = route.waypoint([0, 0], 2., update=False)
    goal = make_goal_fn(route, 2., "path", .5, None,
                        flow.callback.goal_vector, flow.callback.route_goal_heading)
    output = goal([0, 0], 0).numpy()
    assert goal.diagnostics["redirected"] == 1
    np.testing.assert_allclose(output[2:], [np.cos(heading), np.sin(heading)], atol=1e-7)
