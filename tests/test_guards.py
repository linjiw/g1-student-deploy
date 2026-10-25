"""Deploy-side guards (g1deploy/guards.py) on explicit fixtures: no model, no simulator, no evaluation packet."""

import json
import math

import numpy as np
import pytest
import torch

from g1deploy import bundle, contract as C, guards as G


def standing_q(xy=(0.0, 0.0), yaw=0.0, z=0.79):
    return np.r_[xy, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2), C.DEFAULT_MJ]


def shell_planner(ground_legs=False):
    """A FlowPlanner with the real kinematic tree, capsules, joint limits and selection margin; no model."""
    flow = bundle.load_flow()
    P = flow.planner
    p = object.__new__(P.FlowPlanner)
    p.cfg = P.PlannerConfig(run="unused-fixture", device="cpu", candidates=1, fallback="resample_turn",
                            select_margin_m=0.08, ground_legs=ground_legs)
    p.device = torch.device("cpu")
    p.calls = 0
    p.certifier = P.ChunkCertifier(list(P.JOINT_NAMES))
    p.shapes = P.collision_shapes(p.certifier)
    limits = json.loads((bundle.ROOT / "flow/artifacts/clean-reference-v2/receipt.json").read_text())["joint_limits"]
    p.limits = np.array([limits[n] for n in P.JOINT_NAMES], dtype=np.float32)
    return p


@pytest.fixture(scope="module")
def geometry_planner():
    return shell_planner()


def empty_view(n=121, res=0.05):
    lo = np.array([-3.0, -3.0])
    return G.MemoryView(np.full((n, n), np.nan), np.full((n, n), np.nan), lo, res, n, np.zeros((n, n), dtype=bool))


def cell(view, xy):
    return tuple(np.round((np.asarray(xy) - view.lo) / view.res).astype(int))


# ------------------------------------------------------------------------------------------ memory evidence
def test_cloud_memory_counts_far_evidence():
    from g1deploy.perception.cloud_memory import CloudMemory

    mem = CloudMemory((0.0, 0.0), self_filter=False, rng=np.random.default_rng(0))
    origin = np.array([0.0, 0.0, 1.2])
    shoulder = np.array([[0.05, -0.30, 1.0]] * 5)  # beside the sensor: near-only
    post_far = np.array([[1.5, 0.0, 0.5]] * 5)  # seen from 1.5 m
    mem.add_cloud(0.0, np.vstack([shoulder, post_far]), origin)
    mem.step(0.0)
    near = mem.near_only()
    ij = lambda xy: tuple(np.round((np.asarray(xy) - mem.lo) / mem.res).astype(int))  # noqa: E731
    push = lambda p: p[:2] + 0.035 * (p[:2] - origin[:2]) / np.linalg.norm(p[:2] - origin[:2])  # noqa: E731
    assert near[ij(push(shoulder[0]))]
    assert not near[ij(push(post_far[0]))]


@pytest.mark.parametrize("min_high, ceiling", [(1, True), (3, False)])
def test_min_high_points_gates_overhead_cells(min_high, ceiling):
    """One high return over free space is a ceiling at the frozen default (1), not at min_high_points=3."""
    from g1deploy.perception.cloud_memory import CloudMemory

    mem = CloudMemory((0.0, 0.0), self_filter=False, rng=np.random.default_rng(0), min_high_points=min_high)
    origin = np.array([0.0, 0.0, 1.2])
    ground = np.c_[np.full(20, 2.0), np.linspace(-0.02, 0.02, 20), np.zeros(20)]  # rays pass x=1 at z=0.6
    top = np.array([[1.0, 0.0, 1.0]])
    mem.add_cloud(0.0, np.vstack([ground, top]), origin)
    mem.step(0.0)
    c = tuple(np.round((np.array([1.0 + 0.035 * 0.98, 0.0]) - mem.lo) / mem.res).astype(int))
    assert bool(np.isfinite(mem.ceiling[c])) is ceiling


# ------------------------------------------------------------------------------------------ self mask
def test_self_mask_hides_only_leaks_hugging_the_body(geometry_planner):
    v = empty_view()
    v.support[:] = 0.0
    q = standing_q()
    band = G.body_cells(geometry_planner, q, v, G.DEPLOY.self_mask_band_m)
    leak = tuple(np.argwhere(band)[len(np.argwhere(band)) // 3])  # some near-only cell hugging the body
    outside = cell(v, (0.0, -0.60))  # near-only too, but 0.6 m away
    real = tuple(np.argwhere(band)[len(np.argwhere(band)) // 2])  # as close, but once seen from far: a real post
    for c in (leak, outside, real):
        v.support[c] = 1.06
    v.near_only[leak] = v.near_only[outside] = True
    masked, info = G.self_mask(v, geometry_planner, q, G.DEPLOY)
    assert info == {"self_masked_cells": 1}
    assert masked.support[leak] == 0.0  # the robot stands there
    assert masked.support[outside] == 1.06 and masked.support[real] == 1.06
    assert v.support[leak] == 1.06  # the view passed in is not modified


def test_self_mask_keeps_a_wall_beside_the_robot(geometry_planner):
    """A wall 0.40 m from the pelvis reaches into the band, but it continues outside it: a real object, kept - even
    if every wall cell were only ever seen up close."""
    v = empty_view()
    v.support[:] = 0.0
    q = standing_q()
    band = G.body_cells(geometry_planner, q, v, G.DEPLOY.self_mask_band_m)
    wall = [cell(v, (x, 0.40)) for x in np.arange(-1.0, 1.01, 0.05)]
    assert any(band[c] for c in wall)  # the band does reach it
    for c in wall:
        v.support[c] = 1.2
        v.near_only[c] = True
    masked, info = G.self_mask(v, geometry_planner, q, G.DEPLOY)
    assert masked is v and info == {}


def test_self_mask_refuses_an_object_hugging_the_robot(geometry_planner):
    v = empty_view()
    v.support[:] = 0.0
    q = standing_q()
    band = G.body_cells(geometry_planner, q, v, G.DEPLOY.self_mask_band_m)
    ring = band & ~G.body_cells(geometry_planner, q, v, 0.0)  # cells hugging the body, all inside the band
    assert ring.sum() > G.DEPLOY.self_mask_max_cells
    v.support[ring] = 0.9
    v.near_only[ring] = True
    masked, info = G.self_mask(v, geometry_planner, q, G.DEPLOY)
    assert masked is v and info == {"self_mask_refused": int(ring.sum())}


def test_self_mask_ignores_evidence_the_planner_does_not_see(geometry_planner):
    """Sub-threshold near-only returns (no support, no ceiling) neither get hidden nor count toward the cap."""
    v = empty_view()
    v.support[:] = 0.0
    q = standing_q()
    band = G.body_cells(geometry_planner, q, v, G.DEPLOY.self_mask_band_m)
    v.near_only[band] = True  # a ring of 1-2 stray returns per cell, below every threshold
    leak = tuple(np.argwhere(band)[0])
    v.support[leak] = 1.06
    masked, info = G.self_mask(v, geometry_planner, q, G.DEPLOY)
    assert info == {"self_masked_cells": 1} and masked.support[leak] == 0.0


def test_standing_means_no_turning_either():
    still = [standing_q() for _ in range(6)]
    turning = [standing_q(yaw=0.02 * k) for k in range(6)]  # 1 rad/s in place
    walking = [standing_q(xy=(0.02 * k, 0.0)) for k in range(6)]
    assert G.is_standing(still, G.DEPLOY)
    assert not G.is_standing(turning, G.DEPLOY) and not G.is_standing(walking, G.DEPLOY)


def test_walking_drop_needs_the_budget():
    p = shell_planner()
    out = frozen_plan(p, [candidate_batch([(1.2, 0.0), (0.9, 0.0)], z=0.55 + 0.075)], [[0.08, 0.08]])
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=False, allow_drop=False)
    assert not got.get("rejected") and got["rule"] in G.CALM_RULES  # out of drops: the calm path, never a gap


def test_overhead_walls():
    v = empty_view()
    v.ceiling[cell(v, (1.0, 0.0))] = 0.95  # a table top
    v.ceiling[cell(v, (1.0, 1.0))] = 2.0  # a high ceiling stays
    w, n = G.overhead_walls(v, G.PRESETS["walls"])
    assert n == 1 and w.support[cell(v, (1.0, 0.0))] == 1.5 and np.isnan(w.ceiling[cell(v, (1.0, 0.0))])
    assert w.ceiling[cell(v, (1.0, 1.0))] == 2.0


def test_ceiling_waiver_is_a_disc():
    v = empty_view()
    v.ceiling[cell(v, (1.9, 1.9))] = 1.1  # 2.7 m away on the diagonal: inside the old square window
    assert not G.ceiling_near(v, (0.0, 0.0), G.DEPLOY)
    v.ceiling[cell(v, (1.4, 1.4))] = 1.1  # 1.98 m
    assert G.ceiling_near(v, (0.0, 0.0), G.DEPLOY)


# ------------------------------------------------------------------------------------------ guidance boxes
def _inside(box, xy):
    return all(abs(xy[k] - box["center_xy"][k]) <= box["half_xy"][k] for k in range(2))


def test_rectangles_cover_exactly():
    rng = np.random.default_rng(3)
    m = rng.random((17, 23)) < 0.4
    cover = np.zeros_like(m, dtype=int)
    for i0, i1, j0, j1 in G._rectangles(m):
        cover[i0:i1 + 1, j0:j1 + 1] += 1
    np.testing.assert_array_equal(cover, m.astype(int))
    for i0, i1, j0, j1 in G._merge(G._rectangles(m), m, 0.8):
        assert m[i0:i1 + 1, j0:j1 + 1].mean() >= 0.8 - 1e-9


def test_closed_room_is_walls_not_a_block():
    sm = bundle.load_flow().sensor_map
    v = empty_view()
    v.support[:] = 0.0
    for x in np.arange(-2.0, 2.01, 0.05):
        for y in (-2.0, 2.0):
            v.support[cell(v, (x, y))] = 1.2
            v.support[cell(v, (y, x))] = 1.2
    frozen = sm.memory_boxes(v)
    assert len(frozen) == 1 and _inside(frozen[0], (0.0, 0.0))  # the finding: one block around the robot
    boxes = G.split_boxes(v, (0.0, 0.0), G.DEPLOY)
    assert 4 <= len(boxes) <= 8
    assert not any(_inside(b, (0.0, 0.0)) for b in boxes)
    assert not any(_inside(b, (1.0, 1.0)) for b in boxes)  # free floor inside the room is never covered
    occupied = np.argwhere(v.support > 0.05) * v.res + v.lo
    assert all(any(_inside(b, p) for b in boxes) for p in occupied)


def test_l_shape_is_two_boxes_and_tops_are_kept():
    v = empty_view()
    v.support[:] = 0.0
    for x in np.arange(-1.0, 1.51, 0.05):
        v.support[cell(v, (x, -0.7))] = 0.9
    for y in np.arange(-0.7, 1.51, 0.05):
        v.support[cell(v, (-1.0, y))] = 0.9
    boxes = G.split_boxes(v, (0.0, 0.0), G.DEPLOY)
    assert len(boxes) == 2 and not any(_inside(b, (0.0, 0.0)) for b in boxes)
    assert all(b["z_top"] == pytest.approx(0.9) for b in boxes)


def test_robot_in_a_notch_of_a_big_block_is_never_boxed():
    """A 2 x 2 m block with a 0.5 x 0.5 m notch is > 80 % full: one bounding box would contain the robot."""
    v = empty_view()
    v.support[:] = 0.0
    for x in np.arange(-1.0, 1.01, 0.05):
        for y in np.arange(-1.0, 1.01, 0.05):
            if not (-0.25 <= x <= 0.25 and -0.25 <= y <= 1.01):
                v.support[cell(v, (x, y))] = 0.8
    boxes = G.split_boxes(v, (0.0, 0.0), G.DEPLOY)
    assert boxes and not any(_inside(b, (0.0, 0.0)) for b in boxes)


def test_box_cap_keeps_the_nearest():
    v = empty_view()
    v.support[:] = 0.0
    for k in range(60):
        v.support[cell(v, (-2.5 + 0.2 * (k % 25), -2.5 + 0.4 * (k // 25)))] = 0.5
        v.support[cell(v, (-2.5 + 0.2 * (k % 25) + 0.05, -2.5 + 0.4 * (k // 25)))] = 0.5
    boxes = G.split_boxes(v, (0.0, -2.5), G.DEPLOY)
    assert len(boxes) == G.DEPLOY.max_guide_boxes
    d = [np.linalg.norm(np.asarray(b["center_xy"]) - [0.0, -2.5]) for b in boxes]
    assert max(d) < 2.5


# ------------------------------------------------------------------------------------------ selection
def candidate_batch(spec, z=0.875, knees=None):
    """spec: [(end_x, end_y), ...]; 8 frames, SONIC default joints, optional right-knee angle per candidate."""
    k = len(spec)
    q = np.zeros((k, 8, 36))
    for i, (x, y) in enumerate(spec):
        q[i, :, 0] = x * np.arange(1, 9) / 8
        q[i, :, 1] = y * np.arange(1, 9) / 8
    q[:, :, 2] = z
    q[:, :, 3] = 1
    q[:, :, 7:] = C.DEFAULT_MJ
    for i, kn in enumerate(knees or []):
        if kn is not None:
            q[i, :, 7 + G.KNEES[1]] = kn
    states = torch.zeros(k, 8, 65)
    states[:, :, 36:65] = torch.arange(29) / 1000
    return dict(qpos=q, rot=torch.eye(3).repeat(k, 8, 1, 1), states=states, end_xy=q[:, -1, :2],
                goal0=np.array([1.0, 0.0, 1.0, 0.0]))


def frozen_plan(planner, batches, clearances):
    """Run the actual frozen FlowPlanner.plan (with the deploy compat layer) on explicit candidate batches."""
    batches, values = iter(batches), iter([np.asarray(c, dtype=float) for c in clearances])
    planner.encode_history = lambda hist: (torch.zeros(2, 65), torch.zeros(2), torch.tensor(0.))
    planner._candidates = lambda *args: next(batches)
    planner._clearance = lambda *args: next(values)
    return type(planner).plan(planner, np.zeros((2, 36)), [], lambda xy, yaw: None)


def test_materialize_equals_the_frozen_planner_with_leg_grounding():
    p = shell_planner(ground_legs=True)
    batch = candidate_batch([(1.0, 0.0), (0.8, 0.3)], z=0.875 - 0.12)  # pelvis low enough to need grounding
    out = frozen_plan(p, [batch], [[0.08, 0.08]])
    assert out["ground_lift_max_m"] > 0.01  # grounding did move the legs
    for i in (0, 1):
        got = G.materialize(p, p.last_pools, i)
        if i == out["choice"]:
            for k in ("qpos", "rotation", "joint_velocity", "states"):
                np.testing.assert_array_equal(got[k], out[k])
    assert got["candidate_pool"] == "go"


def test_safe_choice_inside_the_envelope_is_unchanged():
    p = shell_planner(ground_legs=True)
    out = frozen_plan(p, [candidate_batch([(1.0, 0.0), (0.8, 0.3)])], [[0.08, 0.08]])
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=True)
    assert got["choice"] == out["choice"] and got["rule"] == out["rule"] == "safe_min_goal_distance"
    for k in ("qpos", "rotation", "joint_velocity", "states"):
        np.testing.assert_array_equal(got[k], out[k])


def test_lunge_outside_the_envelope_is_replaced_by_a_trackable_safe_candidate():
    p = shell_planner()
    out = frozen_plan(p, [candidate_batch([(1.0, 0.0), (0.9, 0.3)], knees=[2.1, None])], [[0.08, 0.08]])
    assert out["choice"] == 0
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=True)
    assert got["choice"] == 1 and got["rule"] == "safe_min_goal_distance+envelope"
    assert got["guard"]["action"] == "envelope_reselect" and "knee 2.10" in got["guard"]["rejected"]


def no_safe(p, go, stop, stop_knees=None):
    """Frozen no-safe path: K go, 3K resampled go, K stop candidates (resample_turn)."""
    batches = [candidate_batch([g[:2] for g in go]), candidate_batch([(0.1, 0.0)] * 3),
               candidate_batch([s[:2] for s in stop], knees=stop_knees)]
    return frozen_plan(p, batches, [[c for *_, c in go], [-1, -1, -1], [c for *_, c in stop]])


def test_no_safe_from_standstill_never_streams_a_go_candidate():
    p = shell_planner()
    out = no_safe(p, [(1.2, 0.8, 0.04)], [(0.05, 0.0, 0.0), (0.6, 0.0, 0.0)])
    assert out["rule"] == "fallback_go_max_clearance" and out["candidate_pool"] == "go"  # the frozen lunge
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=True)
    assert got["rule"] == "calm_stop" and got["candidate_pool"] == "stop"
    assert np.linalg.norm(got["qpos"][-1, :2]) < G.DEPLOY.calm_turn_max_m  # the 0.6 m stop candidate is too far


def test_calm_stop_is_never_predicted_in_contact_and_never_worse_than_standing():
    p = shell_planner()
    out = no_safe(p, [(1.2, 0.8, 0.04)], [(0.05, 0.0, -1.0)])
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=True)
    assert got["rule"] == "calm_stand"  # a stop predicted in contact (-1) loses to standing still
    out = no_safe(p, [(1.2, 0.8, 0.04)], [(0.05, 0.0, 0.0)])
    clear = lambda q, dz: np.full(len(q), 0.08)  # noqa: E731  standing still is clear of everything
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=True, clearance_fn=clear)
    assert got["rule"] == "calm_stand" and got["guard"]["stand_clearance_m"] == 0.08


def test_walking_keeps_a_frozen_fallback_inside_the_envelope():
    p = shell_planner()
    out = no_safe(p, [(1.2, 0.3, 0.04)], [(0.05, 0.0, 0.0)])
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=False)
    assert got["rule"] == out["rule"] and got["choice"] == out["choice"]


def test_no_trackable_stop_settles_to_stand():
    p = shell_planner()
    out = no_safe(p, [(1.2, 0.8, 0.04)], [(0.05, 0.0, 0.0)], stop_knees=[2.2])
    crouched = standing_q()
    crouched[7 + G.KNEES[0]] = 1.2
    got = G.guard_selection(p, out, [crouched], empty_view(), G.DEPLOY, standing=True)
    assert got["rule"] == "calm_stand" and got["candidate_pool"] == "stand"
    assert got["qpos"][-1, 7 + G.KNEES[0]] < 1.2  # moving toward the default pose
    np.testing.assert_allclose(got["qpos"][:, :2], 0.0)
    assert np.isfinite(got["joint_velocity"]).all()


def test_envelope_is_waived_under_a_ceiling():
    p = shell_planner()
    v = empty_view()
    v.ceiling[cell(v, (1.0, 0.0))] = 1.1  # a beam ahead: ducking is the point
    out = frozen_plan(p, [candidate_batch([(1.2, 0.0), (0.9, 0.0)], z=0.55 + 0.075)], [[0.08, 0.08]])
    got = G.guard_selection(p, out, [standing_q()], v, G.DEPLOY, standing=False)
    assert got["choice"] == out["choice"] and not got["guard"]["envelope"]
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=False)
    # open floor: a 0.55 m pelvis crouch is outside the envelope; walking, it is dropped (the current plan plays on)
    assert got["guard"]["action"] == "envelope_drop" and "pelvis 0.55" in got["rejected"]
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=True)
    assert got["rule"] in G.CALM_RULES and not got.get("rejected")  # from a standstill: calm


def test_plan_guarded_gives_the_planner_the_masked_view(geometry_planner):
    """The self mask reaches the map input, the selector and the guidance boxes; the memory keeps the leak."""
    v = empty_view()
    v.support[:] = 0.0
    q = standing_q()
    leak = tuple(np.argwhere(G.body_cells(geometry_planner, q, v, 0.05))[0])
    v.support[leak] = 1.06
    v.near_only[leak] = True
    seen = {}

    def plan(hist, boxes, goal_fn, map_fn, clearance_fn, prev=None, guide_boxes=None):
        seen.update(sel=clearance_fn, guide=guide_boxes)
        raise StopIteration

    import dataclasses

    cfg = geometry_planner.cfg
    geometry_planner.plan = plan
    geometry_planner.cfg = dataclasses.replace(cfg, guide_w=1.0)
    try:
        with pytest.raises(StopIteration):
            G.plan_guarded(geometry_planner, [q, q], [], None, v, G.DEPLOY, measured_q=q, standing=True)
    finally:
        del geometry_planner.plan
        geometry_planner.cfg = cfg
    assert seen["sel"].memory.support[leak] == 0.0 and seen["guide"] == []
    assert v.support[leak] == 1.06


def test_stand_plan():
    q = standing_q(yaw=0.7)
    q[3:7] = [0.97, 0.1, 0.05, 0.2]
    q[3:7] /= np.linalg.norm(q[3:7])
    q[7:] += 0.3
    p = G.stand_plan(q, 80)
    np.testing.assert_allclose(p["qpos"][-1, 7:], C.DEFAULT_MJ)
    np.testing.assert_allclose(p["rotation"][:, 2, 2], 1.0)  # level
    assert np.isfinite(p["joint_velocity"]).all() and np.abs(p["joint_velocity"]).max() < 30


def test_presets_round_trip():
    assert not G.preset("off").any() and G.preset("deploy").calm and G.preset("walls").overhead_walls
    assert G.preset({"preset": "deploy", "max_knee_rad": 1.5}).max_knee_rad == 1.5
    json.dumps(G.DEPLOY.describe())


def test_calm_latch_holds_a_no_safe_streak_until_something_is_safe():
    latch = G.CalmLatch(G.DEPLOY)
    assert not latch.active and latch.budget((0, 0)) is None
    walking_fallback = {"rule": "stop_max_clearance", "no_safe_candidate": True}
    assert latch.update(walking_fallback, (0, 0)) is None and not latch.active  # a walking fallback does not latch
    calm = {"rule": "calm_stand", "guard": {"no_safe": True}}
    assert latch.update(calm, (0, 0)) == "hold" and latch.active
    assert latch.budget((0.1, 0.0)) == pytest.approx(G.DEPLOY.calm_turn_max_m - 0.1)
    assert latch.budget((0.4, 0.0)) == 0.0
    assert latch.update(walking_fallback, (0.1, 0)) is None and latch.active  # still nothing safe: still calm
    assert latch.update({"rule": "safe_min_goal_distance", "no_safe_candidate": False}, (0.1, 0)) == "release"
    assert not latch.active


def test_calm_stops_share_one_travel_budget():
    p = shell_planner()
    out = no_safe(p, [(1.2, 0.8, 0.04)], [(0.15, 0.0, 0.0)])
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=True, calm_move_max_m=0.2)
    assert got["rule"] == "calm_stop"
    got = G.guard_selection(p, out, [standing_q()], empty_view(), G.DEPLOY, standing=True, calm_move_max_m=0.1)
    assert got["rule"] == "calm_stand"  # the streak already used its travel: settle, do not back off further


def test_walls_route_sees_low_overheads_as_walls():
    v = empty_view()
    v.support[:] = 0.0
    v.ceiling[cell(v, (1.5, 0.0))] = 1.17  # a door beam
    v.ceiling[cell(v, (1.5, 1.0))] = 2.0  # a high ceiling
    assert G.route_memory(v, G.DEPLOY) is v  # deploy: the route ducks under overheads, as frozen
    r = G.route_memory(v, G.PRESETS["walls"])
    assert r.support[cell(v, (1.5, 0.0))] == 1.5 and r.support[cell(v, (1.5, 1.0))] == 0.0
    assert r.res == v.res and r.n == v.n
    v.ceiling[cell(v, (0.5, 0.0))] = 0.95  # read live: a table seen later
    assert r.support[cell(v, (0.5, 0.0))] == 1.5
