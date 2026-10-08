"""The fast planner computes what the evaluated planner computes (g1deploy/fastplan.py).

CPU geometry (FK, height-field selection, leg grounding) must be bitwise equal; the GPU sampler must pick
the same candidate by the same rule, with joint differences at float32 noise (the eager planner itself
differs run to run by ~1e-5: index_select's backward accumulates with atomics)."""

import json
from pathlib import Path

import numpy as np
import pytest

from g1deploy import bundle

EP = Path("/home/linjiw/flow/artifacts/closed-loop-test-v4/student-guide10m8-grow06__beams2-101__s8871/callback")
pytestmark = pytest.mark.skipif(not EP.exists(), reason="recorded episode not available")


def _scene(tick=150):
    """Planner inputs at `tick` of a recorded two-beam episode (memory has seen both beams)."""
    from sim.lockstep import make_goal_fn

    flow = bundle.load_flow()
    cfg = bundle.student_config()
    task = json.loads((EP / "task.json").read_text())
    pre = np.load(EP / "closed_loop.npz")["pre_action_qpos"]
    sm = flow.sensor_map
    pc = flow.planner.PlannerConfig(**dict(cfg["planner"], device="cpu"))
    mem = sm.RayCastMemory(task["scene_boxes_world"], pre[0][:2], rng=np.random.default_rng(0), **pc.sensor)
    for k in range(tick + 1):
        mem.step(k * 0.02, pre[k][:3], pre[k][3:7])
    route = flow.route.Route(pre[0][:2], np.asarray(task["goal_world_xy"]), task["scene_boxes_world"])
    goal_fn = make_goal_fn(route, 1.5, "bearing_near", 0.5, None, flow.callback.goal_vector, flow.callback.route_goal_heading)
    return flow, cfg, mem, pre[tick - 5:tick + 1].astype(np.float32), goal_fn


def _candidates_qpos(flow, cfg, mem, hist, goal_fn):
    import torch

    pc = flow.planner.PlannerConfig(**dict(cfg["planner"], device="cpu", seed=0))
    planner = flow.planner.FlowPlanner(pc)
    with torch.no_grad():
        h, xy0, yaw0 = planner.encode_history(hist)
        cand = planner._candidates(h, xy0, yaw0, goal_fn, 16, torch.zeros(0, 9), None, None)
    return planner, cand["qpos"]


def test_fk_selection_and_grounding_bitwise_equal():
    from g1deploy import fastplan

    flow, cfg, mem, hist, goal_fn = _scene()
    from motion2scene_flow.closed_loop.grounding import ground_legs as ground_ref
    planner, qpos = _candidates_qpos(flow, cfg, mem, hist, goal_fn)
    tree = planner.certifier.tree
    q = np.asarray(qpos[0], dtype=np.float64)
    for a, b in zip(tree.forward(q), fastplan.fast_tree(tree).forward(q)):
        assert np.array_equal(a, b)
    sel = flow.sensor_map.HeightFieldSelector(mem, tree, planner.certifier.bodies, planner.shapes)
    for dz in (0.075, 0.0, 0.2):  # 0.2: candidates sink into the floor -> failures at every margin
        ref = sel(qpos, dz)
        assert np.array_equal(ref, fastplan.height_field_clearance(sel, qpos, dz)), dz
    shifted = qpos.copy()
    shifted[..., 0] += 0.6  # into the first beam: mixed pass/fail across margins
    assert np.array_equal(sel(shifted, 0.075), fastplan.height_field_clearance(sel, shifted, 0.075))
    low = np.array(qpos[2], dtype=np.float64)
    low[:, 2] -= 0.075 + 0.06  # forces the IK branch
    a, la = ground_ref(low, tree, planner.certifier.bodies, 0.035, planner.limits)
    b, lb = fastplan.ground_legs(low, tree, planner.certifier.bodies, 0.035, planner.limits)
    assert la.max() > 0.01 and np.array_equal(a, b) and np.array_equal(la, lb)


@pytest.mark.parametrize("impl", ["graph", "fused"])
def test_plans_match_the_evaluated_planner(impl):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")
    from g1deploy.fastplan import fast_planner_class
    from g1deploy.node import memory_view

    flow, cfg, mem, hist, goal_fn = _scene()
    sm = flow.sensor_map
    pc = flow.planner.PlannerConfig(**dict(cfg["planner"], device="cuda", seed=3))
    ref = flow.planner.FlowPlanner(pc)
    fast = fast_planner_class()(pc, fuse=impl == "fused")
    fast.warmup(ks=(16,), buckets=(8,))
    view = memory_view(mem)
    gb = sm.memory_boxes(view, grow_overhead_m=pc.guide_grow_overhead_m)
    assert len(gb) > 0  # guidance active

    def run(p):
        sel = sm.HeightFieldSelector(view, p.certifier.tree, p.certifier.bodies, p.shapes)
        return p.plan(hist, [], goal_fn, lambda xy, yaw: view.ego_map(xy, yaw, pc.datum_offset_m, 0.8)[0], sel, guide_boxes=gb)

    for _ in range(3):
        fast.generator.set_state(ref.generator.get_state())
        a, b = run(ref), run(fast)
        assert (a["choice"], a["rule"]) == (b["choice"], b["rule"])
        assert np.array_equal(a["clearance_m"], b["clearance_m"])
        assert np.abs(a["qpos"] - b["qpos"]).max() < 1e-3
        assert np.abs(a["joint_velocity"] - b["joint_velocity"]).max() < 0.1


def test_fused_velocity_equals_model_forward():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from g1deploy.fastplan import fast_planner_class

    flow, cfg, mem, hist, goal_fn = _scene()
    pc = flow.planner.PlannerConfig(**dict(cfg["planner"], device="cuda", seed=0))
    p = fast_planner_class()(pc, graphs=False, fuse=False)
    k = 4
    with torch.no_grad():
        h, xy0, yaw0 = p.encode_history(hist)
        hk = h.cuda()[None].repeat(k, 1, 1)
        xy, yaw = xy0.cuda()[None].repeat(k, 1), yaw0.cuda()[None].repeat(k)
        goal = torch.tensor([[1.5, 0.0, 1.0, 0.0]], device="cuda").repeat(k, 1)
        cond, _ = p._condition(hk, xy, yaw, goal, torch.zeros(0, 9, device="cuda"),
                               lambda a, b: torch.as_tensor(mem.ego_map(a, b, 0.075, 0.8)[0]))
        x = torch.randn(k, 25, 65, device="cuda")
        zeros, com = torch.zeros_like(x), torch.zeros(k, 25, dtype=torch.bool, device="cuda")
        temb, t0, _ = p._time_embeddings()
        mods, kvs = p._consts(cond)
        for i in (0, 9):
            ref = p.model(x, torch.full((k, 25, 1), 1.0 - i / 10, device="cuda"), zeros, com, cond)
            out = p._velocity_tokens(x, temb[i], t0, zeros, com, cond["prior_mask"], cond["history"], mods, kvs)
            assert (out - ref).abs().max().item() < 1e-4 * max(1.0, ref.abs().max().item())
