"""The fast planner reads guide_margin_m at replay time (g1deploy/fastplan.py): a per-replan override of the margin
must reach a captured CUDA graph, which would otherwise keep the value it was captured with."""

import pytest
import torch

from g1deploy import bundle

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


def box(cx, cy, hx, hy, z0, z1):
    return {"center_xy": [cx, cy], "half_xy": [hx, hy], "yaw": 0.0, "z_bottom": z0, "z_top": z1}


def test_graph_follows_the_guide_margin():
    from g1deploy.fastplan import fast_planner_class

    flow = bundle.load_flow()
    cfg = bundle.student_config()
    P = flow.planner

    def make(**kw):
        return fast_planner_class()(P.PlannerConfig(**dict(cfg["planner"], device="cuda", seed=5)), **kw)

    fast, eager = make(), make(graphs=False, fuse=False)
    k = fast.cfg.candidates
    hist = torch.zeros(fast.history, 65)
    hist[:, 0] = 0.79 + fast.cfg.datum_offset_m
    hist[:, 3] = -1.0
    xy, yaw = torch.zeros(k, 2, device="cuda"), torch.zeros(k, device="cuda")
    goal = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda").repeat(k, 1)
    cond, anchor = fast._condition(hist.cuda()[None].repeat(k, 1, 1), xy, yaw, goal, torch.zeros(0, 9, device="cuda"))
    boxes = [box(0.5, 0.15, 0.1, 0.1, 0.0, 1.0), box(0.5, -0.25, 0.1, 0.1, 0.0, 1.0)]
    obs = P.pack_world([P.box_obstacle(P.inflate_box(b, 0.08, 0.08, 0.8), fast.cfg.datum_offset_m) for b in boxes], "cuda")
    f, e = {}, {}
    for i, m in enumerate((0.05, 0.02, 0.05)):  # the third replays the graph captured at 0.05 after the 0.02 one
        for p, d in ((fast, f), (eager, e)):
            p.cfg.guide_margin_m = m
            p.generator.manual_seed(11)
            d[m if i < 2 else "again"] = p._sample_guided(cond, anchor, xy, yaw, obs).clone()

    def dist(a, b):
        return float((a - b).abs().mean())

    # each replay follows the margin in force: far closer to the eager loop at that margin than at the other one
    # (fused vs eager differ by float32 noise, amplified by the normalised guidance gradient)
    for m, other in ((0.05, 0.02), (0.02, 0.05), ("again", 0.02)):
        same = 0.05 if m == "again" else m
        assert dist(f[m], e[same]) < 0.1 * dist(f[m], e[other]), m
