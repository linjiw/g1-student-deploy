"""Actual frozen selector with explicit candidate fixtures, no sealed episodes.

The fixtures reproduce the discrete 0.04/-1 clearance ties observed in the
owned corridor runs. They are algorithm regressions, not policy success data.
"""

import numpy as np
import pytest
import torch

from g1deploy import bundle
from g1deploy.planner_compat import fallback_choice, install


def candidate(xs, *, z=0.875):
    q = np.zeros((len(xs), 4, 36), dtype=np.float64)
    q[:, :, 0] = np.asarray(xs)[:, None] * np.arange(1, 5) / 4
    q[:, :, 2] = z
    q[:, :, 3] = 1
    # Nontrivial joint order/velocity payload makes rematerialization testable.
    q[:, :, 7:] = np.arange(29) / 100
    states = torch.zeros(len(xs), 4, 65)
    states[:, :, 36:65] = torch.arange(29) / 1000
    return dict(qpos=q, rot=torch.eye(3).repeat(len(xs), 4, 1, 1), states=states,
                end_xy=q[:, -1, :2], goal0=np.array([1., 0., 1., 0.]))


def planner(clearances):
    flow = bundle.load_flow()
    p = object.__new__(flow.planner.FlowPlanner)
    p.cfg = flow.planner.PlannerConfig(run="unused-fixture", device="cpu", candidates=1,
                                     fallback="resample_turn", select_margin_m=0.08)
    p.device = torch.device("cpu")
    p.calls = 0
    p.encode_history = lambda hist: (torch.zeros(2, 65), torch.zeros(2), torch.tensor(0.))
    batches = iter([candidate([0.5]), candidate([0.4, 0.3, 0.2]), candidate([0.01])])
    p._candidates = lambda *args: next(batches)
    values = iter([np.asarray(x) for x in clearances])
    p._clearance = lambda *args: next(values)
    return p


def run(p, corrected=True):
    method = type(p).plan if corrected else type(p).plan.__wrapped__
    return method(p, np.zeros((2, 36)), [], lambda xy, yaw: None)


@pytest.mark.parametrize("level", [-1., 0., 0.04])
def test_quantized_tie_chooses_actual_stop_pool_without_relaxing_clearance(level):
    values = [[level], [level, level, level], [level]]
    before, after = run(planner(values), False), run(planner(values))
    assert before["choice"] == 0 and before["rule"] == "stop_max_clearance"
    assert after["choice"] == 4 and after["candidate_pool"] == "stop"
    assert after["rule"] == "stop_max_clearance" and after["no_safe_candidate"]
    assert after["clearance_m"] == before["clearance_m"] == [level] * 5
    assert after["chosen_clearance_m"] == before["chosen_clearance_m"] == level
    np.testing.assert_array_equal(after["qpos"][:, 2], np.full(4, 0.8))
    np.testing.assert_array_equal(after["qpos"][:, 7:], candidate([0.01])["qpos"][0, :, 7:])
    np.testing.assert_array_equal(after["joint_velocity"], candidate([0.01])["states"][0, :, 36:65])
    assert after["qpos"][-1, 0] == 0.01


def test_strictly_higher_clearance_go_stays_selected_and_is_labelled_honestly():
    values = [[0.04], [0, 0, 0], [-1]]
    before, after = run(planner(values), False), run(planner(values))
    assert before["choice"] == after["choice"] == 0
    assert after["rule"] == "fallback_go_max_clearance"
    assert after["candidate_pool"] == "go" and after["no_safe_candidate"]
    for key in ("qpos", "rotation", "joint_velocity", "states", "clearance_m"):
        np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("values", [[[0.08]], [[0.0], [0.08, 0.0, 0.0]]])
def test_safe_choice_is_byte_equal_to_frozen_selector(values):
    before, after = run(planner(values), False), run(planner(values))
    assert before["choice"] == after["choice"] and before["rule"] == after["rule"]
    assert not after["no_safe_candidate"] and not after["fallback_tie_changed"]
    for key in before:
        if isinstance(before[key], (np.ndarray, list)):
            np.testing.assert_array_equal(before[key], after[key])
        else:
            assert before[key] == after[key]


def test_least_translation_tie_preserves_global_array_order():
    pools = [(candidate([0.4, 0.1]), np.array([-1., -1.]), "go"),
             (candidate([0.03, 0.01]), np.array([-1., -1.]), "stop")]
    assert fallback_choice(pools, [0, 0]) == 3
    pools[1][0]["qpos"][1] = pools[1][0]["qpos"][0]
    assert fallback_choice(pools, [0, 0]) == 2  # stable index for exact ties


def test_installer_idempotent_for_existing_class_and_fast_subclasses():
    flow = bundle.load_flow()
    cls = flow.planner.FlowPlanner
    method = cls.plan
    install(flow.planner)
    assert cls is flow.planner.FlowPlanner and cls.plan is method
    from g1deploy.fastplan import fast_planner_class
    fast_cls = fast_planner_class()
    assert fast_cls.__mro__[1].plan is method
    p = planner([[-1.], [-1., -1., -1.], [-1.]])
    fast = object.__new__(fast_cls)
    fast.__dict__.update(p.__dict__)
    result = fast.plan(np.zeros((2, 36)), [], lambda xy, yaw: None)
    assert result["candidate_pool"] == "stop" and result["choice"] == 4
