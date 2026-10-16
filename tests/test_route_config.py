"""Deployment route uncertainty defaults and source memory preservation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from g1deploy import bundle


@pytest.mark.parametrize("override,expected", [(None, .1), (0., 0.), (.2, .2)])
def test_effective_config_defaults_only_when_absent_and_does_not_rewrite_bundle(tmp_path, override, expected):
    route = {"period_s": .1}
    if override is not None:
        route["grow_unknown_m"] = override
    raw = dict(planner={"run": "fixture-model"}, route_variants={"sensor": {"route": route},
                                                               "privileged": {"route": {}}})
    p = tmp_path / "config/student.json"
    p.parent.mkdir()
    original = json.dumps(raw).encode()
    p.write_bytes(original)
    cfg = bundle.student_config(tmp_path)
    assert cfg["route_variants"]["sensor"]["route"]["grow_unknown_m"] == expected
    assert cfg["route_variants"]["sensor"]["route"]["period_s"] == .1
    assert cfg["route_variants"]["privileged"]["route"] == {}
    assert p.read_bytes() == original


def test_unknown_growth_leaves_observed_free_cells_and_sensor_layers_unchanged():
    flow = bundle.load_flow()
    support = np.full((21, 21), np.nan)
    support[8, 10] = 1.9
    support[9, :] = 0.  # an observed free strip cannot become uncertainty growth
    mem = SimpleNamespace(support=support, n=21, lo=np.array([0., 0.]), res=.05)
    before = support.copy()
    guarded = flow.route.SensorRoute([0, 0], [.9, 0], mem, grow_unknown_m=.1, inflate_m=0)
    legacy = flow.route.SensorRoute([0, 0], [.9, 0], mem, grow_unknown_m=0, inflate_m=0)
    assert guarded.occupied[6, 10] and not legacy.occupied[6, 10]
    assert not guarded.occupied[9, :].any()
    np.testing.assert_array_equal(mem.support, before)
    assert np.isnan(mem.support[6, 10])  # route buffer is not sensor evidence


def test_shared_effective_config_is_accepted_by_actual_sensor_route():
    flow = bundle.load_flow()
    cfg = bundle.student_config()
    route_cfg = cfg["route_variants"]["sensor"]["route"]
    mem = SimpleNamespace(support=np.zeros((21, 21)), n=21, lo=np.array([0., 0.]), res=.05)
    route = flow.route.SensorRoute([0, 0], [.9, 0], mem, **route_cfg)
    assert route.grow_unknown_m == .1 and route.period_s == .1
    # Both callers obtain route_cfg from student_config() and pass it through;
    # rollout acceptance additionally checks native/run_config.json provenance.
    assert route.found
