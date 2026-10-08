import json

import numpy as np
import pytest

from g1deploy import bundle, contract as C
from g1deploy.sonic import SonicTracker, project_rotation_6d


@pytest.fixture(scope="module")
def tracker():
    return SonicTracker(bundle.sonic_onnx())


def test_onnx_matches_export_parity_vectors(tracker):
    z = np.load(bundle.ROOT / "sonic/export_parity_vectors.npz")
    for obs, expected in zip(z["observation"], z["expected_action"]):
        got = tracker.policy(obs[0])
        assert np.abs(got - expected[0]).max() < 1e-4


def test_joint_orders_match_export_receipt():
    r = json.loads((bundle.ROOT / "sonic/export_receipt.json").read_text())
    assert tuple(r["action_joint_names"]) == C.ISAAC_JOINTS
    assert tuple(r["native_vector_joint_names"]) == C.MJ_JOINTS
    assert np.array_equal(C.isaac_to_mj(C.mj_to_isaac(np.arange(29))), np.arange(29))


def test_joint_order_matches_planner_and_contract():
    flow = bundle.load_flow()
    assert flow.planner.JOINT_NAMES == C.MJ_JOINTS
    contract = json.loads((bundle.ROOT / "config/sonic_model_contract.json").read_text())
    assert tuple(j["name"] for j in contract["joints"] if j["type"] != "free") == C.MJ_JOINTS


def test_observation_layout_and_history_fill(tracker):
    tracker.reset()
    cur = SonicTracker.proprio([1, 0, 0, 0], [0.1, 0.2, 0.3], C.DEFAULT_MJ + 0.01, np.zeros(29))
    ref_q = np.tile(C.DEFAULT_MJ, (10, 1))
    ori = np.tile([1, 0, 0, 1, 0, 0], (10, 1)).astype(float)
    obs = tracker.observation(cur, ref_q, np.zeros((10, 29)), ori)
    assert obs.shape == (1570,)
    assert np.allclose(obs[:29], C.mj_to_isaac(C.DEFAULT_MJ))
    assert np.allclose(obs[580:586], [1, 0, 0, 1, 0, 0])
    assert np.allclose(obs[640:643], [0.1, 0.2, 0.3]) and np.allclose(obs[667:670], [0.1, 0.2, 0.3])
    assert np.allclose(obs[-3:], [0, 0, -1])  # gravity, upright


def test_project_rotation_6d_is_identity_on_rotations():
    from scipy.spatial.transform import Rotation

    r = Rotation.random(5, random_state=0).as_matrix()
    six = r[:, :, :2].reshape(5, 6)
    assert np.allclose(project_rotation_6d(six), six)
