"""Released SONIC g1 tracker in Python: 1570-float observation + fused ONNX + PD targets.

Port of lucid-g1-deploy `tools/mujoco_player.py` (`Player._proprio_isaac`, `build_obs`,
`step_policy`), which was gated on reproducing Isaac tracking at nominal physics, and
whose layout the fused release export (`sonic_release_g1.onnx`, receipt
`release_export_v2`) was built against:

    [ ref joint pos f0..f9 (290, absolute, Isaac order) | ref joint vel f0..f9 (290)
    | ref pelvis orientation 6D f0..f9 (60) | ang-vel hist (30) | joint pos - default hist (290)
    | joint vel hist (290) | last action hist (290) | gravity hist (30) ]

History: 10 frames, oldest first; the first push after reset fills every slot. The
reference block is exactly what the Isaac callback writes into the released encoder
(`command_multi_future_nonflat` + `motion_anchor_ori_b_mf_nonflat`): flow `feed.target`
rows (10, 67) -> joints [:29], velocities [29:58], 6D [58:64] (projected to orthonormal).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import contract as C

HISTORY = 10
OBS_DIM = 1570


def quat_rotate_inv_wxyz(q, v):
    """Rotate world vector v into the frame of quaternion q (wxyz)."""
    w, x, y, z = q
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return r.T @ np.asarray(v)


def project_rotation_6d(values):
    """humanoid_lfh.native_closed_loop.project_rotation_6d (row-major first two columns -> orthonormal)."""
    x = np.asarray(values, dtype=np.float64)
    mat = x.reshape(*x.shape[:-1], 3, 2)
    a = mat[..., 0].copy()
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    a = np.where(norm > 1e-8, a / np.maximum(norm, 1e-8), np.array([1., 0., 0.]))
    b = mat[..., 1] - (mat[..., 1] * a).sum(-1, keepdims=True) * a
    bn = np.linalg.norm(b, axis=-1, keepdims=True)
    fallback = np.eye(3)[np.argmin(np.abs(a), axis=-1)]
    fallback -= (fallback * a).sum(-1, keepdims=True) * a
    fallback /= np.linalg.norm(fallback, axis=-1, keepdims=True)
    b = np.where(bn > 1e-8, b / np.maximum(bn, 1e-8), fallback)
    return np.stack([a, b], axis=-1).reshape(x.shape)


class SonicTracker:
    """Stateful SONIC policy. Call `reset()` at episode start, then once per 50 Hz tick:
    `act(proprio, reference)` -> PD position targets (MuJoCo order)."""

    def __init__(self, onnx_path, providers=("CPUExecutionProvider",), threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=list(providers))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if int(inp.shape[1]) != OBS_DIM:
            raise ValueError(f"expected a {OBS_DIM}-wide fused SONIC policy, got {inp.shape}")
        self.hist = {k: deque(maxlen=HISTORY) for k in ("ang_vel", "joint_pos", "joint_vel", "action", "gravity")}
        self.last_action = np.zeros(29)  # Isaac order

    def reset(self):
        for h in self.hist.values():
            h.clear()
        self.last_action = np.zeros(29)

    @staticmethod
    def proprio(base_quat_wxyz, ang_vel_body, q_mj, dq_mj):
        """Current proprioception (Isaac order), as the Isaac `actor_obs` terms."""
        return {
            "ang_vel": np.asarray(ang_vel_body, dtype=np.float64),
            "joint_pos": C.mj_to_isaac(np.asarray(q_mj) - C.DEFAULT_MJ),
            "joint_vel": C.mj_to_isaac(np.asarray(dq_mj)),
            "gravity": quat_rotate_inv_wxyz(base_quat_wxyz, [0.0, 0.0, -1.0]),
        }

    def _push(self, cur):
        cur = dict(cur, action=self.last_action.copy())
        for k, v in cur.items():
            h = self.hist[k]
            if not h:
                for _ in range(HISTORY):
                    h.append(np.array(v, copy=True))
            else:
                h.append(np.array(v, copy=True))

    def observation(self, cur, ref_q_mj, ref_dq_mj, ref_ori6):
        """Push `cur` into the history and build the 1570 observation.

        ref_q_mj, ref_dq_mj: (10, 29) MuJoCo order, absolute; ref_ori6: (10, 6) row-major first two
        columns of R(pelvis_now)^T R(ref pelvis)."""
        self._push(cur)
        ref = np.concatenate([
            C.mj_to_isaac(np.asarray(ref_q_mj)).reshape(-1),
            C.mj_to_isaac(np.asarray(ref_dq_mj)).reshape(-1),
            project_rotation_6d(np.asarray(ref_ori6)).reshape(-1),
        ])
        proprio = np.concatenate([np.concatenate(list(self.hist[k]))
                                  for k in ("ang_vel", "joint_pos", "joint_vel", "action", "gravity")])
        obs = np.concatenate([ref, proprio])
        if obs.shape != (OBS_DIM,):
            raise AssertionError(obs.shape)
        return obs

    def policy(self, obs):
        return self.session.run(None, {self.input_name: np.asarray(obs, dtype=np.float32)[None]})[0][0]

    def act(self, cur, ref_q_mj, ref_dq_mj, ref_ori6):
        """One tick: returns PD position targets in MuJoCo order."""
        action = self.policy(self.observation(cur, ref_q_mj, ref_dq_mj, ref_ori6)).astype(np.float64)
        self.last_action = action
        return C.DEFAULT_MJ + C.ACTION_SCALE_MJ * C.isaac_to_mj(action)


def pd_torque(target_mj, q_mj, dq_mj):
    tau = C.KP_MJ * (np.asarray(target_mj) - q_mj) - C.KD_MJ * np.asarray(dq_mj)
    return np.clip(tau, -C.EFFORT_MJ, C.EFFORT_MJ)
