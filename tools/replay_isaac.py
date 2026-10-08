#!/usr/bin/env python3
"""Gate G2: execute recorded Isaac closed-loop references open-loop in MuJoCo.

An Isaac episode (`<packet>/<episode>/callback/closed_loop.npz`) stores, per 50 Hz tick, the
(10, 67) SONIC target it fed (`references`, 6D relative to Isaac's measured pelvis) and the
measured pre-action state (`pre_action_qpos`). The absolute reference pelvis rotation is
R_isaac_measured · R6 (third column by cross product); MuJoCo recomputes the 6D against its own
pelvis, so the replay is the same reference stream executed by the same SONIC weights under
MuJoCo physics, from the same start state. No planner, no feedback to the plan.

Reported per episode: Isaac outcome, MuJoCo fall/contact, pelvis height and tilt traces vs Isaac,
xy divergence, joint tracking error to the reference (row 0) in both simulators.

usage: replay_isaac.py <episode_dir> [<episode_dir> ...] [--out runs/replay.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1deploy import bundle  # noqa: E402
from g1deploy.scoring import TaskScorer, root_tilt_rad  # noqa: E402
from sim.episode import Physics, read_json, task_start_state  # noqa: E402
from sim.scene import quat_matrix  # noqa: E402


def absolute_reference_rotations(refs, pre_q):
    six = refs[:, :, 58:64].reshape(len(refs), 10, 3, 2)
    a, b = six[..., 0], six[..., 1]
    rel = np.stack([a, b, np.cross(a, b)], axis=-1)
    meas = np.stack([quat_matrix(q[3:7]) for q in pre_q])
    return np.einsum("tij,tkjl->tkil", meas, rel)


def replay(ep: Path, onnx: Path) -> dict:
    cb = ep / "callback" if (ep / "callback").is_dir() else ep
    task, score, run = read_json(cb / "task.json"), read_json(cb / "score.json"), read_json(cb / "run_config.json")
    z = np.load(cb / "closed_loop.npz")
    refs, pre_q, real_q = z["references"], z["pre_action_qpos"], z["realized_qpos"]
    r_abs = absolute_reference_rotations(refs, pre_q)
    sim = Physics(task["scene_boxes_world"], onnx)
    sim.reset(task_start_state(task, run.get("initial_root_velocity_policy", "recorded_world"),
                               run.get("initial_joint_velocity_policy", "source")))
    scorer = TaskScorer(task)
    mj_q, joint_err_mj = [], []
    outcome = None
    for t in range(len(refs)):
        q_now = sim.data.qpos.copy()
        r_m = quat_matrix(q_now[3:7])
        ori6 = np.einsum("ij,kjl->kil", r_m.T, r_abs[t])[:, :, :2].reshape(10, 6)
        joint_err_mj.append(float(np.abs(q_now[7:] - refs[t, 0, :29]).mean()))
        pos, quat, force, _ = sim.step(refs[t, :, :29], refs[t, :, 29:58], ori6)
        m = sim.measure()
        mj_q.append(m["qpos"])
        if scorer.outcome is None:
            outcome = scorer.update(m["qpos"][:3], m["root_linear_velocity_w"], physics_root_positions=pos,
                                    physics_root_quaternions=quat, physics_contact_force_n=force)
        if outcome is not None and outcome != "goal_completion":
            break
        if root_tilt_rad(quat).max() > task["fall_tilt_rad"] or pos[:, 2].min() < task["fall_height_m"]:
            outcome = "fall_after_goal" if outcome == "goal_completion" else "fall"
            break
    mj_q = np.asarray(mj_q)
    n = len(mj_q)
    joint_err_isaac = [float(np.abs(pre_q[t, 7:] - refs[t, 0, :29]).mean()) for t in range(n)]
    return {
        "episode": ep.name, "family": task.get("family"), "ticks_isaac": len(refs), "ticks_mujoco": n,
        "isaac_outcome": score["outcome"], "mujoco_outcome": outcome or "ran_to_end",
        "isaac_max_tilt": float(root_tilt_rad(real_q[:, 3:7]).max()),
        "mujoco_max_tilt": float(root_tilt_rad(mj_q[:, 3:7]).max()),
        "isaac_min_pelvis_z": float(real_q[:, 2].min()), "mujoco_min_pelvis_z": float(mj_q[:, 2].min()),
        "pelvis_z_rmse": float(np.sqrt(np.mean((mj_q[:, 2] - real_q[:n, 2]) ** 2))),
        "xy_divergence_final_m": float(np.linalg.norm(mj_q[-1, :2] - real_q[n - 1, :2])),
        "joint_track_err_isaac_rad": float(np.mean(joint_err_isaac)),
        "joint_track_err_mujoco_rad": float(np.mean(joint_err_mj)),
        "mujoco_max_contact_n": scorer.max_force_n,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rows = []
    for e in args.episodes:
        r = replay(Path(e), bundle.sonic_onnx())
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
