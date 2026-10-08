"""G1 29-DoF joint contract shared by the planner, the SONIC tracker and the stream.

Two joint orders exist:
- MuJoCo / Unitree SDK / planner ("native") order: legs L then R, waist, arms L then R
  (`flow closed_loop.planner.JOINT_NAMES`, `rt/lowcmd` motor index 0..28).
- IsaacLab order: breadth-first over the kinematic tree; SONIC's observation, action
  and the ZMQ stream (`joint_pos`/`joint_vel`) use it.

Gains, defaults, effort limits and the action scale are SONIC's training constants,
as verified in lucid-g1-deploy `tools/mujoco_player.py` (against
gear_sonic/envs/manager_env/robots/g1.py and `policy_parameters.hpp`).
"""

from __future__ import annotations

import numpy as np

MJ_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

#: isaac_to_mj[i] = MuJoCo index of IsaacLab joint i (gear_sonic_deploy visualize_motion.py:65, inverted).
ISAAC_TO_MJ = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25,
                        19, 26, 20, 27, 21, 28])
MJ_TO_ISAAC = np.argsort(ISAAC_TO_MJ)
ISAAC_JOINTS = tuple(MJ_JOINTS[j] for j in ISAAC_TO_MJ)

_OMEGA, _ZETA = 2.0 * np.pi * 10.0, 2.0
_KP = {"7520_22": 99.13, "7520_14": 40.18, "2x5020": 28.51, "5020": 14.25, "4010": 16.78}
_KD = {"7520_22": 6.31, "7520_14": 2.56, "2x5020": 1.81, "5020": 0.907, "4010": 1.068}


def _group(name: str) -> str:
    n = name.replace("_joint", "")
    if any(k in n for k in ("hip_pitch", "hip_roll", "knee")):
        return "7520_22"
    if any(k in n for k in ("hip_yaw", "waist_yaw")):
        return "7520_14"
    if any(k in n for k in ("ankle", "waist_roll", "waist_pitch")):
        return "2x5020"
    if any(k in n for k in ("wrist_pitch", "wrist_yaw")):
        return "4010"
    return "5020"


def _effort(name: str) -> float:
    n = name.replace("_joint", "")
    if "hip_yaw" in n or "waist_yaw" in n:
        return 88.0
    if any(k in n for k in ("hip_roll", "hip_pitch", "knee")):
        return 139.0
    if any(k in n for k in ("ankle", "waist_roll", "waist_pitch")):
        return 50.0
    if any(k in n for k in ("wrist_pitch", "wrist_yaw")):
        return 5.0
    return 25.0


def _default(name: str) -> float:
    n = name.replace("_joint", "")
    if n.endswith("hip_pitch"):
        return -0.312
    if n.endswith("knee"):
        return 0.669
    if n.endswith("ankle_pitch"):
        return -0.363
    if n.endswith("elbow"):
        return 0.6
    if n == "left_shoulder_roll":
        return 0.2
    if n == "right_shoulder_roll":
        return -0.2
    if n.endswith("shoulder_pitch"):
        return 0.2
    return 0.0


KP_MJ = np.array([_KP[_group(n)] for n in MJ_JOINTS])
KD_MJ = np.array([_KD[_group(n)] for n in MJ_JOINTS])
EFFORT_MJ = np.array([_effort(n) for n in MJ_JOINTS])
DEFAULT_MJ = np.array([_default(n) for n in MJ_JOINTS])
#: q_des = default + scale * action (per joint, MuJoCo order): 0.25 * effort / kp.
ACTION_SCALE_MJ = 0.25 * EFFORT_MJ / KP_MJ
#: Isaac's actuator model: armature = kp / omega^2.
ARMATURE_MJ = KP_MJ / _OMEGA**2

DT = 0.02  # policy / planner tick
SIM_DT = 0.005  # Isaac physics dt (decimation 4)


def mj_to_isaac(x):
    return np.asarray(x)[..., ISAAC_TO_MJ]


def isaac_to_mj(x):
    return np.asarray(x)[..., MJ_TO_ISAAC]
