"""Shared episode machinery: task start state, physics stepping with the SONIC tracker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from g1deploy import contract as C
from g1deploy.sonic import SonicTracker, pd_torque

from .scene import SceneIndex, build_model, measurement, obstacle_contact_force, quat_matrix, set_state

DECIMATION = int(round(C.DT / C.SIM_DT))


def read_json(path):
    return json.loads(Path(path).read_text())


def task_start_state(task, root_velocity_policy="recorded_world", joint_velocity_policy="source"):
    """The Isaac callback's start: recorded SONIC walk frame `initial_frame`, xy from the task."""
    z = np.load(task["initial_state_record"], allow_pickle=False)
    names = tuple(str(n) for n in z["joint_names"])
    if names != C.MJ_JOINTS:
        raise ValueError("initial-state record joint order differs from the contract")
    f = int(task.get("initial_frame", 0))
    pos = np.asarray(z["root_position_m"][f], dtype=np.float64)
    pos[:2] = np.asarray(task.get("initial_root_xy", pos[:2]))
    pos[2] += float(task.get("initial_root_z_offset_m", 0.0))
    has_vel = "root_linear_velocity_w" in z.files and "root_angular_velocity_w" in z.files
    lin = z["root_linear_velocity_w"][f] if has_vel and root_velocity_policy == "recorded_world" else np.zeros(3)
    ang = z["root_angular_velocity_w"][f] if has_vel and root_velocity_policy == "recorded_world" else np.zeros(3)
    dq = np.zeros(29) if joint_velocity_policy == "zero" else z["joint_velocities_rad_s"][f]
    return dict(pos=pos, quat=np.asarray(z["root_quaternion_wxyz"][f], dtype=np.float64), lin=np.asarray(lin),
                ang=np.asarray(ang), q=np.asarray(z["joint_positions_rad"][f], dtype=np.float64), dq=np.asarray(dq))


def stand_start_state(xy=(0.0, 0.0), yaw=0.0, height=0.793):
    """Standing at SONIC's default pose (the hardware start after the runner's INIT ramp)."""
    return dict(pos=np.array([xy[0], xy[1], height]), quat=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
                lin=np.zeros(3), ang=np.zeros(3), q=C.DEFAULT_MJ.copy(), dq=np.zeros(29))


class Physics:
    """MuJoCo G1 + task boxes + SONIC tracker, stepped one 50 Hz tick at a time."""

    def __init__(self, boxes, onnx_path, friction=None):
        import mujoco

        self.mujoco = mujoco
        self.boxes = list(boxes)
        self.model = build_model(self.boxes, friction=friction)
        self.data = mujoco.MjData(self.model)
        self.index = SceneIndex(self.model)
        self.tracker = SonicTracker(onnx_path)
        self.t = 0.0
        self.hooks = []  # called with the sim time after every physics step (e.g. the MID-360 scanner)

    def reset(self, start):
        set_state(self.model, self.data, start["pos"], start["quat"], start["lin"], start["ang"], start["q"], start["dq"])
        self.tracker.reset()
        self.t = 0.0
        # Isaac settles the start pose so feet rest on the floor: drop the robot onto the ground if it
        # starts penetrating (recorded start heights come from a slightly different collision model).
        lowest = self._lowest_foot_z()
        if lowest < 0.0:
            self.data.qpos[2] -= lowest
            self.mujoco.mj_forward(self.model, self.data)

    def _lowest_foot_z(self):
        m, d = self.model, self.data
        zs = [d.geom_xpos[g][2] - m.geom_size[g][0] for g in range(m.ngeom)
              if m.geom_type[g] == self.mujoco.mjtGeom.mjGEOM_SPHERE and self.index.robot_geom[g]]
        return float(min(zs)) if zs else 0.0

    def measure(self):
        return measurement(self.data)

    def step(self, ref_q_mj, ref_dq_mj, ref_ori6, push=None):
        """Run SONIC on the current state and integrate one control tick (4 physics steps).
        Returns per-substep root positions, quaternions and obstacle contact forces."""
        d = self.data
        m = measurement(d)
        cur = SonicTracker.proprio(d.qpos[3:7], m["root_angular_velocity_b"], d.qpos[7:], d.qvel[6:])
        target = self.tracker.act(cur, ref_q_mj, ref_dq_mj, ref_ori6)
        if push is not None:
            d.qvel[0:2] += push[:2]
            d.qvel[3:6] += quat_matrix(d.qpos[3:7]).T @ np.array([0.0, 0.0, push[2]])
        pos, quat, force = [], [], []
        for _ in range(DECIMATION):
            d.ctrl[:] = pd_torque(target, d.qpos[7:], d.qvel[6:])
            self.mujoco.mj_step(self.model, d)
            self.t += C.SIM_DT
            for hook in self.hooks:
                hook(self.t)
            pos.append(d.qpos[0:3].copy())
            quat.append(d.qpos[3:7].copy())
            force.append(obstacle_contact_force(self.model, d, self.index))
        return np.asarray(pos), np.asarray(quat), np.asarray(force), target
