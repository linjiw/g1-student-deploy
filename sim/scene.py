"""MuJoCo G1 scene with a task's obstacle boxes, plus measurement/contact helpers.

Robot: lucid-g1-deploy `runner/g1/g1_29dof.xml` (Unitree G1 29-DoF rev 1.0 with rubber hands;
mesh collision on links, 4 sphere contacts per foot), copied to `sim/assets`. Physics at
5 ms with Isaac's armature (kp / omega^2) on every actuated dof, as the lucid MuJoCo player.
Task boxes (native harness schema: center_xy, half_xy, yaw, z_bottom, z_top) become static
colliding box geoms named `obstacle_<i>`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from g1deploy import contract as C

ASSETS = Path(__file__).resolve().parent / "assets"
G1_XML = ASSETS / "g1_29dof.xml"
LIDAR_TRANSPARENT_MESHES = ("head_link",)


def augment_spec(spec, boxes=()):
    """Add task boxes as static colliders `obstacle_<i>` and make the head shell lidar-transparent."""
    import mujoco

    for g in spec.geoms:
        # The MID-360 sits inside the head shell's mesh in this MJCF; on the robot it looks out of a
        # cutout. Group 3 keeps the head colliding but outside the lidar's ray mask (groups 0 only).
        if g.meshname in LIDAR_TRANSPARENT_MESHES and g.contype:
            g.group = 3
    for i, b in enumerate(boxes):
        h = 0.5 * (b["z_top"] - b["z_bottom"])
        yaw = float(b["yaw"])
        g = spec.worldbody.add_geom()
        g.name = f"obstacle_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [b["half_xy"][0], b["half_xy"][1], h]
        g.pos = [b["center_xy"][0], b["center_xy"][1], b["z_bottom"] + h]
        g.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        g.rgba = [0.85, 0.45, 0.2, 0.9] if b["z_bottom"] >= 0.8 else [0.3, 0.55, 0.85, 0.9]
    return spec


def build_model(boxes=(), xml: Path = G1_XML, friction: float | None = None):
    import mujoco

    model = augment_spec(mujoco.MjSpec.from_file(str(xml)), boxes).compile()
    model.opt.timestep = C.SIM_DT
    joints = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if tuple(joints) != C.MJ_JOINTS:
        raise ValueError(f"MJCF joint order differs from the contract: {joints}")
    model.dof_armature[6:] = C.ARMATURE_MJ
    if friction is not None:
        model.geom_friction[:, 0] = friction
    return model


class SceneIndex:
    """Geom ids for obstacle-contact scoring and lidar exclusion."""

    def __init__(self, model):
        import mujoco

        self.obstacles = np.array([g for g in range(model.ngeom) if (model.geom(g).name or "").startswith("obstacle_")],
                                  dtype=int)
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.robot_body = np.zeros(model.nbody, dtype=bool)
        for b in range(model.nbody):
            r = b
            while r != 0 and r != pelvis:
                r = model.body_parentid[r]
            self.robot_body[b] = r == pelvis
        self.robot_geom = self.robot_body[model.geom_bodyid]
        self.pelvis = pelvis
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")


def obstacle_contact_force(model, data, index: SceneIndex) -> float:
    """Max normal force (N) over robot-obstacle contacts in the current state."""
    import mujoco

    if len(index.obstacles) == 0 or data.ncon == 0:
        return 0.0
    obst = set(index.obstacles.tolist())
    best = 0.0
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in obst and index.robot_geom[g2]) or (g2 in obst and index.robot_geom[g1]):
            mujoco.mj_contactForce(model, data, i, f6)
            best = max(best, abs(float(f6[0])))
    return best


def quat_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def measurement(data):
    """Isaac `read_native_measurement` equivalent: world qpos (36), dq (29), world root velocities."""
    q = data.qpos.copy()
    r = quat_matrix(q[3:7])
    return {
        "qpos": q.astype(np.float64),
        "dq": data.qvel[6:].copy(),
        "root_linear_velocity_w": data.qvel[0:3].copy(),
        "root_angular_velocity_w": r @ data.qvel[3:6],  # MuJoCo free-joint angular velocity is body frame
        "root_angular_velocity_b": data.qvel[3:6].copy(),
    }


def set_state(model, data, root_pos, root_quat_wxyz, root_lin_w, root_ang_w, q_mj, dq_mj):
    import mujoco

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = np.asarray(root_quat_wxyz) / np.linalg.norm(root_quat_wxyz)
    data.qpos[7:] = q_mj
    data.qvel[0:3] = root_lin_w
    data.qvel[3:6] = quat_matrix(data.qpos[3:7]).T @ np.asarray(root_ang_w)
    data.qvel[6:] = dq_mj
    mujoco.mj_forward(model, data)
