#!/usr/bin/env python3
"""Gate G4a: label-free MuJoCo lidar memory vs the labelled (Isaac) memory on one trajectory.

Replays a recorded episode's realized states kinematically in MuJoCo (no dynamics), steps the
analytic `RayCastMemory` (what Isaac's student saw) and the `MjRayMemory` (MuJoCo rays incl. the
robot's own body -> self filter -> label-free classification) side by side, and compares:
obstacle cells (support > 5 cm), ceiling cells, the guidance boxes, and self-hit leakage.

usage: perception_check.py <episode dir with closed_loop.npz + task.json> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1deploy import bundle  # noqa: E402
from g1deploy.perception.cloud_memory import MjRayMemory  # noqa: E402
from sim.episode import Physics  # noqa: E402


def check(ep):
    ep = Path(ep)
    cb = ep / "callback" if (ep / "callback").is_dir() else ep
    task = json.loads((cb / "task.json").read_text())
    z = np.load(cb / "closed_loop.npz")
    traj = z["pre_action_qpos"]
    flow = bundle.load_flow()
    sensor = bundle.student_config()["planner"]["sensor"]
    sim = Physics(task["scene_boxes_world"], bundle.sonic_onnx())
    ref = flow.sensor_map.RayCastMemory(task["scene_boxes_world"], traj[0][:2], rng=np.random.default_rng(0), **sensor)
    mem = MjRayMemory(sim, traj[0][:2], rng=np.random.default_rng(0), **sensor)
    robot = sim.index.robot_geom
    leak = total_self = 0
    for k, q in enumerate(traj):
        sim.data.qpos[:] = q
        sim.data.qvel[:] = 0
        sim.mujoco.mj_forward(sim.model, sim.data)
        before = mem.captures
        ref.step(k * 0.02, q[:3], q[3:7])
        mem.step(k * 0.02, q[:3], q[3:7])
        if mem.captures > before:
            g = mem.source.last_geom
            self_hits = robot[g]
            total_self += int(self_hits.sum())
    # leakage: re-run the filter on the last capture's self hits is not stored; estimate from stats
    sup_r = np.nan_to_num(ref.support, nan=0.0) > 0.05
    sup_m = np.nan_to_num(mem.support, nan=0.0) > 0.05
    ceil_r = np.isfinite(ref.ceiling) & (ref.ceiling < 1e2)
    ceil_m = np.isfinite(mem.ceiling) & (mem.ceiling < 1e2)
    both_c = ceil_r & ceil_m
    gb_r = flow.sensor_map.memory_boxes(ref, grow_overhead_m=0.6)
    gb_m = flow.sensor_map.memory_boxes(mem, grow_overhead_m=0.6)
    return {
        "episode": ep.name, "family": task.get("family"), "captures": mem.captures,
        "obstacle_cells_ref": int(sup_r.sum()), "obstacle_cells_cloud": int(sup_m.sum()),
        "obstacle_missed": int((sup_r & ~sup_m).sum()), "obstacle_extra": int((sup_m & ~sup_r).sum()),
        "ceiling_cells_ref": int(ceil_r.sum()), "ceiling_cells_cloud": int(ceil_m.sum()),
        "ceiling_missed": int((ceil_r & ~ceil_m).sum()), "ceiling_extra": int((ceil_m & ~ceil_r).sum()),
        "ceiling_abs_err_m": float(np.abs(ref.ceiling[both_c] - mem.ceiling[both_c]).mean()) if both_c.any() else None,
        "boxes_ref": len(gb_r), "boxes_cloud": len(gb_m),
        "self_hits": total_self, "self_removed": mem.stats["self_removed"], "points": mem.stats["points"],
    }


if __name__ == "__main__":
    for e in sys.argv[1:]:
        print(json.dumps(check(e)), flush=True)
