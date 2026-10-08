"""Episode scoring identical to the Isaac harness (`humanoid_lfh.native_closed_loop.TaskScorer`).

Precedence: unexpected reset > fall (min physics-rate root height < fall_height_m or tilt >
fall_tilt_rad) > obstacle collision (physics-rate body-obstacle normal force > threshold) >
goal (xy distance <= goal_radius_m and xy speed <= goal_speed_m_s held goal_hold_s) > timeout.
"""

from __future__ import annotations

import math

import numpy as np


def root_tilt_rad(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1, 1))


class TaskScorer:
    def __init__(self, task: dict, control_dt_s: float = 0.02):
        self.task = task
        self.dt = control_dt_s
        self.goal = np.asarray(task["goal_world_xy"], dtype=np.float64)
        self.steps = self.hold_steps = 0
        self.goal_hold_achieved = False
        self.max_force_n = self.max_tilt_rad = 0.0
        self.minimum_root_height_m = math.inf
        self.rows = []
        self.outcome = None

    def update(self, root_position_m, root_linear_velocity_w, *, physics_root_positions,
               physics_root_quaternions, physics_contact_force_n, reset=False, planner_latency_s=0.0):
        if self.outcome is not None:
            raise RuntimeError("attempt already ended")
        t = self.task
        pp = np.asarray(physics_root_positions)
        tilt = float(np.max(root_tilt_rad(np.asarray(physics_root_quaternions))))
        height = float(np.min(pp[:, 2] - t.get("ground_z_m", 0.0)))
        contact = float(np.max(physics_contact_force_n)) if np.size(physics_contact_force_n) else 0.0
        p, v = np.asarray(root_position_m), np.asarray(root_linear_velocity_w)
        distance = float(np.linalg.norm(p[:2] - self.goal))
        speed = float(np.linalg.norm(v[:2]))
        self.steps += 1
        self.max_force_n = max(self.max_force_n, contact)
        self.max_tilt_rad = max(self.max_tilt_rad, tilt)
        self.minimum_root_height_m = min(self.minimum_root_height_m, height)
        inside = distance <= t["goal_radius_m"] and speed <= t["goal_speed_m_s"]
        self.hold_steps = self.hold_steps + 1 if inside else 0
        held = self.hold_steps * self.dt >= t["goal_hold_s"] - 1e-9
        self.goal_hold_achieved = self.goal_hold_achieved or held
        if reset:
            self.outcome = "unexpected_reset_or_native_termination"
        elif height < t["fall_height_m"] or tilt > t["fall_tilt_rad"]:
            self.outcome = "fall"
        elif contact > t["obstacle_contact_threshold_n"]:
            self.outcome = "obstacle_collision"
        elif held:
            self.outcome = "goal_completion"
        elif self.steps * self.dt >= t["deadline_s"] - 1e-9:
            self.outcome = "timeout"
        self.rows.append(dict(control_step=self.steps, elapsed_s=self.steps * self.dt, goal_distance_xy_m=distance,
                              root_speed_xy_m_s=speed, root_height_min_m=height, root_tilt_max_rad=tilt,
                              obstacle_contact_max_n=contact, planner_latency_s=float(planner_latency_s),
                              outcome=self.outcome))
        return self.outcome

    def report(self):
        arrival = next((r["elapsed_s"] for r in self.rows if r["goal_distance_xy_m"] <= self.task["goal_radius_m"]), None)
        return dict(outcome=self.outcome or "incomplete", completed=self.outcome == "goal_completion",
                    control_steps=self.steps, elapsed_s=self.steps * self.dt,
                    minimum_root_height_m=self.minimum_root_height_m if self.steps else None,
                    maximum_root_tilt_rad=self.max_tilt_rad, maximum_obstacle_contact_n=self.max_force_n,
                    final_goal_distance_xy_m=self.rows[-1]["goal_distance_xy_m"] if self.rows else None,
                    minimum_goal_distance_xy_m=min((r["goal_distance_xy_m"] for r in self.rows), default=None),
                    first_goal_radius_entry_s=arrival)
