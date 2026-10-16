"""Deployment selection corrections; the frozen Flow files stay hash-identical.

The height-field selector returns discrete clearance levels. An all-unsafe
fallback therefore often ties at -1. Frozen Flow chooses the first GO sample
even after generating a STOP pool. Preserve maximum clearance, but resolve a
tie using the intended STOP pool and then the least translation. This is a
preference between learned candidates, not a synthesized stop controller.
"""

from functools import wraps

import numpy as np


REVISION = "fallback-pool-ties-v1"


def fallback_choice(pools, anchor_xy):
    """Return a global candidate index without changing clearance values/order."""
    clearance = np.concatenate([c for _, c, _ in pools])
    tied = np.flatnonzero(clearance == np.max(clearance))
    labels = np.concatenate([np.repeat(kind, len(c)) for _, c, kind in pools])
    stops = tied[labels[tied] == "stop"]
    if len(stops):
        tied = stops
    q = np.concatenate([p["qpos"] for p, _, _ in pools])
    motion = np.square(q[tied, :, :2] - np.asarray(anchor_xy)).sum(axis=2).mean(axis=1)
    return int(tied[np.argmin(motion)])


def install(planner_module):
    """Patch the original class in place, including fast planner subclasses.

No extra model evaluation is performed. Candidate arrays, clearance thresholds,
and random stream remain unchanged. The optional execution verifier retains its
own selection: this adapter never substitutes a candidate after that verifier.
"""
    original = planner_module.FlowPlanner.plan
    if getattr(original, "_g1_fallback_pool_ties", False):
        return

    @wraps(original)
    def plan(self, qpos_hist, *args, **kwargs):
        goal_fn = args[1] if len(args) > 1 else kwargs.get("goal_fn")
        if hasattr(goal_fn, "reset_diagnostics"):
            goal_fn.reset_diagnostics()
        out = original(self, qpos_hist, *args, **kwargs)
        pools = self.last_pools
        choice = out["choice"]
        old_choice = choice
        if out["rule"] == "stop_max_clearance" and out["exec_check"] is None:
            choice = fallback_choice(pools, np.asarray(qpos_hist)[-1, :2])
        offset = 0
        for candidate, clearance, kind in pools:
            if choice < offset + len(clearance):
                local = choice - offset
                break
            offset += len(clearance)
        if choice != old_choice:
            q = candidate["qpos"][local].copy()
            q[:, 2] -= self.cfg.datum_offset_m
            states = candidate["states"][local].cpu().numpy()
            jvel = states[:, 36:65]
            lift = None
            if self.cfg.ground_legs:
                from motion2scene_flow.closed_loop.grounding import ground_legs

                q, lift = ground_legs(q, self.certifier.tree, self.certifier.bodies,
                                     self.cfg.stance_ankle_z_m, self.limits)
                jvel = np.concatenate([np.diff(q[:, 7:], axis=0), np.diff(q[-2:, 7:], axis=0)]) / planner_module.DT
            out.update(qpos=q, rotation=candidate["rot"][local].cpu().numpy(),
                       joint_velocity=jvel, states=states, choice=choice,
                       ground_lift_max_m=None if lift is None else float(lift.max()),
                       chosen_clearance_m=float(clearance[local]))
        # A higher-clearance GO candidate can still win. Do not call it a stop.
        if out["rule"] == "stop_max_clearance" and kind != "stop":
            out["rule"] = "fallback_go_max_clearance"
        out.update(candidate_pool=kind,
                   no_safe_candidate=not any(np.any(c >= self.cfg.select_margin_m) for _, c, _ in pools),
                   selection_revision=REVISION,
                   fallback_tie_changed=choice != old_choice)
        if hasattr(goal_fn, "diagnostics"):
            import copy
            out["goal_diagnostics"] = copy.deepcopy(goal_fn.diagnostics)
        return out

    plan._g1_fallback_pool_ties = True
    planner_module.FlowPlanner.plan = plan
