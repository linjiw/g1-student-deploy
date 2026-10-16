"""Deployment route corrections without changing the hash-pinned Flow bundle."""

from functools import wraps

import numpy as np


def _install_sensor_grid_validation(route_module):
    original = route_module.SensorRoute.__init__
    if getattr(original, "_g1_validate_unknown_growth", False):
        return

    @wraps(original)
    def init(self, start_xy, goal_xy, memory, **kwargs):
        growth = kwargs.get("grow_unknown_m", 0.0)
        if growth > 0 and int(round(growth / memory.res)) < 1:
            # scipy binary_dilation(iterations=0) means until stable, NOT zero
            # growth. Reject unsupported sub-cell settings before touching maps.
            raise ValueError("positive grow_unknown_m must round to at least one support-grid cell "
                             f"(buffer={growth} m, resolution={memory.res} m)")
        return original(self, start_xy, goal_xy, memory, **kwargs)

    init._g1_validate_unknown_growth = True
    route_module.SensorRoute.__init__ = init


def install(route_module):
    """Keep a segment when A* start and goal resolve to the same grid cell.

    Frozen Route._plan replaces both endpoints of a one-cell path in place,
    leaving only the goal. Route.project then has no segment to project onto.
    Preserve the actual endpoints, including a zero-length segment at the goal.
    This does not declare arrival, change occupancy or bypass goal hold/speed
    scoring. All other paths retain the frozen implementation's output.

    Patch the original class so SensorRoute and previously imported class
    references (including callback.Route) use the same correction. Idempotent.
    Also reject a positive sensor-route buffer that rounds to zero grid cells;
    SciPy would interpret zero dilation iterations as unbounded growth.
    """
    _install_sensor_grid_validation(route_module)
    original = route_module.Route._plan
    if getattr(original, "_g1_preserve_route_endpoints", False):
        return

    @wraps(original)
    def plan(self):
        path = original(self)
        if self._found and path.shape == (1, 2):
            return np.stack([self.start, self.goal])
        return path

    plan._g1_preserve_route_endpoints = True
    route_module.Route._plan = plan
