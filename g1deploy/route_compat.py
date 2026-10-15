"""Deployment route corrections without changing the hash-pinned Flow bundle."""

from functools import wraps

import numpy as np


def install(route_module):
    """Keep a segment when A* start and goal resolve to the same grid cell.

    Frozen Route._plan replaces both endpoints of a one-cell path in place,
    leaving only the goal. Route.project then has no segment to project onto.
    Preserve the actual endpoints, including a zero-length segment at the goal.
    This does not declare arrival, change occupancy or bypass goal hold/speed
    scoring. All other paths retain the frozen implementation's output.

    Patch the original class so SensorRoute and previously imported class
    references (including callback.Route) use the same correction. Idempotent.
    """
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
