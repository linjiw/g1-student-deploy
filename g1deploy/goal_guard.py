"""Keep an extrapolated local goal on the free prefix of its route-grid ray.

This uses the same inflated occupancy and unknown-cell convention as Route.
It does not establish whole-body clearance or certify unobserved space.
"""

import numpy as np

REVISION = "route-free-prefix-v1"


def limit_goal(route, xy, target):
    xy, target = np.asarray(xy, dtype=float), np.asarray(target, dtype=float)
    dist = float(np.linalg.norm(target - xy))
    n = int(np.ceil(dist / (0.5 * route.res))) + 1
    points = np.linspace(xy, target, n)
    ij = np.round((points - route.lo) / route.res).astype(int)
    inside = ((ij >= 0) & (ij < np.asarray(route.shape))).all(axis=1)
    clipped = np.clip(ij, 0, np.asarray(route.shape) - 1)
    blocked = ~inside | route.occupied[clipped[:, 0], clipped[:, 1]]
    if blocked[0]:
        # Route._nearest_free already allows escape from an inflated occupied
        # start. Do not replace the learned escape with an artificial freeze.
        return target, "occupied_start_uncertified"
    hits = np.flatnonzero(blocked)
    if not len(hits):
        return target, "grid_line_free"
    return points[max(0, hits[0] - 1)], "limited_at_occupied_grid"
