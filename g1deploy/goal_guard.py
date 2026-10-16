"""Keep an extrapolated local goal connected to the visible route corridor.

This uses the same inflated occupancy and unknown-cell convention as Route.
It does not establish whole-body clearance or certify unobserved space.
"""

import numpy as np

REVISION = "route-visible-goal-v4-anchor-arrival"


def update_arrival_latch(state, anchor, goal, radius, release_distance, scope):
    """Latch a world yaw from a real plan request, never candidate iteration.

    Entry uses the existing navigation position tolerance. Release uses the
    existing near-goal lookahead region for hysteresis; a new goal/run resets it.
    """
    if radius is None:
        return None
    goal = np.asarray(goal, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    distance = float(np.linalg.norm(anchor[:2] - goal))
    if state is not None and (state["scope"] != scope or state["goal_xy"] != goal.tolist()
                              or state["radius_m"] != radius or distance > release_distance):
        state = None
    if state is None and distance <= radius:
        w, x, y, z = anchor[3:7]
        state = dict(scope=scope, goal_xy=goal.tolist(), radius_m=float(radius),
                     heading_rad=float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))),
                     entry_anchor_xy=anchor[:2].tolist(), release_distance_m=float(release_distance))
    return state


def _ray(route, xy, target):
    dist = float(np.linalg.norm(target - xy))
    n = int(np.ceil(dist / (0.5 * route.res))) + 1
    points = np.linspace(xy, target, n)
    ij = np.round((points - route.lo) / route.res).astype(int)
    inside = ((ij >= 0) & (ij < np.asarray(route.shape))).all(axis=1)
    clipped = np.clip(ij, 0, np.asarray(route.shape) - 1)
    blocked = ~inside | route.occupied[clipped[:, 0], clipped[:, 1]]
    return points, blocked


def limit_goal(route, xy, target, lookahead=None):
    xy, target = np.asarray(xy, dtype=float), np.asarray(target, dtype=float)
    points, blocked = _ray(route, xy, target)
    if blocked[0]:
        # Route._nearest_free already allows escape from an inflated occupied
        # start. Do not replace the learned escape with an artificial freeze.
        return target, "occupied_start_uncertified"
    hits = np.flatnonzero(blocked)
    if not len(hits):
        return target, "grid_line_free"
    if lookahead is not None and hasattr(route, "point"):
        # Capping the old bearing can leave a tiny goal immediately in front of
        # a wall while the route turns around it. Search the already-computed
        # route, not a new privileged path, for the furthest visible local point.
        start = max(route.progress, route.project(xy))
        distance = min(lookahead, float(np.linalg.norm(target - xy)))
        end = min(start + distance, route.arc[-1])
        count = int(np.ceil(max(0, end - start) / (0.5 * route.res))) + 1
        for arc in np.linspace(end, start, count):
            waypoint, _ = route.point(arc)
            if np.linalg.norm(waypoint - xy) > 0.5 * route.res and not _ray(route, xy, waypoint)[1].any():
                return waypoint, "redirected_to_visible_route"
    return points[max(0, hits[0] - 1)], "limited_at_occupied_grid"
