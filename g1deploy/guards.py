"""Deploy-side guards between the lidar memory, the frozen planner and SONIC (opt-in: `--guards deploy`).

The lab findings of 2026-10-23 (docs/ROBUSTNESS.md) traced the worst failures to the step where the
real robot's memory reaches the planner, not to the student itself:

  self mask      two leaked self-return cells beside the shoulder made every candidate unsafe, standing
                 still included. Evidence blobs lying entirely within `self_mask_band_m` of the planner's own
                 capsules at the measured pose, and only ever seen from closer than `near_m` to the sensor, are
                 hidden from the planner's view (map input, guidance boxes, selection) - at most
                 `self_mask_max_cells` cells; more is refused and reported (an object hugging the robot, a
                 site-check NO-GO). A blob that reaches outside the band (a wall beside the robot) stays. The
                 memory keeps everything; the stand deferral and the site check see it.
  split boxes    `memory_boxes` wraps each connected component in one axis-aligned box: a closed room
                 became one solid block with the robot inside. Guidance boxes here are rectangles of a
                 component's own cells on 10 cm cells, merged while >= `split_fill` occupied (so up to 20 % of
                 a box can be free floor), nearest first, up to `max_guide_boxes`; a component whose boxes
                 would contain the robot is covered exactly instead.
  walk envelope  a one-legged lunge (knee 2.1 rad) and open-floor crouches passed the node's checks.
                 With no ceiling within reach, candidates must keep the pelvis, knees and tilt inside the
                 envelope measured on clean walks; the planner's own preference picks among those that do.
  calm fallback  with no safe candidate, the frozen rule streams the highest-clearance candidate of any
                 pool: a 0.8-1.2 m motion from a standstill, then a crouch. From a standstill here it is the
                 best stop-pool candidate that stays near, inside the envelope and at least as clear as
                 standing still (never predicted contact), else a settle-to-stand from the anchor. While
                 walking, the frozen choice stays if it is inside the envelope.
  anchor check   the next plan is anchored on the measured state, not a prediction, after a calm plan or
                 when a predicted anchor missed the measured pose by more than `anchor_check_m`.
  overhead walls (H5-H6 only, `--guards walls`) every remembered overhead cell lower than standing head
                 height is a wall: the planner routes around tables and beams instead of ducking.

The frozen planner, its model and its random stream are unchanged; the self mask, walls and split boxes change
the map and guidance it is given, so its candidates differ. A safe frozen choice inside the envelope is returned
byte-equal.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from g1deploy import contract as C

DT = 0.02
KNEES = (C.MJ_JOINTS.index("left_knee_joint"), C.MJ_JOINTS.index("right_knee_joint"))
CALM_RULES = ("calm_stop", "calm_stand")
FALLBACK_RULES = ("stop_max_clearance", "fallback_go_max_clearance", "forward_max_clearance", "max_clearance") + CALM_RULES


@dataclasses.dataclass(frozen=True)
class GuardConfig:
    self_mask: bool = False
    self_mask_band_m: float = 0.10  # leaks: filter margin 3-5 cm + 2 cm range noise + 3.5 cm face push
    self_mask_max_cells: int = 8  # the lab twin leaked 2 cells; more near-only cells that close is a real object
    split_boxes: bool = False
    # Nearest boxes kept for guidance. 16 = the fast planner's middle graph bucket: on the authored dense room and
    # closed room the nearest occupied cell left out was 1.8-2.0 m away with 16 or 32 (single-cell specks are
    # dropped, as by memory_boxes), and guidance cost grows with the bucket.
    max_guide_boxes: int = 16
    split_fill: float = 0.8  # a guidance box is at least this fraction occupied cells
    split_coarse: int = 2  # decompose on 10 cm cells (guidance inflates every box by 8 cm anyway)
    guide_range_m: float = 3.0  # components nearer than this are always decomposed before the nearest-box cap
    envelope: bool = False
    # Envelope from 192 plans of clean authored walks (tools/robust_panel.py, kit and tag conditions, 2026-10-23):
    # pelvis min 0.65 m, knee max 1.94 rad, planned tilt p99 0.44 / max 0.56 rad. The failures it must stop:
    # crouches to 0.44-0.58 m, lunges with a 2.1-2.9 rad knee, plans tilted 0.83-0.96 rad before a fall.
    min_pelvis_z_m: float = 0.60
    max_knee_rad: float = 2.05
    max_tilt_rad: float = 0.60
    ceiling_reach_m: float = 2.0  # the envelope is waived when a ceiling below `ceiling_below_m` is this near
    ceiling_below_m: float = 1.45
    envelope_tries: int = 6
    calm: bool = False
    still_speed_m_s: float = 0.25
    calm_turn_max_m: float = 0.25  # a stop-pool candidate from a standstill may move the pelvis this far
    anchor_check_m: float | None = None
    anchor_hold_plans: int = 2
    overhead_walls: bool = False
    wall_top_m: float = 1.5
    min_high_points: int | None = None  # CloudMemory: high returns a cell needs before it is overhead / ambiguous

    def any(self):
        return self != GuardConfig()

    def describe(self):
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


# anchor check: clean walks miss their predicted anchor by 0.09-0.26 m (median), at most 0.5 m; the walks that fell
# in boxed-in rooms missed by 0.5-1.6 m.
DEPLOY = GuardConfig(self_mask=True, split_boxes=True, envelope=True, calm=True, anchor_check_m=0.4)
PRESETS = {"off": GuardConfig(), "deploy": DEPLOY, "walls": dataclasses.replace(DEPLOY, overhead_walls=True)}


def preset(name_or_cfg):
    if name_or_cfg is None:
        return GuardConfig()
    if isinstance(name_or_cfg, GuardConfig):
        return name_or_cfg
    if isinstance(name_or_cfg, dict):
        return dataclasses.replace(PRESETS[name_or_cfg.get("preset", "off")],
                                   **{k: v for k, v in name_or_cfg.items() if k != "preset"})
    return PRESETS[name_or_cfg]


# ------------------------------------------------------------------------------------------- memory view
class MemoryView:
    """Picklable copy of the memory layers with the frozen `SensorMapMemory.ego_map` semantics."""

    def __init__(self, support, ceiling, lo, res, n, near_only=None):
        self.support, self.ceiling, self.lo, self.res, self.n = support, ceiling, lo, res, n
        self.near_only = near_only

    def ego_map(self, xy, yaw, datum_offset_m, support_clip_m=0.8, obstacle_threshold_m=0.05):
        from g1deploy import bundle

        sm = bundle.load_flow().sensor_map
        return sm.SensorMapMemory.ego_map(self, xy, yaw, datum_offset_m, support_clip_m, obstacle_threshold_m)


def memory_view(memory, near_only=True):
    near = memory.near_only().copy() if near_only and hasattr(memory, "near_only") else None
    return MemoryView(memory.support.copy(), memory.ceiling.copy(), np.asarray(memory.lo).copy(), memory.res,
                      memory.n, near)


def _seg_dist_xy(p, a, b):
    """Horizontal distance of points p (M, 2) to segments a-b (S, 2) -> (M, S)."""
    d = b - a
    t = np.clip(np.einsum("msj,sj->ms", p[:, None] - a[None], d) / np.maximum((d * d).sum(1), 1e-12)[None], 0, 1)
    return np.linalg.norm(p[:, None] - (a[None] + t[..., None] * d[None]), axis=2)


def body_cells(planner, qpos, view, band_m):
    """(n, n) cells whose centre is within `band_m` (horizontally) of a planner capsule's surface for the robot posed
    at `qpos`, allowing for the cell size."""
    from g1deploy import bundle

    bundle.load_flow()
    from motion2scene_flow.native_geometry import world_capsules

    q = np.asarray(qpos, dtype=np.float64)[None].copy()
    p, r = planner.certifier.tree.forward(q)
    s, e, rad = world_capsules(p, r, planner.certifier.bodies, planner.shapes)
    s, e = s[0], e[0]
    reach = rad + band_m + view.res * math.sqrt(0.5)
    lo, res, n = view.lo, view.res, view.n
    k = int(math.ceil((1.0 + reach.max()) / res))
    ij0 = np.round((np.asarray(q[0, :2]) - lo) / res).astype(int)
    a0, a1 = max(ij0[0] - k, 0), min(ij0[0] + k + 1, n)
    b0, b1 = max(ij0[1] - k, 0), min(ij0[1] + k + 1, n)
    out = np.zeros((n, n), dtype=bool)
    if a0 >= a1 or b0 >= b1:
        return out
    gi, gj = np.meshgrid(np.arange(a0, a1), np.arange(b0, b1), indexing="ij")
    centers = np.stack([gi, gj], -1).reshape(-1, 2) * res + lo
    out[a0:a1, b0:b1] = (_seg_dist_xy(centers, s[:, :2], e[:, :2]) <= reach[None]).any(1).reshape(gi.shape)
    return out


def self_mask(view, planner, measured_q, cfg):
    """Hide near-only cells hugging the body at the measured pose from the view. Returns (view, info)."""
    if view.near_only is None or measured_q is None:
        return view, {}
    from scipy.ndimage import label

    band = body_cells(planner, measured_q, view, cfg.self_mask_band_m)
    cand = band & view.near_only
    if not cand.any():
        return view, {}
    # A leak is an isolated blob hugging the body. A component of evidence that reaches outside the band, or that
    # holds a cell once seen from farther away, is a real object (a wall the robot stands beside) and stays.
    occ = (np.nan_to_num(view.support, nan=0.0) > 0.05) | (np.isfinite(view.ceiling) & (view.ceiling < 1e2)) | cand
    lab, _ = label(occ, structure=np.ones((3, 3), dtype=bool))
    ids = np.unique(lab[cand])
    bad = np.unique(lab[(lab > 0) & ~(band & view.near_only)])
    hide = cand & np.isin(lab, np.setdiff1d(ids, bad))
    count = int(hide.sum())
    if not count:
        return view, {}
    if count > cfg.self_mask_max_cells:
        return view, {"self_mask_refused": count}
    sup, ceil = view.support.copy(), view.ceiling.copy()
    sup[hide & np.isfinite(sup)] = 0.0  # the robot stands there: ground
    ceil[hide] = np.inf  # seen open (NaN would let overhead growth re-cover it)
    return MemoryView(sup, ceil, view.lo, view.res, view.n, view.near_only), {"self_masked_cells": count}


def overhead_walls(view, cfg):
    """Every remembered underside below standing head height becomes a wall `wall_top_m` tall."""
    low = np.isfinite(view.ceiling) & (view.ceiling < cfg.ceiling_below_m)
    if not low.any():
        return view, 0
    sup, ceil = view.support.copy(), view.ceiling.copy()
    sup[low] = cfg.wall_top_m
    ceil[low] = np.nan
    return MemoryView(sup, ceil, view.lo, view.res, view.n, view.near_only), int(low.sum())


# ------------------------------------------------------------------------------------------ guidance boxes
def _rectangles(mask):
    """Exact cover of a boolean grid by rectangles: runs per row, merged down identical columns."""
    rects, open_ = [], {}
    for i in range(mask.shape[0] + 1):
        runs = set()
        if i < mask.shape[0]:
            row = np.r_[False, mask[i], False]
            d = np.flatnonzero(np.diff(row.astype(np.int8)))
            runs = {(int(d[k]), int(d[k + 1]) - 1) for k in range(0, len(d), 2)}
        for run in list(open_):
            if run not in runs:
                rects.append((open_.pop(run), i - 1) + run)
        for run in runs:
            open_.setdefault(run, i)
    return rects  # (i0, i1, j0, j1) inclusive


def _merge(rects, mask, fill):
    """Greedily merge touching rectangles while the merged bounds stay at least `fill` occupied (lidar edges are
    ragged: an exact cover of a straight wall is dozens of slivers). Vectorized over all pairs per merge."""
    csum = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r = np.array(rects, dtype=np.int64).reshape(-1, 4)
    while len(r) > 1:
        a, b = np.triu_indices(len(r), 1)
        ra, rb = r[a], r[b]
        touch = ~((ra[:, 0] > rb[:, 1] + 1) | (rb[:, 0] > ra[:, 1] + 1) | (ra[:, 2] > rb[:, 3] + 1) | (rb[:, 2] > ra[:, 3] + 1))
        if not touch.any():
            break
        a, b, ra, rb = a[touch], b[touch], ra[touch], rb[touch]
        u = np.stack([np.minimum(ra[:, 0], rb[:, 0]), np.maximum(ra[:, 1], rb[:, 1]),
                      np.minimum(ra[:, 2], rb[:, 2]), np.maximum(ra[:, 3], rb[:, 3])], 1)
        occ = csum[u[:, 1] + 1, u[:, 3] + 1] - csum[u[:, 0], u[:, 3] + 1] - csum[u[:, 1] + 1, u[:, 2]] + csum[u[:, 0], u[:, 2]]
        f = occ / ((u[:, 1] - u[:, 0] + 1) * (u[:, 3] - u[:, 2] + 1))
        k = int(np.argmax(f))
        if f[k] < fill:
            break
        keep = np.ones(len(r), dtype=bool)
        keep[[a[k], b[k]]] = False
        r = np.vstack([r[keep], u[k][None]])
    return [tuple(int(x) for x in row) for row in r]


def split_boxes(view, xy, cfg, grow_overhead_m=0.0, obstacle_threshold_m=0.05, min_cells=2, ceiling_thickness_m=0.1):
    """`memory_boxes` that never covers a free cell: each connected component as well-filled rectangles of its own
    cells, components nearest to `xy` first, at most `max_guide_boxes` boxes. Overhead fragments keep the frozen
    growth into unseen cells."""
    from scipy.ndimage import binary_dilation, find_objects, label

    f = max(int(cfg.split_coarse), 1)  # boxes on an f x coarser grid: f x f max-pooled (higher tops, lower undersides)
    sup = np.nan_to_num(view.support, nan=0.0)
    ceil = view.ceiling
    if f > 1:  # the frozen `min_cells` rule on the fine grid first: specks are dropped, not pooled into boxes
        sup, ceil = sup.copy(), ceil.copy()
        for occ, layer, empty in ((sup > obstacle_threshold_m, sup, 0.0), (np.isfinite(ceil) & (ceil < 1e2), ceil, np.nan)):
            lab, _ = label(occ)
            size = np.bincount(lab.reshape(-1))
            layer[(size[lab] < min_cells) & occ] = empty
        min_cells = 1
    lo0, res = np.asarray(view.lo, dtype=float), view.res * f
    if f > 1:
        n = -(-sup.shape[0] // f) * f
        pad = lambda a, v: np.pad(a, ((0, n - a.shape[0]), (0, n - a.shape[1])), constant_values=v)
        sup = pad(sup, 0.0).reshape(n // f, f, n // f, f).max((1, 3))
        cl = pad(ceil, np.nan).reshape(n // f, f, n // f, f)
        seen_c = np.isfinite(cl)
        ceil = np.where(seen_c.any((1, 3)), np.nanmin(np.where(seen_c, cl, np.inf), axis=(1, 3)),
                        np.where(np.isnan(cl).all((1, 3)), np.nan, np.inf))
        lo0 = lo0 + 0.5 * view.res * (f - 1)  # coarse cell centres
    xy_ij = (np.asarray(xy, dtype=float) - lo0) / res
    comps = []
    for occ, kind in ((sup > obstacle_threshold_m, "ground"), (np.isfinite(ceil) & (ceil < 1e2), "overhead")):
        lab, n = label(occ)
        for k, sl in enumerate(find_objects(lab), 1):
            gap = np.maximum(np.maximum(np.array([sl[0].start, sl[1].start]) - xy_ij,
                                        xy_ij - np.array([sl[0].stop - 1, sl[1].stop - 1])), 0)
            comps.append((float(np.linalg.norm(gap)) * res, kind, lab, k, sl))
    comps.sort(key=lambda c: c[0])
    boxes = []
    for dist, kind, lab, k, sl in comps:
        if dist > cfg.guide_range_m and len(boxes) >= cfg.max_guide_boxes:
            break
        grow = int(round(grow_overhead_m / res)) if kind == "overhead" and grow_overhead_m > 0 else 0
        w = (slice(max(sl[0].start - grow, 0), sl[0].stop + grow), slice(max(sl[1].start - grow, 0), sl[1].stop + grow))
        win_o = np.array([w[0].start, w[1].start])
        seen_w = lab[w] == k
        if seen_w.sum() < min_cells:
            continue
        comp_w = seen_w
        if grow:
            comp_w = binary_dilation(seen_w, iterations=grow, mask=seen_w | np.isnan(ceil[w]))
        idx = np.argwhere(comp_w)
        o = idx.min(0) + win_o
        sub = comp_w[idx[:, 0].min():idx[:, 0].max() + 1, idx[:, 1].min():idx[:, 1].max() + 1]
        seen = np.zeros(sup.shape, dtype=bool)
        seen[w] = seen_w
        rects = [(0, sub.shape[0] - 1, 0, sub.shape[1] - 1)] if sub.mean() >= cfg.split_fill \
            else _merge(_rectangles(sub), sub, cfg.split_fill)
        rob = np.round(xy_ij).astype(int) - o
        holds = lambda r: r[0] <= rob[0] <= r[1] and r[2] <= rob[1] <= r[3]  # noqa: E731
        if any(holds(r) for r in rects) and not (0 <= rob[0] < sub.shape[0] and 0 <= rob[1] < sub.shape[1] and sub[tuple(rob)]):
            rects = _merge(_rectangles(sub), sub, 1.0)  # the robot stands in free floor: exact cover around it
        part_boxes = []
        for i0, i1, j0, j1 in rects:
            cells = (slice(o[0] + i0, o[0] + i1 + 1), slice(o[1] + j0, o[1] + j1 + 1))
            lo = np.array([o[0] + i0, o[1] + j0]) * res + lo0 - res / 2
            hi = np.array([o[0] + i1, o[1] + j1]) * res + lo0 + res / 2
            box = {"center_xy": ((lo + hi) / 2).tolist(), "half_xy": ((hi - lo) / 2).tolist(), "yaw": 0.0}
            part = seen[cells]
            if kind == "ground":
                box.update(z_bottom=0.0, z_top=float(sup[cells][part].max() if part.any() else sup[seen].max()))
            else:
                bottom = float(ceil[cells][part].min() if part.any() else ceil[seen].min())
                box.update(z_bottom=bottom, z_top=bottom + ceiling_thickness_m)
            gap = np.maximum(np.abs(np.asarray(xy) - np.asarray(box["center_xy"])) - np.asarray(box["half_xy"]), 0)
            part_boxes.append((float(np.linalg.norm(gap)), box))
        boxes += part_boxes
    boxes.sort(key=lambda b: b[0])
    return [b for _, b in boxes[:cfg.max_guide_boxes]]


# ------------------------------------------------------------------------------------------ plans
def stand_plan(q, n, settle_s=1.0):
    """Settle-to-stand from world qpos `q`: joints -> SONIC default pose over `settle_s` (smoothstep), pelvis xy,
    height and yaw kept, level; `n` frames (frame i at anchor + (i+1) dt)."""
    q = np.asarray(q, dtype=np.float64)
    n_settle = max(int(settle_s / DT), 1)
    a = np.clip(np.arange(1, n + 1) / n_settle, 0.0, 1.0)[:, None]
    a = a * a * (3 - 2 * a)
    frames = np.repeat(q[None], n, 0)
    frames[:, 7:] = (1 - a) * q[7:] + a * C.DEFAULT_MJ
    w, x, y, z = q[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    frames[:, 3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    dq = np.vstack([(frames[0, 7:] - q[7:])[None] / DT, np.diff(frames[:, 7:], axis=0) / DT])
    return {"qpos": frames, "rotation": np.repeat(rot[None], n, 0), "joint_velocity": dq}


def candidate_metrics(planner, pools):
    """Per candidate (all pools, global order): physical pelvis z min, knee max, planned tilt max, end offset."""
    dz = planner.cfg.datum_offset_m
    q = np.concatenate([p["qpos"] for p, _, _ in pools])
    rot = np.concatenate([p["rot"].cpu().numpy() if hasattr(p["rot"], "cpu") else np.asarray(p["rot"]) for p, _, _ in pools])
    zmin = q[:, :, 2].min(1) - dz
    knee = q[:, :, [7 + KNEES[0], 7 + KNEES[1]]].max((1, 2))
    tilt = np.arccos(np.clip(rot[:, :, 2, 2], -1, 1)).max(1)
    return dict(zmin=zmin, knee=knee, tilt=tilt, xy=q[:, :, :2])


def materialize(planner, pools, choice):
    """The plan arrays of global candidate `choice`, exactly as the frozen planner materializes its choice."""
    import motion2scene_flow.closed_loop.planner as P

    offset = 0
    for cand, clearance, kind in pools:
        if choice < offset + len(clearance):
            local = choice - offset
            break
        offset += len(clearance)
    q = np.array(cand["qpos"][local], dtype=np.float64)
    q[:, 2] -= planner.cfg.datum_offset_m
    states = cand["states"][local].cpu().numpy()
    jvel = states[:, 36:65]
    lift = None
    if planner.cfg.ground_legs:
        if type(planner).__module__ == "g1deploy.fastplan":  # the fast planner grounds with its bitwise-equal copy
            from g1deploy.fastplan import ground_legs
        else:
            from motion2scene_flow.closed_loop.grounding import ground_legs

        q, lift = ground_legs(q, planner.certifier.tree, planner.certifier.bodies, planner.cfg.stance_ankle_z_m, planner.limits)
        jvel = np.concatenate([np.diff(q[:, 7:], axis=0), np.diff(q[-2:, 7:], axis=0)]) / P.DT
    rot = cand["rot"][local]
    return dict(qpos=q, rotation=rot.cpu().numpy() if hasattr(rot, "cpu") else np.asarray(rot), joint_velocity=jvel,
                states=states, choice=int(choice), candidate_pool=kind, chosen_clearance_m=float(clearance[local]),
                ground_lift_max_m=None if lift is None else float(lift.max()))


def ceiling_near(view, xy, cfg):
    ij = np.round((np.asarray(xy) - view.lo) / view.res).astype(int)
    k = int(math.ceil(cfg.ceiling_reach_m / view.res))
    a0, a1 = max(ij[0] - k, 0), min(ij[0] + k + 1, view.n)
    b0, b1 = max(ij[1] - k, 0), min(ij[1] + k + 1, view.n)
    win = view.ceiling[a0:a1, b0:b1]
    gi, gj = np.ogrid[a0 - ij[0]:a1 - ij[0], b0 - ij[1]:b1 - ij[1]]
    disc = (gi * gi + gj * gj) <= k * k
    return bool(np.any(np.isfinite(win) & (win < cfg.ceiling_below_m) & disc))


def plan_metrics(plan):
    """Streamed-plan envelope quantities: physical pelvis z min, knee max, planned pelvis tilt max."""
    q = np.asarray(plan["qpos"])
    rot = np.asarray(plan["rotation"])
    return dict(zmin=float(q[:, 2].min()), knee=float(q[:, [7 + KNEES[0], 7 + KNEES[1]]].max()),
                tilt=float(np.arccos(np.clip(rot[:, 2, 2], -1, 1)).max()))


def inside(met, cfg):
    return met["zmin"] >= cfg.min_pelvis_z_m and met["knee"] <= cfg.max_knee_rad and met["tilt"] <= cfg.max_tilt_rad


def stand_clearance(planner, plan, clearance_fn):
    """The selector's clearance of a (physical-height) plan, e.g. the synthesized stand; None without a selector."""
    if clearance_fn is None:
        return None
    q = np.array(plan["qpos"], dtype=np.float64)
    q[:, 2] += planner.cfg.datum_offset_m  # the selector takes planner-datum heights
    return float(clearance_fn(q[None], planner.cfg.datum_offset_m)[0])


def guard_selection(planner, out, qpos_hist, view, cfg, *, standing, clearance_fn=None):
    """Envelope + calm fallback over the planner's own pools (see module docstring). Returns the plan.

    The envelope is checked on materialized plans (leg grounding changes the knees): the frozen choice first, then
    the planner's own preference order, at most `envelope_tries` materializations. A calm stop must be at least as
    clear as standing still and never predicted in contact (clearance >= 0); otherwise the robot settles to stand."""
    pools = getattr(planner, "last_pools", None)
    info = {"no_safe": bool(out.get("no_safe_candidate")), "frozen_rule": out["rule"]}
    if pools is None:
        return dict(out, guard=info)
    anchor = np.asarray(qpos_hist[-1], dtype=np.float64)
    env_on = cfg.envelope and not ceiling_near(view, anchor[:2], cfg)
    info["envelope"] = bool(env_on)
    clearance = np.concatenate([c for _, c, _ in pools])
    kinds = np.concatenate([np.repeat(kind, len(c)) for _, c, kind in pools])
    safe = clearance >= planner.cfg.select_margin_m
    choice = int(out["choice"])
    met = plan_metrics(out)
    info["metrics"] = met
    ok_choice = not env_on or inside(met, cfg)
    if ok_choice and (safe[choice] or (cfg.calm and not standing) or not cfg.calm):
        return dict(out, guard=info)  # the frozen selection, unchanged (walking: also a frozen fallback)
    if not ok_choice:
        info["rejected"] = _why(met, cfg)
    m = candidate_metrics(planner, pools)  # raw candidates: tilt and pelvis height survive leg grounding
    raw_ok = (m["tilt"] <= cfg.max_tilt_rad) & (m["zmin"] >= cfg.min_pelvis_z_m) if env_on else np.ones(len(safe), bool)
    tries = 0
    if safe.any():  # the planner's own preference (end nearest its local goal) among safe candidates
        remaining = np.asarray(out["remaining_m"])
        good = safe & raw_ok & (kinds != "stop")
        order = [int(i) for i in np.flatnonzero(good)[np.argsort(remaining[good])] if int(i) != choice]
        for i in order[:cfg.envelope_tries]:
            tries += 1
            cand = materialize(planner, pools, i)
            if not env_on or inside(plan_metrics(cand), cfg):
                info.update(action="envelope_reselect", tries=tries)
                return dict(out, **cand, rule=out["rule"] + "+envelope", guard=info)
    if not cfg.calm:
        info.update(action="envelope_reject")
        return dict(out, guard=info, rejected=info.get("rejected", "unsafe"))
    # calm: nothing safe and trackable to walk with
    n = len(out["qpos"])
    stand = stand_plan(anchor, n)
    stand_cl = stand_clearance(planner, stand, clearance_fn)
    info["stand_clearance_m"] = stand_cl
    floor = max(0.0, stand_cl) if stand_cl is not None else 0.0
    move = np.linalg.norm(m["xy"][:, -1] - anchor[:2], axis=1)
    stop = (kinds == "stop") & raw_ok & (clearance >= floor)
    if standing:
        stop &= move <= cfg.calm_turn_max_m
    for i in sorted(np.flatnonzero(stop), key=lambda i: (-clearance[i], move[i]))[:cfg.envelope_tries]:
        cand = materialize(planner, pools, int(i))
        if not env_on or inside(plan_metrics(cand), cfg):
            info.update(action="calm_stop", standing=bool(standing))
            return dict(out, **cand, rule="calm_stop", guard=info)
    info.update(action="calm_stand", standing=bool(standing))
    return dict(out, **stand, choice=-1, candidate_pool="stand",
                chosen_clearance_m=float("nan") if stand_cl is None else stand_cl, rule="calm_stand", guard=info)


def _why(met, cfg):
    out = []
    if met["zmin"] < cfg.min_pelvis_z_m:
        out.append(f"pelvis {met['zmin']:.2f} m")
    if met["knee"] > cfg.max_knee_rad:
        out.append(f"knee {met['knee']:.2f} rad")
    if met["tilt"] > cfg.max_tilt_rad:
        out.append(f"tilt {met['tilt']:.2f} rad")
    return ", ".join(out) or "unsafe"


def speed_xy(hist):
    h = np.asarray(hist, dtype=np.float64)
    if len(h) < 2:
        return 0.0
    return float(np.linalg.norm(h[-1, :2] - h[0, :2]) / (DT * (len(h) - 1)))


def plan_guarded(planner, hist, boxes, goal_fn, view, cfg, *, measured_q=None, standing=False, prev=None,
                 guide_grow_overhead_m=0.0):
    """One plan through the guards: view -> (self mask, overhead walls) -> map / guidance / selector -> frozen
    planner -> envelope + calm. `view`: a MemoryView (memory_view(memory))."""
    from g1deploy import bundle

    import time

    sm = bundle.load_flow().sensor_map
    pc = planner.cfg
    info = {}
    t0 = time.perf_counter()
    if cfg.self_mask:
        view, mask_info = self_mask(view, planner, measured_q, cfg)
        info.update(mask_info)
    if cfg.overhead_walls:
        view, info["overhead_wall_cells"] = overhead_walls(view, cfg)
    sel = sm.HeightFieldSelector(view, planner.certifier.tree, planner.certifier.bodies, planner.shapes)

    def map_fn(xy, yaw):
        m, _ = view.ego_map(xy, yaw, pc.datum_offset_m, pc.support_clip_m or 0.8)
        return m

    guide = None
    if pc.guide_w > 0:
        if cfg.split_boxes:
            guide = split_boxes(view, np.asarray(hist[-1])[:2], cfg, grow_overhead_m=guide_grow_overhead_m)
        else:
            guide = sm.memory_boxes(view, grow_overhead_m=guide_grow_overhead_m)
        info["guide_boxes"] = len(guide)
    t1 = time.perf_counter()
    out = planner.plan(np.asarray(hist), boxes, goal_fn, map_fn, sel, prev=prev, guide_boxes=guide)
    t2 = time.perf_counter()
    if cfg.envelope or cfg.calm:
        out = guard_selection(planner, out, hist, view, cfg, standing=standing, clearance_fn=sel)
    info["ms"] = {"view": round(1e3 * (t1 - t0), 1), "plan": round(1e3 * (t2 - t1), 1),
                  "select": round(1e3 * (time.perf_counter() - t2), 1)}
    out.setdefault("guard", {}).update(info)
    return out
