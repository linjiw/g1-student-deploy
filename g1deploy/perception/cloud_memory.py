"""Label-free lidar memory: the deployable counterpart of flow's `RayCastMemory`.

The Isaac student's memory is written from analytic ray casts that know *which box* each ray
hit: ground → support 0, ground-standing box → support = max hit z, overhead box (bottom ≥
0.8 m) → ceiling = min hit z. A real MID-360 cloud has no such labels and contains the robot's
own arms and shoulders. This module rebuilds the same two layers from raw points only:

1. self-filter: drop points within (radius + `self_margin_m`) of the robot's own collision
   capsules at the capture pose (FK of the measured joints; the planner's capsule set);
2. ground: z < `ground_tol_m` (odometry frame with the floor at z = 0) → support 0;
3. per 5 cm cell, among the remaining points (binned `face_push_m` past the surface along the
   ray, as the labelled memory does): any point below `overhead_min_bottom_m` → solid,
   support = max z of the cell; otherwise → overhead, ceiling = min z of the cell.

Latency, capture rate, fusion (support fmax, ceiling fmin), `ego_map`, `memory_boxes` and the
`HeightFieldSelector` are inherited unchanged, so the planner sees the same interface.

Sources: `MjRaySource` (MuJoCo `mj_multiRay` on the full scene, robot included, same 2°×2°
MID-360 pattern and mount as the Isaac memory) for sim; `add_cloud()` for a real driver.
"""

from __future__ import annotations

import numpy as np

from g1deploy import bundle

_flow = bundle.load_flow()
RayCastMemory = _flow.sensor_map.RayCastMemory


def _quat_matrix(q):
    return _flow.feed.quat_matrix(q)


class BodyFilter:
    """Remove lidar returns on the robot itself.

    One bounding capsule per robot collision geom, fitted offline to the geom's vertices (G1 URDF
    meshes as compiled in the MJCF; primitives exactly), posed by MuJoCo kinematics from the
    measured pelvis pose + joints. The same code runs on the robot PC: it only needs the MJCF and
    the joint state. `margin_m` covers mesh-vs-capsule slack, timing skew and range noise.
    """

    # 5 cm (was 3): the shoulder-pitch links are a sphere (r 3.9 cm) in this model and a cylinder in gear_sonic_sim's
    # scene_43dof.xml (and a mesh on the robot); with 2 cm range noise the cylinder's rim leaked ~1 point per 5 frames
    # past a 3 cm margin, always into the same two cells beside the right shoulder at 1.06 m. The memory keeps them, so
    # after ~15 s of standing the robot had a 1.06 m "pillar" against its own side and the planner found no safe
    # forward candidate (lab twin rehearsals 2026-10-23, tools/debug_plan.py). 4 cm: 0 leaks in 170 noisy frames.
    def __init__(self, xml=None, margin_m=0.05, hand_margin_m=0.04):
        """`xml`: the robot's MJCF - rubber hands (sim/assets/g1_29dof.xml, default) or Dex3
        (sim/assets/g1_29dof_dex3.xml; env G1_BODY_MODEL=dex3). Body joints are set by name; hand joints stay
        at zero (open) and hand links get `hand_margin_m` extra, since the finger state is not streamed."""
        import os

        import mujoco

        from g1deploy import contract as C
        from sim.scene import ASSETS, G1_XML

        if xml is None and os.environ.get("G1_BODY_MODEL", "") == "dex3":
            xml = ASSETS / "g1_29dof_dex3.xml"
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml or G1_XML))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.margin = margin_m
        self.adr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in C.MJ_JOINTS])
        geoms, a, b, rad = [], [], [], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0 or m.geom_contype[g] == 0:
                continue  # world geoms, visual-only copies
            pts = self._geom_points(g)
            c = pts.mean(0)
            u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
            t = (pts - c) @ u
            lo, hi = t.min(), t.max()
            seg_a, seg_b = c + lo * u, c + hi * u
            d = pts - seg_a
            tt = np.clip(d @ u / max(hi - lo, 1e-9), 0, 1)[:, None] * (hi - lo)
            r = np.linalg.norm(d - tt * u, axis=1).max()
            # shrink the axis by r at both ends where the capsule caps already cover the vertices
            shrink = min(r, 0.5 * (hi - lo))
            geoms.append(g)
            a.append(seg_a + shrink * u)
            b.append(seg_b - shrink * u)
            rad.append(np.linalg.norm(pts - self._closest(pts, a[-1], b[-1]), axis=1).max())
        self.geoms, self.a, self.b, self.rad = np.array(geoms), np.array(a), np.array(b), np.array(rad)
        hand = np.array(["hand" in (m.body(m.geom_bodyid[g]).name or "") for g in self.geoms])
        self.rad = self.rad + np.where(hand, hand_margin_m, 0.0)

    @staticmethod
    def _closest(p, a, b):
        d = b - a
        t = np.clip((p - a) @ d / max(d @ d, 1e-12), 0, 1)
        return a + t[:, None] * d

    def _geom_points(self, g):
        m, mj = self.model, self.mujoco
        typ, size = int(m.geom_type[g]), m.geom_size[g]
        T = {k: int(getattr(mj.mjtGeom, k)) for k in ('mjGEOM_MESH', 'mjGEOM_SPHERE', 'mjGEOM_CYLINDER', 'mjGEOM_CAPSULE', 'mjGEOM_BOX')}
        if typ == T['mjGEOM_MESH']:
            i = m.geom_dataid[g]
            return m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(np.float64)
        if typ == T['mjGEOM_SPHERE']:
            return np.array([[0, 0, 0], [size[0], 0, 0], [-size[0], 0, 0], [0, size[0], 0], [0, -size[0], 0],
                             [0, 0, size[0]], [0, 0, -size[0]]], dtype=np.float64)
        if typ in (T['mjGEOM_CYLINDER'], T['mjGEOM_CAPSULE']):
            r, h = size[0], size[1]
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ring = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
            return np.concatenate([np.c_[ring, np.full(16, z)] for z in (-h, h)]).astype(np.float64)
        if typ == T['mjGEOM_BOX']:
            return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
        raise ValueError(f"unsupported robot geom type {typ}")

    def capsules(self, qpos36):
        d = self.data
        d.qpos[:] = 0
        d.qpos[:7] = qpos36[:7]
        d.qpos[self.adr] = qpos36[7:36]
        self.mujoco.mj_kinematics(self.model, d)
        pos, mat = d.geom_xpos[self.geoms], d.geom_xmat[self.geoms].reshape(-1, 3, 3)
        return (pos + np.einsum("gij,gj->gi", mat, self.a), pos + np.einsum("gij,gj->gi", mat, self.b), self.rad)

    def mask(self, points, qpos36, reach_m=1.3, origins=None, along_m=0.06):
        """True for points to keep (not on the robot). Only points within `reach_m` of the pelvis are tested.

        With `origins` (sensor origin per point or one for all), a return is also tested `along_m` further along
        its own ray: range noise (~2 cm sigma) puts ~7% of returns from the robot's arms more than 3 cm in front
        of the surface, outside the capsule margin; with thousands of body hits per frame those leaks become
        overhead "ceilings" along the robot's own path (seen in the realistic MID-360 memory)."""
        points = np.asarray(points, dtype=np.float64)
        keep = np.ones(len(points), dtype=bool)
        if len(points) == 0:
            return keep
        near = np.flatnonzero(np.linalg.norm(points - np.asarray(qpos36[:3]), axis=1) < reach_m)
        if len(near) == 0:
            return keep
        s, e, rad = self.capsules(qpos36)
        tests = [points[near]]
        if origins is not None:
            o = np.broadcast_to(np.asarray(origins, dtype=np.float64), points.shape)[near]
            ray = points[near] - o
            ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9)
            tests += [points[near] + 0.5 * along_m * ray, points[near] + along_m * ray]
        d = e - s
        dd = np.maximum((d * d).sum(1), 1e-12)
        lim = (rad + self.margin)[None]
        inside = np.zeros(len(near), dtype=bool)
        for p in tests:
            t = np.clip(np.einsum("nkj,kj->nk", p[:, None] - s[None], d) / dd, 0, 1)
            dist = np.linalg.norm(p[:, None] - (s[None] + t[..., None] * d[None]), axis=2)
            inside |= (dist <= lim).any(1)
        keep[near] = ~inside
        return keep


class CloudMemory(RayCastMemory):
    """Per-cell evidence -> the planner's support / ceiling layers (see module docstring).

    Evidence per 5 cm cell (world grid shared with the parent class): counts of ground returns
    (n_ground) and of non-ground returns below `overhead_min_bottom_m` (n_low), the highest return
    (zmax) and the lowest return above `overhead_min_bottom_m` (zmin_high), and whether a ray ever
    passed through the column at a height in `free_band_m` (free_low, carving each ray up to
    `carve_stop_m` before its return). Both layers are re-derived after every fusion:
      solid     n_low >= min_low_points and n_low >= low_ratio * n_ground   -> support = zmax
                (fewer low returns are range noise on the ground and are ignored)
      overhead  high returns only, free_low, and no solid cell within `edge_cells` reaching
                zmin_high - edge_dz (else it is the top/edge of that obstacle) -> ceiling = zmin_high
      ambiguous high returns only otherwise -> support = zmax (conservative, revisable)
      ground    otherwise, if any ground return -> support 0
    `min_high_points` > 1 also asks that many high returns of a high-only cell before it counts as overhead or
    ambiguous (one stray return made a ceiling over the lab's furniture). Every low / high return is also counted
    as *far* when it lay more than `near_m` from the sensor horizontally: a cell seen only from closer than that
    (`near_only()`) is where the robot's own leaked returns land (g1deploy/guards.py masks such cells next to the
    body from the planner's view; the memory itself keeps them).
    """

    def __init__(self, origin_xy, *, source=None, ground_tol_m=0.05, face_push_m=0.035, self_filter=True,
                 self_margin_m=0.05, overhead_min_bottom_m=0.8, min_low_points=3, low_ratio=0.2,
                 free_band_m=(0.1, 0.8), carve_step_m=0.05, carve_stop_m=0.1, edge_cells=2, edge_dz_m=0.05,
                 carve_rays=4000, min_high_points=1, near_m=0.6, clear_free=False, clear_min_samples=2, clear_keep=0.5, **kw):
        kw.pop("boxes", None)
        super().__init__([], origin_xy, overhead_min_bottom_m=overhead_min_bottom_m, **kw)
        self.source = source
        self.ground_tol, self.face_push, self.over_min = ground_tol_m, face_push_m, overhead_min_bottom_m
        self.min_low, self.low_ratio, self.free_band = int(min_low_points), float(low_ratio), tuple(free_band_m)
        self.edge_cells, self.edge_dz = int(edge_cells), float(edge_dz_m)
        self.carve_step, self.carve_stop, self.carve_rays = carve_step_m, carve_stop_m, int(carve_rays)
        self.min_high, self.near_m = int(min_high_points), float(near_m)
        # free-space clearing (opt-in, deploy side; the student was trained on a memory that only adds evidence): a
        # cell whose obstacle evidence a newer frame's rays pass through, inside the evidence's height range, with no
        # return in it in that frame, keeps `clear_keep` (half) of its low-obstacle count per such frame. It undoes the smear an odometry
        # error writes into the memory (2026-10-27: a 0.62 m gap closed to 0.45 m on leg odometry), where rays reach.
        self.clear_free, self.clear_min, self.clear_keep = bool(clear_free), int(clear_min_samples), float(clear_keep)
        self.body_filter = BodyFilter(margin_m=self_margin_m) if self_filter else None
        shape = (self.n * self.n,)
        self.n_low = np.zeros(shape, dtype=np.int32)
        self.zmin = np.full(shape, np.inf)
        self.zmax = np.full(shape, -np.inf)
        self.n_ground = np.zeros(shape, dtype=np.int32)
        self.zmin_high = np.full(shape, np.inf)
        self.free_low = np.zeros(shape, dtype=bool)
        self.n_high = np.zeros(shape, dtype=np.int32)
        self.n_low_far = np.zeros(shape, dtype=np.int32)
        self.n_high_far = np.zeros(shape, dtype=np.int32)
        self.stats = {"points": 0, "self_removed": 0, "ground": 0, "solid_cells": 0, "overhead_cells": 0,
                      "ambiguous_cells": 0}

    # -- per-capture evidence -----------------------------------------------------------------
    def _cells(self, xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        ij = np.round((xy - self.lo) / self.res).astype(int)
        ok = (ij >= 0).all(1) & (ij < self.n).all(1)
        return ij[:, 0] * self.n + ij[:, 1], ok

    def _apply_pose_error(self, xy, origin):
        if self.pose_error is None:
            return xy
        dx, dy, dyaw = self.pose_error
        c, s = np.cos(dyaw), np.sin(dyaw)
        rel = np.asarray(xy) - np.asarray(origin[:2])
        return np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], -1) + np.asarray(origin[:2]) + [dx, dy]

    def evidence(self, points, origin):
        """One capture -> evidence arrays (cell indices + values), not yet fused.

        `origin`: the sensor origin, (3,) for an instantaneous capture or (N, 3) per point (a moving sensor)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        o = np.broadcast_to(np.asarray(origin, dtype=np.float64), pts.shape)
        center = o.mean(0) if len(o) else np.zeros(3)
        rays = pts - o
        dist = np.linalg.norm(rays, axis=1)
        unit = rays / np.maximum(dist, 1e-9)[:, None]
        # free space: samples along each ray, stopping carve_stop before the return (a random subset of rays when
        # the frame is dense - a real MID-360 frame has ~15-20k returns and free space is highly redundant)
        c = np.arange(len(pts))
        if len(c) > self.carve_rays:
            c = self.rng.choice(len(pts), self.carve_rays, replace=False)
        steps = np.arange(self.carve_step, max(dist.max(initial=0.0), self.carve_step), self.carve_step)
        samp = o[c][:, None] + unit[c][:, None] * steps[None, :, None]
        valid = steps[None] < (dist[c][:, None] - self.carve_stop)
        lo, hi = self.free_band
        valid &= (samp[..., 2] >= lo) & (samp[..., 2] < hi)
        fxy = self._apply_pose_error(samp[valid][:, :2], center)
        fi, fok = self._cells(fxy)
        free_z = (fi[fok], samp[valid][:, 2][fok]) if self.clear_free else None
        ground = pts[:, 2] < self.ground_tol
        gi, gok = self._cells(self._apply_pose_error(pts[ground, :2], center))
        obj, ou = pts[~ground], unit[~ground]
        oi, ook = self._cells(self._apply_pose_error(obj[:, :2] + self.face_push * ou[:, :2], center))
        far = np.linalg.norm(obj[:, :2] - o[~ground, :2], axis=1) > self.near_m
        return dict(free=np.unique(fi[fok]), ground=gi[gok], obj_idx=oi[ook], obj_z=obj[ook, 2], obj_far=far[ook],
                    free_z=free_z)

    def add_cloud(self, t_ready, points_world, origin_world, qpos36=None):
        """A capture in the gravity-aligned odometry frame (floor at z = 0), usable at `t_ready`."""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        org = np.broadcast_to(np.asarray(origin_world, dtype=np.float64), pts.shape)
        keep = np.linalg.norm(pts - org, axis=1) <= self.range_m  # the evaluated memory saw 4 m
        self.stats["points"] += len(pts)
        if self.body_filter is not None and qpos36 is not None and len(pts):
            body = self.body_filter.mask(pts, qpos36, origins=org)
            self.stats["self_removed"] += int((~body & keep).sum())
            keep &= body
        self.pending.append((t_ready, self.evidence(pts[keep], org[keep])))

    def _capture(self, t, pos, quat):
        self.captures += 1
        if self.capture_drop > 0 and self.rng.random() < self.capture_drop:
            return
        if self.source is None:
            return
        pts, origin, qpos36 = self.source(self)
        self.add_cloud(t + self.latency, pts, origin, qpos36)

    # -- fusion -------------------------------------------------------------------------------
    def step(self, t, pelvis_pos=None, pelvis_quat_wxyz=None):
        if pelvis_pos is not None and t + 1e-9 >= self.next_capture:
            self._capture(t, pelvis_pos, pelvis_quat_wxyz)
            self.next_capture += self.period
        keep, touched = [], []
        for item in self.pending:
            if item[0] <= t + 1e-9:
                ev = item[1]
                self.free_low[ev["free"]] = True
                np.add.at(self.n_ground, ev["ground"], 1)
                oi, oz = ev["obj_idx"], ev["obj_z"]
                low = oz < self.over_min
                np.add.at(self.n_low, oi[low], 1)
                np.minimum.at(self.zmin, oi, oz)
                np.maximum.at(self.zmax, oi, oz)
                np.minimum.at(self.zmin_high, oi[~low], oz[~low])
                np.add.at(self.n_high, oi[~low], 1)
                far = ev.get("obj_far")
                if far is not None:
                    np.add.at(self.n_low_far, oi[low & far], 1)
                    np.add.at(self.n_high_far, oi[~low & far], 1)
                self.stats["ground"] += len(ev["ground"])
                if self.clear_free and ev.get("free_z") is not None:
                    self._clear(ev["free_z"], oi)
                touched.append(True)
            else:
                keep.append(item)
        self.pending = keep
        if touched:
            self._derive()

    def _clear(self, free_z, hit_idx, tol_m=0.05):
        """Free-space clearing of one frame (see __init__): rays through a cell's obstacle height range, no return there,
        at least `clear_min` samples -> the cell keeps `clear_keep` of its low-obstacle count."""
        fi, fz = free_z
        if not len(fi):
            return
        hit = np.zeros(self.n * self.n, dtype=bool)
        hit[hit_idx] = True
        inside = (self.n_low[fi] > 0) & ~hit[fi] & (fz >= self.zmin[fi] - tol_m) & (fz <= self.zmax[fi] + tol_m)
        cells = np.flatnonzero(np.bincount(fi[inside], minlength=self.n * self.n) >= self.clear_min)
        if len(cells):
            self.n_low[cells] = (self.n_low[cells] * self.clear_keep).astype(np.int32)
            self.n_low_far[cells] = (self.n_low_far[cells] * self.clear_keep).astype(np.int32)
            self.n_ground[cells] = np.maximum(self.n_ground[cells], 1)  # seen through at floor height: open floor
            self.stats["cleared"] = self.stats.get("cleared", 0) + len(cells)

    def _derive(self):
        """Re-derive both layers on the whole grid (241 x 241: ~1 ms)."""
        from scipy.ndimage import maximum_filter

        solid = (self.n_low >= self.min_low) & (self.n_low >= self.low_ratio * self.n_ground)
        high = ~solid & np.isfinite(self.zmin_high) & (self.n_high >= self.min_high)
        # a high-only cell next to a solid cell reaching its height is the top/edge of that obstacle
        solid_top = np.where(solid, self.zmax, -np.inf).reshape(self.n, self.n)
        near = maximum_filter(solid_top, size=2 * self.edge_cells + 1, mode="constant", cval=-np.inf).reshape(-1)
        attached = high & (near >= self.zmin_high - self.edge_dz)
        over = high & self.free_low & ~attached
        amb = high & ~over
        sup = np.where(self.n_ground > 0, 0.0, np.nan)
        sup = np.where(solid | amb, self.zmax, sup)
        self.support.reshape(-1)[:] = sup
        self.ceiling.reshape(-1)[:] = np.where(over, self.zmin_high, np.nan)
        self.stats["solid_cells"] = int(solid.sum())
        self.stats["overhead_cells"] = int(over.sum())
        self.stats["ambiguous_cells"] = int(amb.sum())

    def near_only(self):
        """(n, n) True where the cell has obstacle evidence but none of it was seen from farther than `near_m`."""
        seen = (self.n_low > 0) | (self.n_high > 0)
        return (seen & (self.n_low_far == 0) & (self.n_high_far == 0)).reshape(self.n, self.n)


class MjRaySource:
    """MID-360 pattern ray-cast in MuJoCo against every collision geom (robot included)."""

    def __init__(self, sim, range_noise_m=0.02, ray_dropout=0.0, rng=None):
        import mujoco

        self.mujoco, self.sim = mujoco, sim
        self.noise, self.dropout = range_noise_m, ray_dropout
        self.rng = rng or np.random.default_rng(0)
        self.group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # collision geoms only (visual meshes are group 1)

    def __call__(self, memory):
        d = self.sim.data
        pos, quat = d.qpos[0:3].copy(), d.qpos[3:7].copy()
        r_base = _quat_matrix(quat)
        all_pts, all_geom, origin = [], [], None
        for mount, d_sensor, rmax in memory.rays:
            origin = pos + r_base @ np.asarray(mount.pos_pelvis)
            dirs = d_sensor @ (r_base @ mount.rot_pelvis).T
            n = len(dirs)
            geom = np.zeros(n, dtype=np.int32)
            dist = np.zeros(n)
            self.mujoco.mj_multiRay(self.sim.model, d, origin, dirs.reshape(-1).astype(np.float64), self.group, 1, -1,
                                    geom, dist, None, n, float(rmax))
            hit = (geom >= 0) & (dist > 0.1) & (dist <= rmax)  # MID-360 minimum range 0.1 m
            if self.dropout > 0:
                hit &= self.rng.random(n) >= self.dropout
            tt = dist[hit] + self.rng.normal(0.0, self.noise, hit.sum())
            all_pts.append(origin + dirs[hit] * tt[:, None])
            all_geom.append(geom[hit])
        self.last_geom = np.concatenate(all_geom)  # validation only: which geom each point came from
        return np.concatenate(all_pts), origin, d.qpos.copy()


def MjRayMemory(sim, origin_xy, rng=None, **sensor_kw):
    """CloudMemory fed by MuJoCo ray casting (the sim stand-in for the real MID-360 driver)."""
    rng = rng or np.random.default_rng(0)
    mem = CloudMemory(origin_xy, rng=rng, **sensor_kw)
    mem.source = MjRaySource(sim, range_noise_m=mem.range_noise, ray_dropout=mem.ray_dropout, rng=rng)
    return mem
