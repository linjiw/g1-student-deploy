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

    def __init__(self, xml=None, margin_m=0.03):
        import mujoco

        from sim.scene import G1_XML

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml or G1_XML))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.margin = margin_m
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
        d.qpos[:] = qpos36
        self.mujoco.mj_kinematics(self.model, d)
        pos, mat = d.geom_xpos[self.geoms], d.geom_xmat[self.geoms].reshape(-1, 3, 3)
        return (pos + np.einsum("gij,gj->gi", mat, self.a), pos + np.einsum("gij,gj->gi", mat, self.b), self.rad)

    def mask(self, points, qpos36):
        """True for points to keep (not on the robot)."""
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        s, e, rad = self.capsules(qpos36)
        keep = np.ones(len(points), dtype=bool)
        lim = rad + self.margin
        for k in range(len(s)):
            d = e[k] - s[k]
            t = np.clip((points - s[k]) @ d / max(d @ d, 1e-12), 0, 1)
            dist = np.linalg.norm(points - (s[k] + t[:, None] * d), axis=1)
            keep &= dist > lim[k]
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
    """

    def __init__(self, origin_xy, *, source=None, ground_tol_m=0.05, face_push_m=0.02, self_filter=True,
                 self_margin_m=0.03, overhead_min_bottom_m=0.8, min_low_points=3, low_ratio=0.2,
                 free_band_m=(0.1, 0.8), carve_step_m=0.05, carve_stop_m=0.1, edge_cells=2, edge_dz_m=0.05, **kw):
        kw.pop("boxes", None)
        super().__init__([], origin_xy, overhead_min_bottom_m=overhead_min_bottom_m, **kw)
        self.source = source
        self.ground_tol, self.face_push, self.over_min = ground_tol_m, face_push_m, overhead_min_bottom_m
        self.min_low, self.low_ratio, self.free_band = int(min_low_points), float(low_ratio), tuple(free_band_m)
        self.edge_cells, self.edge_dz = int(edge_cells), float(edge_dz_m)
        self.carve_step, self.carve_stop = carve_step_m, carve_stop_m
        self.body_filter = BodyFilter(margin_m=self_margin_m) if self_filter else None
        shape = (self.n * self.n,)
        self.n_low = np.zeros(shape, dtype=np.int32)
        self.zmin = np.full(shape, np.inf)
        self.zmax = np.full(shape, -np.inf)
        self.n_ground = np.zeros(shape, dtype=np.int32)
        self.zmin_high = np.full(shape, np.inf)
        self.free_low = np.zeros(shape, dtype=bool)
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
        """One capture -> evidence arrays (cell indices + values), not yet fused."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        origin = np.asarray(origin, dtype=np.float64)
        rays = pts - origin
        dist = np.linalg.norm(rays, axis=1)
        unit = rays / np.maximum(dist, 1e-9)[:, None]
        # free space: samples along each ray, stopping carve_stop before the return
        steps = np.arange(self.carve_step, max(dist.max(initial=0.0), self.carve_step), self.carve_step)
        samp = origin[None, None] + unit[:, None] * steps[None, :, None]
        valid = steps[None] < (dist[:, None] - self.carve_stop)
        lo, hi = self.free_band
        valid &= (samp[..., 2] >= lo) & (samp[..., 2] < hi)
        fxy = self._apply_pose_error(samp[valid][:, :2], origin)
        fi, fok = self._cells(fxy)
        ground = pts[:, 2] < self.ground_tol
        gi, gok = self._cells(self._apply_pose_error(pts[ground, :2], origin))
        obj, ou = pts[~ground], unit[~ground]
        oi, ook = self._cells(self._apply_pose_error(obj[:, :2] + self.face_push * ou[:, :2], origin))
        return dict(free=np.unique(fi[fok]), ground=gi[gok], obj_idx=oi[ook], obj_z=obj[ook, 2])

    def add_cloud(self, t_ready, points_world, origin_world, qpos36=None):
        """A capture in the gravity-aligned odometry frame (floor at z = 0), usable at `t_ready`."""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        self.stats["points"] += len(pts)
        if self.body_filter is not None and qpos36 is not None and len(pts):
            keep = self.body_filter.mask(pts, qpos36)
            self.stats["self_removed"] += int((~keep).sum())
            pts = pts[keep]
        self.pending.append((t_ready, self.evidence(pts, origin_world)))

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
                self.stats["ground"] += len(ev["ground"])
                touched.append(True)
            else:
                keep.append(item)
        self.pending = keep
        if touched:
            self._derive()

    def _derive(self):
        """Re-derive both layers on the whole grid (241 x 241: ~1 ms)."""
        from scipy.ndimage import maximum_filter

        solid = (self.n_low >= self.min_low) & (self.n_low >= self.low_ratio * self.n_ground)
        high = ~solid & np.isfinite(self.zmin_high)
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
