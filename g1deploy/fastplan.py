"""Faster execution of the frozen student planner: same sampler, same random stream, same selection.

`FastFlowPlanner` subclasses the bundle's `FlowPlanner` (whose source stays byte-identical) and changes how
the work is issued, not what is planned. Measured on the deploy PC under full GPU contention, interleaved
(tools/bench_planner.py): p95 0.37-0.39 s (eager) -> 0.24 s (graph) -> 0.16-0.17 s (fused, deployed);
worst case (every pool unsafe: 16 + 48 + 16 candidates) 1.05 s -> 0.38 s. Same candidate and rule chosen
in every compared plan. Design notes: docs/PLANNER_SPEED.md.

1. CUDA graphs ("graph"). The guided Euler sampler (per chunk 10 x [model forward + clearance-gradient
   backward], ~11k small kernels) is captured once per (candidates, obstacle bucket) and replayed: one
   launch per chunk, so per-kernel launch overhead and per-kernel queueing behind other GPU processes go.
   Noise is drawn from the planner's generator outside the graph (the original random stream). Guidance
   obstacles are padded to the bucket with slots whose `valid` flag is false; `batched_point_distance`
   turns those into +inf, which leaves every minimum (so the penetration and its gradient) unchanged.
2. Fused ("fused", default). Inside the same graphs: everything in PassageExpert.forward that does not
   depend on x or t (map CNN, conditioning MLPs, each block's modulation, cross-attention keys/values) is
   computed once per chunk instead of once per Euler step, and the per-step token path and the guidance
   objective run as torch.compile'd (inductor-fused) kernels. Same layers and formulas; float32 results
   differ at the 1e-5 level, which is the eager planner's own run-to-run spread (atomic accumulation in
   index_select's backward).
3. Host side (both). One host sync per chunk for the candidates' anchors instead of one per candidate,
   one transfer of the finished candidates instead of five; height-field selection and leg grounding call
   the numpy FK once on all candidates x frames (once per Gauss-Newton iteration on the three stacked
   configurations) with per-joint constants cached: the same float64 operations in the same order, so
   bitwise-equal results (tests/test_fastplan.py).
"""

from __future__ import annotations

import contextlib
import time

import numpy as np

from . import bundle

PAD_ROW = (1.0e4, 1.0e4, -1.0e4, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0)  # far, finite, masked off
# guidance obstacles per graph. Benchmark scenes have <= 7 (median 1); a real room's lidar memory has 20-40 boxes
# (lab-g1 2026-10-23: 29 boxes after 5 s standing). Above the largest bucket the planner used to leave the graphs and
# hand torch.compile a new shape per box count: seconds per recompile, then the recompile limit (an 11 s plan in a
# rehearsal of that room). Padded slots are masked (+inf distance), so the bucket never changes a plan.
# 256-512: finer box decompositions of a real room (150-320 boxes within 2 m); a pool whose candidates x bucket exceed
# MAX_PAIRS runs off the graphs, unsplit (_sample_guided)
BUCKETS = (8, 16, 32, 64, 128, 256, 512)
PRECISIONS = ("fp32", "tf32", "bf16")  # denoiser forward only; fp32 = the evaluated arithmetic
MAX_PAIRS = 12288  # candidates x padded obstacles per graph (48 x 256); one shared pool sized by the largest graph
RECOMPILE_LIMIT = 64  # torch.compile'd pieces see one shape per (candidates, bucket): ~16 here, default limit is 8


# ---------------------------------------------------------------------------------------- numpy geometry
class FastTree:
    """`native_geometry` tree `forward` with its per-joint constants computed once (same operations)."""

    def __init__(self, tree):
        from motion2scene_flow.native_geometry import rotation_wxyz

        self.tree = tree
        self.body_names, self.joint_names, self.root = tree.body_names, tree.joint_names, tree.root
        self._steps = [
            (j["parent"], j["child"], rotation_wxyz(j["q0"]), j["axis"], 7 + tree.joint_names.index(j["name"]),
             rotation_wxyz(j["q1"]).T, np.asarray(j["p0"], dtype=float), np.asarray(j["p1"], dtype=float))
            for j in tree.ordered
        ]
        self._cols = np.array([st[4] for st in self._steps])
        self._axis = np.array(["XYZ".index(st[3]) + 1 for st in self._steps])

    def __getattr__(self, name):
        return getattr(self.tree, name)

    def forward(self, qpos):
        from motion2scene_flow.native_geometry import rotation_wxyz

        q = np.asarray(qpos, dtype=float)
        if q.ndim != 2 or q.shape[1] != 7 + len(self.joint_names) or not np.isfinite(q).all():
            raise ValueError("Invalid named qpos")
        poses = {self.root: (q[:, :3], rotation_wxyz(q[:, 3:7]))}
        # `axis_rotation(axis, angle)` for every joint at once: the same quaternion -> matrix formulas
        angle = q[:, self._cols]
        qa = np.zeros((*angle.shape, 4))
        qa[..., 0] = np.cos(angle / 2)
        qa[:, np.arange(len(self._cols)), self._axis] = np.sin(angle / 2)
        axis_r = rotation_wxyz(qa)
        for j, (parent, child, r0, _, _, r1t, p0, p1) in enumerate(self._steps):
            p, r = poses[parent]
            child_r = r @ r0 @ np.ascontiguousarray(axis_r[:, j]) @ r1t
            child_p = p + np.einsum("tij,j->ti", r, p0) - np.einsum("tij,j->ti", child_r, p1)
            poses[child] = child_p, child_r
        return (np.stack([poses[b][0] for b in self.body_names], axis=1),
                np.stack([poses[b][1] for b in self.body_names], axis=1))


_TREES: dict[int, FastTree] = {}


def fast_tree(tree):
    if isinstance(tree, FastTree):
        return tree
    ft = _TREES.get(id(tree))
    if ft is None or ft.tree is not tree:
        ft = _TREES[id(tree)] = FastTree(tree)
    return ft


def height_field_hits(sel, qpos, datum_offset_m):
    """(len(margins), K) bool: does any capsule point of candidate k hit at margin m (HeightFieldSelector rules).

    Frames are independent, so hits over a split of the frames OR together into the hits over all of them.
    The disc dilations are computed on the crop the points can reach (k_max cells around the in-grid
    capsule points) instead of the whole memory grid: every value read is the same."""
    from motion2scene_flow.native_geometry import world_capsules

    mem = sel.memory
    qpos = np.asarray(qpos)
    n_cand, n_t = qpos.shape[:2]
    hits = np.zeros((len(sel.margins), n_cand), dtype=bool)
    if n_cand == 0 or n_t == 0:
        return hits
    k_max = int(np.ceil((max(s["radius"] for s in sel.shapes) + max(sel.margins)) / mem.res)) + 1
    q = np.array(qpos, dtype=np.float64).reshape(n_cand * n_t, -1)
    q[:, 2] -= datum_offset_m  # memory heights are physical
    p, r = fast_tree(sel.tree).forward(q)
    s, e, rad = world_capsules(p, r, sel.bodies, sel.shapes)
    pts = np.concatenate([s, 0.5 * (s + e), e], 1).reshape(n_cand, n_t, -1, 3)  # (K, T, 3S, 3)
    radius = np.tile(rad, 3)
    ij = np.round((pts[..., :2] - mem.lo) / mem.res).astype(int)
    inside = (ij >= 0).all(-1) & (ij < mem.n).all(-1)
    ij = np.clip(ij, 0, mem.n - 1)
    if not inside.any():
        return hits
    shape = np.shape(mem.support)
    lo_c = np.maximum(ij[inside].min(0) - k_max, 0)
    hi_c = np.minimum(ij[inside].max(0) + k_max + 1, shape)
    crop = (slice(lo_c[0], hi_c[0]), slice(lo_c[1], hi_c[1]))
    sup0 = np.nan_to_num(np.asarray(mem.support)[crop], nan=0.0)
    sup0 = np.where(sup0 > sel.thr, sup0, -np.inf)
    ceil0 = np.nan_to_num(np.asarray(mem.ceiling)[crop], nan=np.inf)
    ci = np.clip(ij - lo_c, 0, hi_c - lo_c - 1)  # out-of-grid points are masked by `inside`
    cache = {}
    for mi, m in enumerate(sel.margins):
        k = np.ceil((radius + m) / mem.res).astype(int).clip(1, k_max)
        for kk in np.unique(k):
            if kk not in cache:
                y, x = np.ogrid[-kk:kk + 1, -kk:kk + 1]
                disc = (x * x + y * y) <= kk * kk
                cache[kk] = (sel._max(sup0, footprint=disc, mode="constant", cval=-np.inf),
                             sel._min(ceil0, footprint=disc, mode="constant", cval=np.inf))
            sup, ceil = cache[kk]
            cols = k == kk
            ii, jj = ci[:, :, cols, 0], ci[:, :, cols, 1]
            z, rr = pts[:, :, cols, 2], radius[cols]
            h = ((sup[ii, jj] > z - rr - m) | (ceil[ii, jj] < z + rr + m)) & inside[:, :, cols]
            hits[mi] |= h.any((1, 2))
    return hits


def clearance_from_hits(margins, hits):
    """Largest margin a candidate passes together with every smaller one, or -1 (the selector's result)."""
    out = np.full(hits.shape[1], -1.0)
    alive = np.ones(hits.shape[1], dtype=bool)
    for mi, m in enumerate(margins):
        alive &= ~hits[mi]
        out[alive] = m
    return out


def height_field_clearance(sel, qpos, datum_offset_m):
    """`sensor_map.HeightFieldSelector.__call__` on all candidates at once (bitwise-equal result)."""
    return clearance_from_hits(sel.margins, height_field_hits(sel, qpos, datum_offset_m))


def ground_legs(qpos, tree, bodies, stance_ankle_z, limits=None, iters=6, eps=1e-3, tol=0.002):
    """`closed_loop.grounding.ground_legs` with the three FK calls of an iteration stacked into one."""
    from motion2scene_flow.closed_loop.grounding import LEGS

    ft = fast_tree(tree)
    q = np.array(qpos, dtype=np.float64)
    n = len(q)
    ankles = [bodies.index("left_ankle_roll_link"), bodies.index("right_ankle_roll_link")]
    p0, _ = ft.forward(q)
    lift = np.maximum(0.0, stance_ankle_z - p0[:, ankles, 2].min(1))
    if not (lift > tol).any():
        return q, lift
    target = p0[:, ankles].copy()
    target[..., 2] += lift[:, None]
    for _ in range(iters):
        stack = [q]
        for slot in (0, 1):  # perturb hip pitch, then knee, for both legs at once
            qq = q.copy()
            for leg in LEGS.values():
                qq[:, 7 + leg[slot]] += eps
                qq[:, 7 + leg[2]] -= eps  # ankle keeps the sole pitch
            stack.append(qq)
        pall, _ = ft.forward(np.concatenate(stack))
        p = pall[:n]
        err = target - p[:, ankles]
        if np.abs(err[..., [0, 2]]).max() < tol:
            break
        jac = np.zeros((n, 2, 2, 2))
        for j in (0, 1):
            pp = pall[(j + 1) * n:(j + 2) * n]
            d = (pp[:, ankles] - p[:, ankles]) / eps
            jac[:, :, :, j] = d[..., [0, 2]]
        rhs = err[..., [0, 2]]
        step = np.linalg.solve(jac + 1e-6 * np.eye(2), rhs[..., None])[..., 0]
        step = np.clip(step, -0.3, 0.3)
        for li, leg in enumerate(LEGS.values()):
            q[:, 7 + leg[0]] += step[:, li, 0]
            q[:, 7 + leg[1]] += step[:, li, 1]
            q[:, 7 + leg[2]] -= step[:, li, 0] + step[:, li, 1]
        if limits is not None:
            q[:, 7:] = np.clip(q[:, 7:], limits[:, 0], limits[:, 1])
    return q, lift


# ---------------------------------------------------------------------------------------- planner
def _planner_module():
    return bundle.load_flow().planner


class _Graph:
    __slots__ = ("graph", "inputs", "out", "n_pad")


_CLASS = None


def fast_planner_class():
    """`FastFlowPlanner`, built on first use: its base class lives in the frozen bundle."""
    global _CLASS
    if _CLASS is None:
        _CLASS = _build_class()
    return _CLASS


def make_planner(pcfg, fast=True, **kw):
    """The deployed planner: `FastFlowPlanner` (default) or the bundle's eager `FlowPlanner`."""
    return fast_planner_class()(pcfg, **kw) if fast else _planner_module().FlowPlanner(pcfg)


def build_planner(pcfg, impl="fused", precision="fp32", warm=True, rules=None):
    """`impl` eager | graph | fused, `precision` fp32 | tf32 | bf16 (fast planner only); compiled + captured.
    `rules`: a deploy rules plugin (bundle.load_rules) - its planner class, built on this one, and its preparation."""
    if impl == "eager":
        if precision != "fp32":
            raise ValueError("the eager planner runs fp32 only")
        base = _planner_module().FlowPlanner
        planner = (rules.planner_class(base) if rules is not None else base)(pcfg)
    else:
        cls = fast_planner_class() if rules is None else rules.planner_class(fast_planner_class())
        planner = cls(pcfg, fuse=impl == "fused", precision=precision)
        if warm and planner.device.type == "cuda":
            planner.warmup()  # compile + capture now (~25 s warm cache), not on the first plan of a walk
    if rules is not None:
        rules.prepare(planner)
    return planner


def __getattr__(name):
    if name == "FastFlowPlanner":
        return fast_planner_class()
    raise AttributeError(name)


def _build_class():
    base = _planner_module()  # puts the bundle's source on sys.path
    import torch
    import torch.nn.functional as F
    from motion2scene_flow.closed_loop import grounding as grounding_mod
    from motion2scene_flow.closed_loop import sensor_map
    from motion2scene_flow.wholebody import passage_state as ps
    from motion2scene_flow.wholebody.passage_losses import batched_capsule_penetration, broadcast_obstacles
    from scipy.spatial.transform import Rotation

    class FastFlowPlanner(base.FlowPlanner):
        """See the module docstring. `graphs=False` runs the same loop eagerly (debugging, CPU)."""

        def __init__(self, cfg, graphs=True, fuse=True, buckets=BUCKETS, max_pairs=MAX_PAIRS, precision="fp32"):
            super().__init__(cfg)
            if precision not in PRECISIONS:
                raise ValueError(f"precision must be one of {PRECISIONS}")
            self.precision = precision if self.device.type == "cuda" else "fp32"
            self.use_graphs = bool(graphs) and self.device.type == "cuda"
            self.fuse = bool(fuse) and self.device.type == "cuda"
            self.buckets, self.max_pairs = tuple(buckets), int(max_pairs)
            self._graphs: dict = {}
            # one memory pool for all graphs: they never run concurrently and every output is copied out
            # right after its replay, so the pool holds the largest graph's activations, not their sum
            self._pool = torch.cuda.graph_pool_handle() if self.use_graphs else None
            self.capture_s = 0.0
            self.overlap_stats = [0, 0]  # overlapped selections reused / recomputed (frames not bitwise equal)
            self._partial_hits = None
            self.model.eval()
            self._temb = None
            if self.fuse:
                import torch._dynamo.config as dynamo_cfg

                for name in ("recompile_limit", "cache_size_limit"):  # name differs across torch versions
                    if hasattr(dynamo_cfg, name):
                        setattr(dynamo_cfg, name, max(getattr(dynamo_cfg, name), RECOMPILE_LIMIT))
                self._vel = torch.compile(self._velocity_tokens, dynamic=False)
                self._loss = torch.compile(self._guide_loss, dynamic=False)

        def _model_precision(self):
            """Context for the denoiser's forward only; guidance geometry (FK, capsules, distances) stays fp32.

            tf32: matmuls on tensor cores with a 10-bit mantissa (cuDNN convs already use TF32 by default, also
            in the evaluated planner). bf16: autocast (matmuls/attention in bfloat16, norms and softmax fp32)."""
            if self.precision == "bf16":
                return torch.autocast("cuda", dtype=torch.bfloat16)
            if self.precision == "tf32":
                return _flag(torch.backends.cuda.matmul, "allow_tf32", True)
            return contextlib.nullcontext()

        # ----- guided sampler ------------------------------------------------------------------
        def _loop(self, *a):
            return (self._fused_loop if self.fuse else self._guided_loop)(*a)

        def _guided_loop(self, x, cond, anchor_raw, xy0, yaw0, packed, valid, guided, margin):
            """`FlowPlanner._sample_guided` after the noise draw (same operations). `margin`: guide_margin_m as a
            0-d float32 tensor, a graph input: a per-replan override of guide_margin_m would otherwise be ignored,
            since a Python float is baked into the captured graph and the compiled kernels."""
            kin, idx = self._guide_setup()
            b, h = x.shape[0], self.horizon
            prior = torch.zeros_like(x)
            committed = torch.zeros(b, h, dtype=torch.bool, device=self.device)
            obs_b = broadcast_obstacles(packed[None].expand(b, -1, -1), valid[None].expand(b, -1)) if guided else None
            steps = self.cfg.ode_steps
            lo, hi = self.cfg.guide_interval or (0.0, 1.0)
            for i in range(steps):
                t = 1.0 - i / steps
                t_in = torch.full((b, h, 1), t, device=self.device)
                with self._model_precision():
                    v = self.model(x, t_in, prior, committed, cond)
                v = v.float()
                if guided and lo <= t <= hi:
                    with torch.enable_grad():
                        xg = x.detach().requires_grad_(True)
                        m = self._decode(xg - t * v.detach(), anchor_raw)
                        pos, rot, _ = self._integrate(m, anchor_raw, xy0, yaw0)
                        bp, br = kin.forward(pos, rot, m[..., 7:36])
                        st, en, rad = kin.capsules(bp, br)
                        pen = batched_capsule_penetration(st[..., idx, :], en[..., idx, :], rad[idx] + margin, obs_b)
                        loss = pen.pow(2).sum()
                        if self.cfg.guide_tilt_max_rad is not None:
                            tilt = torch.acos(rot[..., 2, 2].clamp(-1.0, 1.0))
                            loss = loss + (tilt - self.cfg.guide_tilt_max_rad).clamp_min(0).pow(2).sum()
                        g, = torch.autograd.grad(loss, xg)
                    rms = g.pow(2).mean((1, 2), keepdim=True).sqrt()
                    v = v + self.cfg.guide_w * torch.where(rms > 0, g / rms.clamp_min(1e-12), torch.zeros_like(g))
                x = x - v / steps
            return self._decode(x, anchor_raw)

        # ----- fused sampler: per-chunk constants + compiled step pieces --------------------------
        def _time_embeddings(self):
            """PassageExpert's flow-time embedding for each Euler step (and t = 0 for history tokens).

            The original embeds the same t for every token of every candidate; one row per step is the
            same function of the same input."""
            if self._temb is None:
                from motion2scene_flow.wholebody.model import sinusoidal

                m, steps = self.model, self.cfg.ode_steps
                ts = torch.tensor([1.0 - i / steps for i in range(steps)] + [0.0], device=self.device)
                emb = m.time(sinusoidal(ts, m.dim))
                self._temb = (emb[:steps], emb[steps], ts[:steps].clone())
            return self._temb

        def _consts(self, cond):
            """Everything in PassageExpert.forward that does not depend on x or t: map tokens, the
            conditioning vector, each block's modulation and its cross-attention keys/values."""
            m = self.model
            map_tokens, map_global = m.map(cond["map"])
            c = m.history(cond["history"].flatten(1)) + m.goal(cond["goal"]) + map_global
            if m.anchor_cond:
                c = c + m.anchor(cond["anchor"])
            mods, kvs = [], []
            for blk in m.blocks:
                mods.append(tuple(blk.modulation(c).chunk(6, -1)))
                e, nh = blk.cross.embed_dim, blk.cross.num_heads
                kv = F.linear(map_tokens, blk.cross.in_proj_weight[e:], blk.cross.in_proj_bias[e:])
                kv = kv.unflatten(-1, (2, nh, e // nh))  # (b, S, 2, heads, head_dim), as F.multi_head_attention_forward
                kvs.append((kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)))
            return mods, kvs

        def _velocity_tokens(self, x_t, t_emb, t0_emb, prior, committed, prior_mask, history, mods, kvs):
            """PassageExpert.forward given `_consts` (same layers, same order)."""
            m = self.model
            b, h, _ = x_t.shape
            flags = torch.stack([committed.float(), prior_mask.float()], -1)
            tokens = m.frame_in(torch.cat([x_t, prior * prior_mask[..., None], flags], -1))
            tokens = tokens + m.position + t_emb
            if m.history_tokens:
                tokens = torch.cat([m.history_in(history) + m.history_position + t0_emb, tokens], 1)
            for blk, (s1, sc1, g1, s2, sc2, g2), (k, v) in zip(m.blocks, mods, kvs):
                hh = blk.norm1(tokens) * (1 + sc1[:, None]) + s1[:, None]
                tokens = tokens + g1[:, None] * blk.attn(hh, hh, hh, need_weights=False)[0]
                hh = blk.norm2(tokens)
                e, nh = blk.cross.embed_dim, blk.cross.num_heads
                q = F.linear(hh, blk.cross.in_proj_weight[:e], blk.cross.in_proj_bias[:e])
                n_q = q.shape[1]
                q = q.view(b, n_q, nh, e // nh).transpose(1, 2)
                o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n_q, e)
                tokens = tokens + blk.cross.out_proj(o)
                hh = blk.norm3(tokens) * (1 + sc2[:, None]) + s2[:, None]
                tokens = tokens + g2[:, None] * blk.mlp(hh)
            if m.history_tokens:
                tokens = tokens[:, m.n_history:]
            return m.out(m.norm(tokens))

        def _guide_loss(self, xg, vd, t, anchor_raw, xy0, yaw0, packed, valid, margin):
            """The guidance objective of `FlowPlanner._sample_guided` (squared capsule penetration)."""
            kin, idx = self._guide_setup()
            b = xg.shape[0]
            obs_b = broadcast_obstacles(packed[None].expand(b, -1, -1), valid[None].expand(b, -1))
            m = self._decode(xg - t * vd, anchor_raw)
            pos, rot, _ = self._integrate(m, anchor_raw, xy0, yaw0)
            bp, br = kin.forward(pos, rot, m[..., 7:36])
            st, en, rad = kin.capsules(bp, br)
            pen = batched_capsule_penetration(st[..., idx, :], en[..., idx, :], rad[idx] + margin, obs_b)
            loss = pen.pow(2).sum()
            if self.cfg.guide_tilt_max_rad is not None:
                tilt = torch.acos(rot[..., 2, 2].clamp(-1.0, 1.0))
                loss = loss + (tilt - self.cfg.guide_tilt_max_rad).clamp_min(0).pow(2).sum()
            return loss

        def _fused_loop(self, x, cond, anchor_raw, xy0, yaw0, packed, valid, guided, margin):
            self._guide_setup()
            b, h = x.shape[0], self.horizon
            prior = torch.zeros_like(x)
            committed = torch.zeros(b, h, dtype=torch.bool, device=self.device)
            temb, t0emb, tts = self._time_embeddings()
            with self._model_precision():
                mods, kvs = self._consts(cond)
            steps = self.cfg.ode_steps
            lo, hi = self.cfg.guide_interval or (0.0, 1.0)
            for i in range(steps):
                t = 1.0 - i / steps
                with self._model_precision():
                    v = self._vel(x, temb[i], t0emb, prior, committed, cond["prior_mask"], cond["history"], mods, kvs)
                v = v.float()
                if guided and lo <= t <= hi:
                    with torch.enable_grad():
                        xg = x.detach().requires_grad_(True)
                        g, = torch.autograd.grad(self._loss(xg, v.detach(), tts[i], anchor_raw, xy0, yaw0, packed, valid,
                                                            margin), xg)
                    rms = g.pow(2).mean((1, 2), keepdim=True).sqrt()
                    v = v + self.cfg.guide_w * torch.where(rms > 0, g / rms.clamp_min(1e-12), torch.zeros_like(g))
                x = x - v / steps
            return self._decode(x, anchor_raw)

        def _pack(self, guide_obs, n_pad):
            """Obstacles padded to `n_pad` rows (device ops only, no host round trip)."""
            n, n_pad = guide_obs.shape[0], max(n_pad, 1)
            if getattr(self, "_pad_rows", None) is None or self._pad_rows.shape[0] < n_pad:
                self._pad_rows = torch.tensor(PAD_ROW, dtype=torch.float32, device=self.device).repeat(max(n_pad, 64), 1)
                self._slots = torch.arange(max(n_pad, 64), device=self.device)
            packed = self._pad_rows[:n_pad].clone()
            packed[:n] = guide_obs
            return packed, self._slots[:n_pad] < n

        def _margin(self):
            m = getattr(self, "_margin_t", None)
            if m is None:
                m = self._margin_t = torch.zeros((), dtype=torch.float32, device=self.device)
            m.fill_(float(self.cfg.guide_margin_m))  # in place: captured graphs read this tensor
            return m

        @torch.no_grad()
        def _sample_guided(self, cond, anchor_raw, xy0, yaw0, guide_obs, **kw):
            if any(v is not None for key, v in kw.items() if key != "t0"):  # a newer bundle's options: its own sampler
                return super()._sample_guided(cond, anchor_raw, xy0, yaw0, guide_obs, **kw)
            b = anchor_raw.shape[0]
            cond = {k_: v_ for k_, v_ in cond.items() if k_ != "null"}
            x = torch.randn(b, self.horizon, ps.DIM, generator=self.generator, device=self.device)
            n = int(guide_obs.shape[0])
            guided = n > 0  # the eager sampler skips guidance without obstacles (incl. the tilt term)
            n_pad = next((s for s in self.buckets if s >= n), None) if guided else 0
            # candidates x boxes over max_pairs: off the graphs, the whole pool in one pass. Splitting the pool would
            # change the denoiser's GEMM shapes, and the normalised guidance amplifies those last-bit differences
            # (up to 4e-3 in the raw state on 48 candidates): not the evaluated sampler any more.
            graphs = self.use_graphs and n_pad is not None and b * max(n_pad, 1) <= self.max_pairs
            if n_pad is None:  # beyond the largest bucket: padded to a power of two (few compiled shapes)
                n_pad = 1 << max(n - 1, 0).bit_length()
            packed, valid = self._pack(guide_obs, n_pad)
            args = (x, cond, anchor_raw, xy0, yaw0, packed, valid, guided, self._margin())
            return self._replay(*args) if graphs else self._loop(*args)

        def _replay(self, x, cond, anchor_raw, xy0, yaw0, packed, valid, guided, margin):
            key = (x.shape[0], packed.shape[0], tuple(sorted((k_, tuple(v_.shape)) for k_, v_ in cond.items())))
            gr = self._graphs.get(key)
            if gr is None:
                gr = self._graphs[key] = self._capture(x, cond, anchor_raw, xy0, yaw0, packed, valid, guided, margin)
            inp = gr.inputs
            inp["x"].copy_(x)
            for k_, v_ in cond.items():
                inp["cond"][k_].copy_(v_)
            inp["anchor_raw"].copy_(anchor_raw)
            inp["xy0"].copy_(xy0)
            inp["yaw0"].copy_(yaw0)
            inp["packed"].copy_(packed)
            inp["valid"].copy_(valid)
            gr.graph.replay()
            return gr.out.clone()

        def _capture(self, x, cond, anchor_raw, xy0, yaw0, packed, valid, guided, margin):
            t0 = time.perf_counter()
            inp = {"x": x.clone(), "cond": {k_: v_.clone() for k_, v_ in cond.items()},
                   "anchor_raw": anchor_raw.clone(), "xy0": xy0.clone(), "yaw0": yaw0.clone(),
                   "packed": packed.clone(), "valid": valid.clone()}

            def run():  # `margin` is the planner's own tensor (refilled in place before every replay)
                return self._loop(inp["x"], inp["cond"], inp["anchor_raw"], inp["xy0"], inp["yaw0"],
                                  inp["packed"], inp["valid"], guided, margin)

            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):  # warm-up: lazy caches, cuBLAS handles, autograd state
                for _ in range(2):
                    run()
            torch.cuda.current_stream(self.device).wait_stream(side)
            gr = _Graph()
            gr.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr.graph, pool=self._pool):
                gr.out = run()
            gr.inputs, gr.n_pad = inp, packed.shape[0]
            torch.cuda.synchronize(self.device)
            self.capture_s += time.perf_counter() - t0
            return gr

        @torch.no_grad()
        def warmup(self, ks=None, buckets=None):
            """Capture the graphs a run will need before the first real plan (seconds each)."""
            ks = ks or (self.cfg.candidates, 3 * self.cfg.candidates)
            buckets = buckets or self.buckets
            hist = torch.zeros(self.history, ps.DIM)
            hist[:, 0] = 0.79 + self.cfg.datum_offset_m
            hist[:, 3] = -1.0  # gravity in the root frame
            for k in ks:
                hist_k = hist.to(self.device)[None].repeat(k, 1, 1)
                xy = torch.zeros(k, 2, device=self.device)
                yaw = torch.zeros(k, device=self.device)
                goal = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device=self.device).repeat(k, 1)
                world = torch.zeros(0, 9, device=self.device)
                cond, anchor_raw = self._condition(hist_k, xy, yaw, goal, world, None)
                for n in (0,) + tuple(buckets):  # 0: no obstacles seen yet (unguided graph)
                    if k * n > self.max_pairs:
                        continue
                    obs = torch.tensor([PAD_ROW], device=self.device).repeat(n, 1)
                    gen = self.generator.get_state()
                    self._sample_guided(cond, anchor_raw, xy, yaw, obs)
                    self.generator.set_state(gen)  # warm-up must not consume the planner's random stream

        # ----- candidates (fewer host syncs) ----------------------------------------------------
        def _condition(self, hist_raw, xy0, yaw0, goal, world_obs, map_fn=None, host=None):
            """`FlowPlanner._condition`; `host` = (xy, yaw) already on the host (saves a sync)."""
            if map_fn is None or host is None:
                return super()._condition(hist_raw, xy0, yaw0, goal, world_obs, map_fn)
            b = hist_raw.shape[0]
            window = torch.cat([hist_raw, torch.zeros(b, self.horizon, hist_raw.shape[-1], device=hist_raw.device)], 1)
            if self.residual:
                window = window.clone()
                for sl in base.RESIDUAL_SLICES:
                    window[:, :, sl] = window[:, :, sl] - window[:, self.history - 1: self.history, sl]
            states = self.normalizer.encode(window)
            anchor_raw = hist_raw[:, -1]
            if self.cfg.scene_input not in ("true", "empty"):
                raise ValueError("scene_input must be 'true' or 'empty'")
            xs, ys = host
            if (xs == xs[0]).all() and (ys == ys[0]).all():  # first chunk: every candidate starts at the anchor
                maps = map_fn(xs[0], float(ys[0]))[None].repeat(b, 1, 1, 1).to(hist_raw.device)  # map_fn is pure
            else:
                maps = torch.stack([map_fn(xs[i], float(ys[i])) for i in range(b)]).to(hist_raw.device)
            if self.cfg.support_clip_m is not None:
                maps[:, 0] = maps[:, 0].clamp(max=self.cfg.support_clip_m)
                maps[:, 1] = maps[:, 1].clamp(max=self.cfg.support_clip_m)
            cond = {
                "history": states[:, : self.history],
                "anchor": self.abs_normalizer.encode(anchor_raw),
                "goal": goal,
                "map": maps,
                "prior_mask": torch.zeros(b, self.horizon, dtype=torch.bool, device=hist_raw.device),
            }
            if self.cfg.guidance_w != 1.0:
                return super()._condition(hist_raw, xy0, yaw0, goal, world_obs, map_fn)
            return cond, anchor_raw

        def _overlap_selector(self):
            fn = getattr(self, "_clearance_fn", None)
            plain = isinstance(fn, sensor_map.HeightFieldSelector) and type(fn).__call__ is sensor_map.HeightFieldSelector.__call__
            return fn if plain and self.cfg.select_frames is None else None

        def _to_qpos(self, flat, k, n):
            """Host rows [rot (n*9) | positions (n*3) | joints (n*29)] -> qpos (k, n, 36), as `_candidates`."""
            rot_h = flat[:, : n * 9].reshape(-1, 3, 3)
            pos_h = flat[:, n * 9: n * 12].reshape(k, n, 3)
            joints_h = flat[:, n * 12:].reshape(k, n, 29)
            quat = Rotation.from_matrix(rot_h).as_quat()[:, [3, 0, 1, 2]]
            return np.concatenate([pos_h, quat.reshape(k, -1, 4), joints_h], -1), pos_h

        @torch.no_grad()
        def _candidates(self, hist0, xy0, yaw0, goal_fn, k, world, map_fn=None, guide_obs=None, **kw):
            """`FlowPlanner._candidates` with one host sync per chunk; while chunk c samples on the GPU, the
            CPU runs the height-field selection on chunk c-1's frames (exact: frames are independent).

            Keyword options of a newer bundle's `_candidates` that this loop does not implement run the bundle's own
            method when set (not None); a deploy rules plugin's planner class covers that bundle's config options."""
            if any(v is not None for key, v in kw.items() if key != "prior_t"):
                return super()._candidates(hist0, xy0, yaw0, goal_fn, k, world, map_fn, guide_obs, **kw)
            sel = self._overlap_selector()
            hist = hist0.to(self.device)[None].repeat(k, 1, 1)
            xy = xy0.to(self.device)[None].repeat(k, 1)
            yaw = yaw0.to(self.device)[None].repeat(k)
            xs = np.repeat(np.asarray(xy0, dtype=np.float32)[None], k, 0)  # chunk 0 anchors are on the host
            ys = np.repeat(np.asarray(yaw0, dtype=np.float32)[None], k, 0)
            states, positions, yaws, goals = [], [], [], []
            done_q, hits = [], None
            for c in range(self.cfg.chunks):
                prev_q = None
                if c > 0:
                    parts = [xy, yaw[:, None]]
                    if sel is not None:  # the previous chunk's frames ride along with the anchors (same sync)
                        n_prev = states[-1].shape[1]
                        rot_prev = ps.compose(yaws[-1], states[-1][..., 1:4])
                        parts += [rot_prev.reshape(k, -1), positions[-1].reshape(k, -1), states[-1][..., 7:36].reshape(k, -1)]
                    host = torch.cat(parts, 1).cpu().numpy()  # one sync per chunk
                    xs, ys = host[:, :2], host[:, 2]
                    if sel is not None:
                        prev_q = self._to_qpos(host[:, 3:], k, n_prev)[0]
                goal_h = torch.stack([goal_fn(xs[i].astype(float), float(ys[i])) for i in range(k)])
                goals.append(goal_h)
                goal = goal_h.to(self.device)
                cond, anchor_raw = self._condition(hist, xy, yaw, goal, world, map_fn, host=(xs, ys))
                if guide_obs is not None and self.cfg.guide_w > 0:
                    chunk = self._sample_guided(cond, anchor_raw, xy, yaw, guide_obs)
                else:
                    chunk = self._sample(cond, anchor_raw)
                if prev_q is not None:  # CPU work while the GPU samples this chunk
                    h = height_field_hits(sel, prev_q, self.cfg.datum_offset_m)
                    hits = h if hits is None else hits | h
                    done_q.append(prev_q)
                pos, _, cyaw = self._integrate(chunk, anchor_raw, xy, yaw)
                states.append(chunk)
                positions.append(pos)
                yaws.append(cyaw)
                hist = chunk[:, -self.history:]
                xy, yaw = pos[:, -1, :2], cyaw[:, -1]
            states = torch.cat(states, 1)
            positions = torch.cat(positions, 1)
            yaws = torch.cat(yaws, 1)
            rot = ps.compose(yaws, states[..., 1:4])
            n = states.shape[1]
            flat = torch.cat([rot.reshape(k, -1), positions.reshape(k, -1), states[..., 7:36].reshape(k, -1)], 1)
            qpos, pos_h = self._to_qpos(flat.cpu().numpy(), k, n)  # one transfer
            self._partial_hits = None
            if done_q:
                n0 = sum(q_.shape[1] for q_ in done_q)
                same = np.array_equal(qpos[:, :n0], np.concatenate(done_q, 1))
                self.overlap_stats[0 if same else 1] += 1
                if same:  # bitwise the frames the final selection will see -> reuse their hits
                    self._partial_hits = (qpos, n0, hits)
            return {"states": states, "rot": rot, "qpos": qpos, "goal0": goals[0][0].numpy(),
                    "end_xy": pos_h[:, min(self.horizon, n) - 1, :2]}

        # ----- selection and grounding (vectorised, bitwise equal) ------------------------------
        def _clearance_raw(self, qpos, obstacles):
            fn = getattr(self, "_clearance_fn", None)
            if isinstance(fn, sensor_map.HeightFieldSelector) and type(fn).__call__ is sensor_map.HeightFieldSelector.__call__:
                pend, self._partial_hits = getattr(self, "_partial_hits", None), None
                if pend is not None and pend[0] is qpos:  # earlier chunks were checked during sampling
                    _, n0, h0 = pend
                    return clearance_from_hits(fn.margins, h0 | height_field_hits(fn, qpos[:, n0:], self.cfg.datum_offset_m))
                return height_field_clearance(fn, qpos, self.cfg.datum_offset_m)
            return super()._clearance_raw(qpos, obstacles)

        def plan(self, *a, **kw):
            with _patched(grounding_mod, "ground_legs", ground_legs):
                return super().plan(*a, **kw)

    return FastFlowPlanner


@contextlib.contextmanager
def _flag(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def _patched(mod, name, fn):
    old = getattr(mod, name)
    setattr(mod, name, fn)
    try:
        yield
    finally:
        setattr(mod, name, old)


def bench(planner, args, kw, n=20, sync=None):
    """Per-plan wall times (s) of `planner.plan(*args, **kw)` after one warm-up call."""
    import torch

    sync = sync or (torch.cuda.synchronize if planner.device.type == "cuda" else (lambda: None))
    planner.plan(*args, **kw)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        planner.plan(*args, **kw)
        sync()
        ts.append(time.perf_counter() - t)
    return np.asarray(ts)


__all__ = ["FastTree", "fast_tree", "height_field_hits", "clearance_from_hits", "height_field_clearance", "ground_legs", "fast_planner_class", "make_planner", "bench"]
