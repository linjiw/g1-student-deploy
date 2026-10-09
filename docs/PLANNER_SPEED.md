# Planner inference speed (2026-10-08)

**Result.** The deployed student planner (K = 16 candidates, 10 guided Euler steps, 2 chunks, sensor selection,
leg grounding) now runs **2.3–3.4× faster on the deploy PC** under the same GPU contention from other jobs. The model,
the config and the sampler's maths are unchanged. Implementation: `g1deploy/fastplan.py` (`FastFlowPlanner`,
`--planner-impl fused`, now the default everywhere). The frozen bundle source is byte-identical.

| per plan, deploy PC (RTX 5080 shared at ~100 %), interleaved | p50 | p95 |
|---|---|---|
| eager (as evaluated), one recorded state | 0.341 s | 0.372 s |
| graph (CUDA graphs + host-side fixes) | 0.194 s | 0.237 s |
| **fused (deployed)**, one recorded state | **0.126 s** | **0.158 s** |
| eager, 37 states replayed along 6 episodes (`--sequence`) | 0.348–0.376 s | 0.390–0.408 s |
| **fused**, the same 37 states | **0.128–0.134 s** | **0.167–0.176 s** |
| worst case, every pool unsafe (16 + 48 + 16 candidates), eager | 0.96–1.20 s | 1.05–1.25 s |
| worst case, fused | 0.34–0.54 s | 0.38–0.59 s |
| lighter background load, 37 states: eager / fused | 0.317 / **0.074 s** | 0.358 / **0.104 s** |
| end to end in the real-time rehearsal (node + C++ runner + DDS MuJoCo, beams2-100, 28 plans), fused | **0.083 s** | **0.119 s** |

The ranges are two runs at different background load. The worst case occurs in 4 % of plans in the realistic
panel (0.5 % in slow mode). Note that the eager arms modelled those plans at the panel's flat 0.32 s, not ~1 s.

**Not a different planner.**
- Same random state → same chosen candidate and rule in 30/30, 37/37 and 12/12 compared plans.
- Clearance values are bitwise equal.
- Joints differ by ≤ 4e-5 rad. The eager planner itself differs by ~2e-5 between two identical runs (atomic
  accumulation in index_select's backward).
- Closed loop (98 held-out episodes, same conditions as the eager arms):

| condition | eager | fused | per-episode agreement |
|---|---|---|---|
| ideal lidar, 0.32 s predictive (`g4-lat032-pred` / `g4w-fused-base-lat032`) | 79 | 78 | 97/98 same outcome |
| realistic: MID-360 + stand + lidar route + LIO noise, 0.32 s (`g4t-deploy-real` / `g4w-fused-real-lat032`) | 75 | 76 | 97/98 same outcome |
| realistic at the fused planner's measured p95, 0.18 s (`g4w-fused-real-lat018`) | 75 at 0.32 s, 84 at 0 s | **82** | the latency gap is closed (Isaac 83) |

## 1. Where the time went (eager, one plan, 350 ms under contention)

`tools/profile_planner.py` (stage timers + torch.profiler):

| stage | ms | why |
|---|---|---|
| guided sampler, 2 chunks × 10 steps | 226 | 23 272 CUDA kernel launches: model forward plus a backward through FK, capsules and box distances at every step. Each kernel is tiny (16 × 25 tokens), so launch overhead and per-kernel queueing behind the other GPU processes dominate; the kernels themselves sum to ~60 ms. |
| height-field selection | 50 | numpy FK per candidate (16 Python loops over 29 joints) + 10 disc dilations of the whole 241 × 241 memory grid |
| leg grounding | 28 | ≤ 7 Gauss-Newton iterations × 3 FK calls |
| host syncs | ~55 | 44 `.cpu()` per plan (one per candidate for the goal) |
| maps, goals, misc | ~20 | |

## 2. What changed (in order of effect)

1. **CUDA graphs.** The guided sampler of one chunk is captured once per (candidates, obstacle bucket) and
   replayed as a single launch.
   - Noise is drawn from the planner's own generator outside the graph, so the random stream is the original's.
   - Guidance obstacles are padded to 8 / 16 / 32 rows with `valid = False`; the distance turns those into
     +inf, so every minimum, penetration and gradient is unchanged.
   - Graphs for K = 16 and 48 × {0, 8, 16, 32} obstacles share one memory pool: < 1 GB in total. Benchmark
     scenes have ≤ 7 guidance boxes (median 1); more than 32 falls back to the same loop run eagerly.
2. **Per-chunk model constants.** In `PassageExpert.forward`, the map CNN, the history/goal/anchor MLPs, each
   block's modulation and the cross-attention keys/values do not depend on x or t. They are now computed once
   per chunk instead of at each of the 10 steps; the flow-time embedding is computed once per step value.
3. **Compiled step pieces.** The per-step token path and the guidance objective (decode → integrate → FK →
   capsules → box distance → squared penetration) run as `torch.compile` (inductor) kernels inside the graphs.
   - Same layers and formulas; float32 results within 1e-5.
   - The compile cache lives in `.cache/inductor`. Cold start ~110 s, warm start ~24 s, done before the node
     reports ready. Run `./g1 bench` once after installing or updating.
4. **Host side, bitwise equal.**
   - One host sync per chunk instead of one per candidate.
   - One transfer of the finished candidates instead of five.
   - Selection and grounding run the numpy FK once on all candidates × frames, with per-joint constants and
     the 29 axis rotations batched: the same float64 operations in the same order.
   - Dilations are cropped to the reachable cells.
   - While chunk 2 samples on the GPU, the CPU already runs the selection on chunk 1's frames. Frames are
     independent; the hits are reused only after checking those frames are bitwise identical in the final
     plan, which was true in 14/14 pools.
   - The first chunk's ego map is computed once, since all candidates start at the anchor.

Tests: `tests/test_fastplan.py`
- FK on 5 000 random configurations, selection (incl. mixed pass/fail and floor-sinking cases) and grounding:
  bitwise equal;
- plans from `graph` and `fused` against eager: same choice, rule and clearances;
- cached velocity against `model.forward` < 1e-4 relative.

## 2b. Reduced precision for the denoiser (tested 2026-10-09)

`--planner-precision tf32|bf16` (node, console, lockstep, eval_panel; bench `--impl fused-tf32|fused-bf16`).
- It applies only to the DiT forward: TF32 matmuls, or bf16 autocast with norms and softmax kept in fp32.
- The guidance geometry stays fp32: FK, capsules, box distances and their gradient.
- cuDNN convs already ran in TF32 in the evaluated planner, which is PyTorch's default.

| | fp32 (default) | tf32 | bf16 |
|---|---|---|---|
| denoiser velocity error vs eager (relative, per step) | 3–5e-7 | 4–5e-4 | 4–7e-3 |
| p50 / p95 per plan (37 recorded states, interleaved, loaded GPU; eager 0.377 / 0.425 s) | 0.119 / 0.167 s | 0.093 / 0.142 s | 0.083 / 0.125 s |
| same candidate + rule as eager (same random state) | 37/37 | 34/37 | 34/37 |
| plan difference vs eager, median / max | 1e-6 / 1.5e-5 | 1.5e-3 / 1.0 | 1.5e-2 / 1.6 |
| realistic panel at 0.18 s (98 eps, Isaac 83) | 82 (`g4w-fused-real-lat018`) | 82 (`g4x-tf32-real-lat018`) | 85 (`g4x-bf16-real-lat018`) |
| same per-episode outcome as fp32 | — | 88/98 (5 : 5) | 87/98 (4 : 7, p = 0.55) |

**Reading.**
- Both are genuinely different planners: they pick another candidate in ~1 of 12 plans and agree with fp32 on
  87–88/98 episodes, where fused fp32 agrees with eager on 97/98.
- Neither loses on the panel. bf16 is +3 (not significant); per family it gains on slalom and corridor and
  loses 2 on beam.
- They save 25–40 ms per plan (15–25 %). The latency curve is flat below 0.32 s, so that buys margin, not
  success.

**Decision.**
- Keep fp32, the evaluated arithmetic, as the default.
- bf16 is the validated fallback if the robot-day GPU is so loaded that `./g1 bench` p95 approaches 0.32 s.
- One panel resolves only large differences (about ±5 episodes). Re-run a panel before making bf16 the default.

## 3. How to measure on any machine

```bash
./g1 bench                                    # fused, one state: p95 must be <= 0.32 s
.venv/bin/python tools/bench_planner.py --impl eager --impl fused --sequence     # A/B along recorded episodes
.venv/bin/python tools/bench_planner.py --impl fused --force-fallback            # worst case (all pools unsafe)
.venv/bin/python tools/profile_planner.py --fast                                 # stage breakdown
```

The node itself adapts its predictive-anchoring lead to the measured latency (0.9 quantile of the last 8 plans).
So a faster planner gets earlier anchors without configuration.

## 4. Levers not taken (and why)

| lever | expected | why not (yet) |
|---|---|---|
| free the GPU (pause training on the deploy PC) | largest real-world effect; kernel time is ~25 ms of a fused plan | operational, not code: do it before robot runs |
| TF32 / bf16 tensor cores for the DiT | tested: see §2b | available as `--planner-precision`; default stays fp32 |
| 6 ODE steps | −40 % sampler time; 81/98 at 0.28 s (`g4v-k16s6-lat028`) | a different planner; not needed at 0.17 s |
| K = 8 | −40 % GPU work | loses: 70/98 at 0.32 s (`g4v-k8s10-lat032`) |
| resample + stop pools in one batch | worst case −30 % | changes the order of random draws (statistically equal, not the same plans) |
| selection FK on the GPU | −5 ms | float32/float64 GPU FK is not bitwise equal to the selector's numpy FK |
| TensorRT export of the sampler | similar to the graphs | the guidance needs autograd through FK: a large export effort |
