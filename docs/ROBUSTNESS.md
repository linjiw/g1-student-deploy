# Deployment robustness: deploy-side guards

The lab G1 findings of 2026-10-23 traced the worst failures to the point where the real robot's map reaches the
planner, not to walking itself. Two leaked self-return cells beside the shoulder made all 16 candidates unsafe.
A closed room or an L of furniture became one solid guidance box with the robot inside. With nothing safe, the
frozen rule streamed a 0.8–1.2 m max-clearance motion from a standstill, which SONIC tracked as a lunge and a
crouch. `g1deploy/guards.py` adds five guards between the memory and SONIC, plus one mode for the early rungs.
The frozen bundle, the model and its random stream are unchanged. The self mask, the walls and the split boxes do
change the map and guidance the planner is given, so its candidates differ. A safe frozen choice inside the
envelope is streamed byte for byte. The guards are opt-in: `--guards deploy` on the node, `sim/lockstep.py` and
`sim/realtime/rehearse.py`.

| guard | finding | what it does | what it never does |
|---|---|---|---|
| self mask | leaked shoulder returns stored as a 1.06 m post; 0 of 16 candidates safe | Hides occupied cells from the planner's view (map input, guidance boxes, selection) when three things hold:
- the cell's centre is within 10 cm (plus half a cell diagonal, about 13.5 cm in all) of the planner's capsules at the measured pose;
- every return in it was seen from closer than 0.6 m to the sensor (`CloudMemory` now counts far returns per cell);
- its connected evidence lies entirely inside that band.

At most 8 cells are hidden. Above that nothing is hidden, and the node reports `self_mask` with `refused`. In practice it acts from a standstill: a leak trail laid down while walking reaches outside the band and is kept. | Remove anything from the memory: the stand deferral, `--dump-memory` and the site check still see it. Hide a cell that was ever seen from farther away, such as a wall the robot walked up to. A real object within 10 cm of the body that was only ever seen up close is hidden, so the site check keeps that a NO-GO. |
| split boxes | room walls as one 5.4 × 6.1 m box; L-shaped furniture as a 3.0 × 6.9 m box around the robot | Builds guidance boxes from each component's own cells: row runs on 10 cm cells, merged while at least 80% occupied, nearest 16. A component whose boxes would contain the robot is covered exactly instead. | Contain the robot's own position. Up to 20% of a box can be free floor. |
| walk envelope | a one-legged lunge (knee 2.1 rad) and open-floor crouches passed the node's checks | With no ceiling below 1.45 m within 2 m, the streamed plan needs pelvis ≥ 0.60 m, knees ≤ 2.05 rad and planned tilt ≤ 0.60 rad. Otherwise the planner's own preference (end nearest its local goal) picks among safe candidates that fit, materializing at most 6. If none fits while walking, the plan is dropped and the current one keeps playing. This is allowed at most 2 times in a row, and only while the stream has a planner latency to spare; after that, and from a standstill, the calm fallback takes over. | Apply under a ceiling: ducking under a beam is the point there. |
| calm fallback | no safe candidate → highest-clearance candidate from any pool → lunge, crouch, 1 fall in 3 | **From a standstill** (pelvis below 0.25 m/s and 0.3 rad/s), with nothing safe and inside the envelope, take the stop-pool candidate with the highest clearance, then the least translation. It must stay within 0.25 m, at least as clear as standing still, and never predicted in contact (clearance ≥ 0). If there is none, settle to stand from the anchor: joints to SONIC's default over 1 s, level, yaw kept. **While walking**, the frozen fallback stays when it is inside the envelope, and the next plan is anchored on the measured state. | From a standstill, stream a go-pool max-clearance motion, or a move predicted to collide when standing still is clear. |
| anchor check | predicted anchors 0.5–0.7 m apart within one second | Anchors on the measured state, not a prediction, after a fallback or calm plan. It also does so for two re-plans (plus their landing latency) after a predicted anchor missed the measured pose by more than 0.4 m. A measured plan from a standstill starts when it lands, as the first plan does. | Hold a later predicted plan back. Measured plans are ordered by request, so an older plan never replaces a newer one. Clean walks crossed 0.4 m now and then (at most 0.5 m in the panel). |
| `--guards walls` (H5–H6 mode) | table tops at 0.8–1.35 m filed as ceilings; open-floor crouches | Turns every remembered underside below 1.45 m into a 1.5 m wall, so the planner routes around tables instead of ducking. | Belong in H7, which is about overheads. |

The thresholds are measured, not tuned for success. The envelope comes from 192 plans of clean authored walks:
- pelvis no lower than 0.65 m;
- knee at most 1.94 rad;
- planned tilt at p99 0.44 rad, at most 0.56 rad.

The plans it must reject went much further: crouches to 0.44–0.58 m, lunges with a 2.1–2.9 rad knee, and tilts of
0.83–0.96 rad just before a fall. Clean walks miss their predicted anchor by 0.09–0.26 m at the median and 0.5 m at
most. The walks that fell in boxed-in rooms missed by 0.5–1.6 m.

Smaller changes in the same branch:
- **Node goal command.** `{"cmd": "goal", "frame": "odom", "x", "y", "seq"}` sets an absolute goal, and the status
  echoes `goal_seq` (Studio's U1).
- **Node status.** It now carries `yaw_rad`, `guards`, `guard_counts`, `last_guard`, `self_mask_plans` and
  `self_mask_max_cells`. Guard actions arrive as `{"event": "guard"}`, the self mask as `self_mask` (on first sight,
  growth or refusal), and measured anchoring as `anchor_miss` / `anchor_measured`. The plan log keeps per plan the
  masked cells, guidance boxes and the guards' time (`guard_ms`).
- **Guards off.** Behaviour is unchanged, but the logs gain fields: `no_safe`, `candidate_pool` and the knee/tilt of
  each plan.
- **Self-filter override.** `--self-margin` on the node and on rehearse.py.
- **Two-band head occlusion.** Mid360Scanner accepts `occlusion.bands`; the lab G1 is `[[-45, -35], [40, 50]]`.
- **`CloudMemory(min_high_points=…)`.** Asks for several high returns before a cell counts as overhead. The default
  of 1 keeps the frozen classification.

## Evidence: robustness panel (lock-step)

`tools/robust_panel.py` authors nine courses that start standing:
- open floor, a pillar and a slalom;
- a closed room, L-shaped furniture, a table beside the goal, and a dense room with 26 pieces of furniture;
- starts 0.42 m and 0.33 m from a wall.

They are not drawn from any evaluation packet. Every walk uses:
- the MID-360 scan pattern with per-point timing, 2 cm range noise and the self filter;
- a 15 s stand before GO;
- the sensor route;
- 0.3 s planner latency with predictive anchoring;
- LIO-like odometry noise.

Three seeds per course and condition, 27 walks per condition. Runs on the RTX 5080 dev box, 2026-10-23, with the
code of this branch after two independent reviews.

| condition | done | fall | collision | timeout | first plan with nothing safe | lowest pelvis (m) | highest realized knee (rad) | plan latency p95 / max (s) |
|---|---|---|---|---|---|---|---|---|
| tag: margin 3 cm, no guards (deploy-2026-10-23) | 12 | 9 | 3 | 3 | 19 | 0.42 | 2.62 | 0.36 / 0.55 |
| kit: margin 5 cm, no guards (deploy-kit) | 14 | 9 | 4 | 0 | 0 | 0.49 | 2.26 | 0.13 / 0.27 |
| kit + `--guards deploy` | 23 | 0 | 4 | 0 | 0 | 0.64 | 1.84 | 0.14 / 0.28 |
| leaky (3 cm) + `--guards deploy` | 22 | 0 | 5 | 0 | 0 | 0.62 | 1.86 | 0.15 / 0.50 |
| kit + `--guards walls` | 23 | 0 | 4 | 0 | 0 | 0.65 | 1.84 | 0.15 / 0.44 |
| kit, slow (`--goal-lookahead 1.0`) | 16 | 10 | 1 | 0 | 1 | 0.37 | 2.05 | 0.11 / 0.22 |
| **kit + `--guards deploy`, slow** | **26** | 0 | 1 | 0 | 1 | 0.52 | 1.94 | 0.13 / 0.49 |
| **kit + `--guards walls`, slow** | **25** | 0 | 2 | 0 | 1 | 0.64 | 1.90 | 0.12 / 0.35 |

Completed walks per course, out of 3:

| course | tag | kit | kit+guards | leaky+guards | kit+walls | kit slow | kit+guards slow | kit+walls slow |
|---|---|---|---|---|---|---|---|---|
| open-2m | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| pillar | 1 | 2 | 2 | 2 | 2 | 3 | 3 | 3 |
| slalom | 2 | 0 | 0 | 0 | 0 | 2 | 2 | 1 |
| room | 0 | 0 | 3 | 3 | 3 | 0 | 3 | 3 |
| l-shape | 0 | 0 | 3 | 3 | 3 | 0 | 3 | 3 |
| table | 1 | 3 | 3 | 3 | 3 | 2 | 3 | 3 |
| dense | 0 | 0 | 3 | 3 | 3 | 0 | 3 | 3 |
| wall-start | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| wall-close | 2 | 3 | 3 | 2 | 3 | 3 | 3 | 3 |

What the numbers say:
- **Boxed-in rooms decide it.** Without the split boxes, every closed room, L of furniture and dense room fell
  within about 1 s: plans tilted 0.9 rad and predicted anchors missed by 0.5–1.6 m. With the guards, all 45 of those
  walks completed, across five guarded conditions. The 5 cm margin alone fixes the first plan; it does not fix these.
- **No guarded walk fell.** The guards got 22–26 of 27 walks done, against 12–16 without them.
- **The open-floor failure from the lab twin reproduces.** At 3 cm, seed 2 lunged 1.2 m on its first plan, crouched
  to 0.53 m, then stalled for 22 re-plans 0.4 m short of the goal, taking 9.1 s. With the guards, the same leak was
  2 masked cells and two envelope re-selections; the walk kept a lowest pelvis of 0.65 m and arrived in 3.4 s.
- **Column brushes remain.** Almost every remaining failure is an arm brushing a 0.24–0.3 m column at a pelvis gap
  of 0.15–0.3 m, with tens to about 200 N of contact (the scorer's threshold is 5 N). The predicted anchors before
  them missed by 0.1–0.4 m, mostly below the 0.4 m check. They happen with and without guards, and slow mode
  removes most of them (slalom 0/3 → 1–2/3, pillar 2/3 → 3/3). The plans clear the column by the 8 cm selection
  margin and the executed motion does not, which only the planner side can fix (see below).
- **One cost of the guards.** With the leaky 3 cm filter and a start 0.33 m from a wall, the tag stack stood still
  until the timeout. With the guards, one masked leak cell made a stop candidate safe; the robot moved and its arm
  brushed the wall (51 N) within 1 s. With the 5 cm filter that course is 3/3 either way. Starts this close to
  anything are a site-check NO-GO.
- **Guard actions were rare and cheap.** Over 1,545 guarded plans there were 74 plans with masked cells (none
  refused), 29 envelope re-selections, 7 walking drops, 3 calm stops and 1 calm stand. Each plan logs the guards'
  own work in `guard.ms`: view preparation (self mask, box split) about 9 ms at the median and 15 ms at p95 on this
  CPU, selection up to about 40 ms when it re-materializes candidates.
- **Latency spikes are the frozen fallback.** Planner p95 stays at 0.12–0.15 s on this GPU with the guards, against
  0.13 s without. Capping guidance at the 16-box graph bucket matters: with 32 boxes, dense-room p95 was 0.22 s.
  The maxima of 0.35–0.50 s are plans where nothing was safe: the frozen planner then resamples 3K candidates and a
  stop pool, 3–5 model evaluations. The 4070 Ti SUPER deploy GPU and its CPU are untested; there those plans will
  exceed the 0.32 s gate.

## Evidence: real-time chain

`tools/robust_panel.py --realtime` runs the same courses through `sim/realtime/rehearse.py`, one at a time: the
SONIC C++ runner (TensorRT), DDS on loopback, the raw MID-360 DDS path and the node. Lock-step walked the lab twin
cleanly while real time failed it, so this is the check that counts. The runs use five courses, two seeds and a 15 s
stand before GO, with the node's own latency (no fixed 0.3 s).

| condition | done | fall | collision | did not finish |
|---|---|---|---|---|
| tag: margin 3 cm, no guards | 3/10 | 4 | 1 | 2: the first plan rejected (pelvis 0.24 m, 2.6 m/s), so the node never streamed (the deadlock deploy-kit fixes) |
| kit: margin 5 cm, no guards | 3/10 | 4 | 1 | 2: one never streamed; one stood after 99 rejected plans |
| **kit + `--guards deploy`, slow** | **8/10** | 0 | 2 | 0 |
| **kit + `--guards walls`, slow** | **7/10** | 0 | 2 | 1: the lidar stream stopped for 1 s and the supervisor settled to stand |

| course | tag | kit | kit+guards slow | kit+walls slow |
|---|---|---|---|---|
| open-2m | 2 | 2 | 2 | 2 |
| room | 0 | 0 | 2 | 1 |
| l-shape | 0 | 0 | 2 | 2 |
| slalom | 1 | 1 | 0 | 0 |
| dense | 0 | 0 | 2 | 2 |

What these runs show:
- **Rooms, L-shapes and dense rooms.** All fell or never started without the guards, in real time as in lock-step,
  under both filter margins. With the guards they walked: 11 of 12 (one stopped on the lidar stall).
- **Falls and E-stops.** The unguarded runs fell 8 times and set off 9 tilt E-stops; the guarded runs had none.
- **Slalom.** It collided in all 4 guarded real-time runs and in 2 of 4 unguarded ones. Tight columns stay outside
  what this student can be trusted with (see below).
- **Guard actions.** In real time they came to 1 calm stand (a frozen `fallback_go_max_clearance` from a
  standstill), 5 envelope re-selections (knees 2.1–2.4 rad, a 0.60 m pelvis) and 1 walking drop (tilt 0.63 rad).
  None of the ten worst per-run plan latencies exceeded 0.25 s at p95 on this GPU.

## Recommended use on the ladder

| rung | node flags | course limits |
|---|---|---|
| H5 open floor | `--guards walls --goal-lookahead 1.0` | site check GO; nothing within 0.5 m of the start |
| H6 ground obstacles | `--guards walls --goal-lookahead 1.0` | a single pillar or a wall detour; no column slalom (arms brush columns in sim, guards or not) |
| H7 overheads | `--guards deploy --goal-lookahead 1.0` (the envelope is waived under a ceiling) | after the deep-duck envelope check |

The guards are not hardware verification. The site check, the remote, the physical E-stop and a gantry or spotter
stay mandatory at every rung. Before H5, run this branch's panel on the deploy PC: its GPU and CPU set the latency.

## What the guards cannot fix (requests stay open)

- **Self filter.** The self mask hides leaks next to the body; it does not fit the filter to the real body. The
  leak test in CI (0 leaked points in 1,000 frames) and per-link fitting to the robot's meshes stay with perception.
- **Calm stands can repeat.** A memory that never clears can keep the robot standing (safe, but the walk times out).
- **The mask can hide a real object.** A real object hugging the robot, inside the 10 cm band and only ever seen
  up close, is hidden from selection, so the site check stays a NO-GO gate. An example is a strap or a small gantry
  part; a gantry frame 0.25 m behind the pelvis is outside the band. The wall-close course shows the cost: a hidden
  leak let a leaky-filter robot move and brush a wall 0.33 m away.
- **Calm stops are learned stop candidates or a synthesized stand.** The planner team's request for a calm stand
  stays open, and so do training on real-robot memory (self-returns, noise, phantom overheads) and dense rooms.
- **Arms still brush columns.** Moving the anchor back to the measured state after a 0.4 m miss does not prevent
  it. An execution check (`exec_model`) or a trackability-aware planner is still needed, along with SONIC's crouch
  envelope.
- **The memory never forgets,** so people who walked through stay as obstacles. Decay or free-space carving is not
  in this branch.
- **Walking drops are bounded, not free.** Each costs a planner latency of stream; after 2 in a row the calm path
  takes over.
- **The envelope comes from authored courses.** Recalibrate it from robot sessions (`tools/session_report.py`
  on deploy-kit) before H6.

## Reproduce

```bash
.venv/bin/python -m pytest tests/test_guards.py tests/test_node_guards.py
.venv/bin/python tools/robust_panel.py --out runs/robust/panel1 \
    --conditions tag,kit,kit-guards,leaky-guards,kit-walls,kit-slow,kit-guards-slow,kit-walls-slow --seeds 3
.venv/bin/python tools/robust_panel.py --realtime --out runs/robust/rt1 \
    --conditions tag,kit,kit-guards-slow,kit-walls-slow --courses open-2m,room,l-shape,slalom,dense --seeds 2
```

On the robot, the node takes `--guards walls --goal-lookahead 1.0`. Its events and status show every guard action.
