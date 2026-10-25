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

## Evidence: the deploy PC (RTX 4070 Ti SUPER)

The deploy-PC session merged this branch into deploy-kit, on a local branch `integration`, with 124 tests passing.
It then ran its lab-site twins, the courses built from what the lab G1's lidar saw. Each condition is three
real-time runs through `./g1 sim` with `--guards walls --goal-lookahead 1.0` (2026-10-24).

| twin | arrived | lowest pelvis (m) | contact | plan latency p95 (s) | guard actions |
|---|---|---|---|---|---|
| original start inside the clutter (without guards the merged box held the robot) | 3/3 | 0.66–0.70 | 0 | 0.08–0.12 | none needed |
| 1.5 m forward, 5 cm margin | 3/3 | 0.62–0.64 | 0 | 0.09 | 5–6 envelope re-selections and 1–4 walking drops per run |
| 1.5 m forward, 3 cm margin (the shoulder leak behind the back-off) | 3/3 | 0.65–0.70 | 0 | 0.08–0.09 | first plan: 1 leak cell masked, then safe |

Across these runs:
- **Health:** no plan over 0.32 s, no plan with nothing safe, no stall risk, no anchor miss and no mask refusal.
- **Shoulder leak:** at the 3 cm margin it sits inside the self mask's band, and the mask alone recovers the stack.
  With the leak and no guards, this twin backed off and crouched.
- **No plan with nothing safe occurred**, so the 4070's latency on one could not be measured here.

**The gate number.** The deploy PC's real-time panel answered it: on the 4070, a plan where nothing is safe takes
**0.42 s, over the 0.32 s gate**. These are the frozen resampling plus the stop pool. The source was one unguarded
dense-room run: 29 such plans at 0.421–0.424 s, 30 plans over the gate, and it ended in a collision. No guarded run
produced one, and the slowest guarded plan was 0.087 s.

The rest of that panel matched this dev box:

| condition | done | fall | collision | unmeasured |
|---|---|---|---|---|
| kit, no guards | 2/6 | 3 | 1 | 0 |
| kit + `--guards walls`, slow | 3/3 measured | 0 | 0 | 3 |
| kit + `--guards walls`, slow, rmem_max raised (room, dense rerun) | 4/4 | 0 | 0 | 0 |

The 3 unmeasured runs got no lidar. The node fused 0 points (one frame in one run) against 1.3–1.7 M in the
others, because that PC's `net.core.rmem_max` is still 212,992 bytes against lidar frames of about 0.9 MB in
closed rooms. The DDS reader then stalled for the whole run. The panel now scores such runs as `no_lidar`,
unmeasured rather than failed, and the node reports `prescan_blocked` every 5 s when the walk cannot start.

The deploy-PC user then raised `net.core.rmem_max` to 33,554,432 (`rmem_default` 8,388,608). The rerun of room and
dense came out as follows:
- 0 UDP receive-buffer errors, against about 39 k before;
- 2.5–2.8 M lidar points fused per run;
- 4/4 arrived, lowest pelvis 0.64–0.67 m, no contact;
- plan p95 0.084–0.088 s and at most 0.093 s, with no plan where nothing was safe.

On the 4070, `--guards walls --goal-lookahead 1.0` now stands at 6/6 on the panel courses (open floor, room, dense)
and 9/9 on the lab twins. The 0.42 s plans came only from unguarded runs.

### Where a no-safe plan's time goes (4070, offline)

The deploy PC replayed exact node inputs (`--debug-plans` dumps) through the deployed fused fp32 planner, forced
into the fallback. Times are medians in ms, uncontended, from `tools/profile_fallback.py` on deploy-kit:

| state | total | K go | 3K resample | K stop pool | clearance | grounding |
|---|---|---|---|---|---|---|
| lab-site twins (6–9 guide boxes) | 242 | 41 | 104 | 40 | 52 | 2 |
| dense room, at the pose of the 0.42 s run (31 boxes) | 336 | 58 | 154 | 50 | 53 | 20 |
| a normal plan, lab twins (for comparison) | 54 | 41 | – | – | 10 | 1 |

- **The resample dominates.** The 3K resample is about 46% of a fallback, plus about 10% for its share of the
  clearance (CPU, about 0.65 ms per candidate).
- **Contention adds about 25%.** Real time measured 0.42 s against 0.34 s uncontended, so the dense fallback is
  over the gate either way.
- **A shortcut has little to gain.** Skipping the resample from a standstill under `--guards` would save about
  150 ms. It would also change the candidate set, and it helps least where it applies: from a standstill the robot
  is holding a stand while it waits, so latency there is not safety-critical. While walking, the frozen fallback is
  kept as it is. So the faster fallback stays a planner-side request.

### More on the 4070: lock-step and real lidar

**The remaining courses in lock-step.** Pillar, slalom, table and wall-close, 3 seeds each:
- kit, no guards: 9/12, 3 collisions.
- `--guards walls` + slow: 11/12, 1 slalom collision.
- No falls in either.
- Guarded runs did meet a plan with nothing safe twice: a slalom (0.29 s) and wall-close (0.28 s, ending in a calm
  stop). Lock-step plans run about 1.25× faster than real time, so those would be about 0.35 s on the robot.
- The guards' view preparation on that CPU: 9.0 ms at the median, 9.6 ms at p95.

**The self mask on real data.** The deploy PC re-fused its standing recordings of the lab G1 through the node's
path (rec2, rec3; the robot hanging in the gantry). At the 5 cm margin there was one near-only band cell, at the
harness strap 0.20 m behind the pelvis (support 1.17 m). It was not masked, because its evidence reaches outside
the band toward the gantry. At 3 cm there were two such cells, still not masked. On real data the mask has not
acted yet; it acted only in the 3 cm lab twin, and the strap stays visible as intended. SONIC's own stand pose is
the capture still missing, listed for H3 in DEPLOY_KIT.md.

## The lab door test (2026-10-24) and two guard changes

On the robot the door test (beam underside 1.17 m, opening 1.0 m, `--guards deploy`) had nothing safe from the
first plan. The safety rope from the back hook to a spotter 0.6 m behind showed up as a column of cells from
0.19 m behind the pelvis. The self mask rightly kept it: 17–20 cells, extending past the band. The guards held a
calm stand three times, then took a calm stop. The robot then no longer read as standing, so the frozen
max-clearance fallback streamed every 0.35 s for 6 s and backed the robot off 0.3 m, chasing clearance from a rope
that follows it, until the operator's E-stop.

Two changes in this branch:
- **CalmLatch.** A no-safe streak that began from a standstill stays calm until a safe candidate appears. Every plan
  is judged as from a standstill, and there are no walking drops. Together, the streak's calm stops move the pelvis
  at most 0.25 m from where it began; after that the robot only settles to stand. The node reports `calm_hold` and
  `calm_release`, and the status carries `calm_hold`.
- **Walls mode now routes around overheads too.** The sensor route plans on support only. Under `--guards walls`
  it led straight at a door beam that the planner, seeing a wall, would not pass, and the head hit the beam. The
  route now reads the same walls (`WallsRouteMemory`).

The authored door courses (`door`, `rope-behind`, `door-rope`; the rope is static here, while on the robot it
follows the robot) plus the nine earlier courses, three seeds each:

| course | kit, no guards | kit + deploy | kit + deploy, slow | kit + walls, slow |
|---|---|---|---|---|
| door | 2/3 (1 collision) | 3/3 (ducks, pelvis 0.52 m) | 3/3 | 3/3 (around the frame; 0/3 before the route fix) |
| rope-behind | 3/3 | 0/3, stands | 0/3, stands | 0/3, stands |
| door-rope | 3/3 | 0/3, stands | 0/3, stands | 0/3 (2 stand; in 1 the robot drifted 0.11 m back into the static rope during 7 s of standing) |
| the nine earlier courses | 20/27 (9 falls) | 25/27 | 26/27 | 26/27 |

Without guards, the static simulated rope is escaped by lunging forward. On the robot the rope followed, and the
same choice backed the robot off into an E-stop. With the guards, an object hugging the robot means the robot
stands and reports `calm_hold`: safe, but no progress. The real fix is clearing what is behind the robot, or
blinding the rear sector (`--rear-blind-deg`, on deploy-kit's integration branch) for robots with a spotter
behind. Adding the latch changed no outcome on the nine earlier courses.

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
- **A plan with nothing safe is over the gate on the deploy GPU** (0.42 s on the 4070). The guards make these plans
  rare (none in the guarded panels) and plan the next one from the measured state, but they do not make them
  fast. A cheaper fallback, such as fewer resamples or a stop pool first, is a planner-side change.
- **Lidar on any deploy PC needs a large UDP receive buffer.** The lab deploy PC now runs `net.core.rmem_max` =
  33,554,432; persist it in `/etc/sysctl.d/`. With the stock 212,992 bytes, closed and dense rooms lose whole lidar
  frames, and the node then waits in PRESCAN and reports `prescan_blocked`.
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
