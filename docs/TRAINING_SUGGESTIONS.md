# Deployment findings for training and distillation (lab-g1, 2026-10-23 to 10-25)

What the first real deployment of the Flow student taught us, written for the planner's training and distillation
line (owner from 2026-10-25: the Flow driver session "Code workspace"). Each item is: what we saw (with numbers and where the data is), then what training could change. The deploy-side
fixes already in place are listed so training can absorb them instead of the deployment carrying them forever.

Deployed: `student-s2-kcsc_s0` (bundle deploy-2026-10-23), K 16, ODE 10, 2 chunks, sensor map / selection / guidance,
`select_margin_m` 0.08, fallback `resample_turn`, fused fp32 planner on an RTX 4070 Ti SUPER, SONIC C++ runner,
`goal_lookahead_m` 1.0 (slow walking), deploy-side guards (`g1deploy/guards.py`). Robot: lab-g1 (G1 29-DoF, Dex3), head
MID-360 mounted inverted, read on the deploy PC through Livox's driver (this firmware has no Unitree lidar service), leg
odometry, robot in a mobile gantry with a safety rope to the back hook.

## What happened on the robot

| step | result |
|---|---|
| H3 SONIC stand, H4 streamed walk | passed; one catch-up reset; tilt ≤ 0.14 rad |
| H5 student, 1.5 m goal | **arrived**; 22 plans, p95 0.13 s; 10 envelope reselects + 1 envelope drop |
| H5 student, 2.0 m goal | **arrived**; 1 envelope reselect; first real self-mask (2 cells) |
| door (beam underside 1.17 m, opening 1.0 m), 2 attempts | **no motion**: every plan no-safe from the first one; cause below (1.4) |

Simulation of the same scenes (real-time deploy chain, twins rebuilt from the robot's own lidar): 7/7 arrived, 0 N
contact; on the door twin the student ducked through 2 times in 4 and detoured around the frame 2 times after 20–43 s
of hesitation (1.3).

## 1. Behaviours training could fix

### 1.1 Crouching and lunging on open floor
**Seen.** On the real 1.5 m walk, 11 of 22 frozen choices left the walk envelope with nothing overhead: knee
2.05–2.18 rad, pelvis 0.56–0.60 m (standing 0.77 m). The guard re-selected a candidate inside the envelope 10 times and
dropped the plan once. In the lab-site twins with the self-filter leak, the frozen choice crouched, lunged and fell
(3/3) on a clear 2 m path.
**Training.** Make posture depend on overhead clearance: penalise pelvis drop / knee flexion beyond the clean-walk
envelope when no ceiling is within reach (teacher reward or a student auxiliary loss), filter demonstrations that
crouch without a reason, and report an *envelope-violation rate* (pelvis < 0.62 m or knee > 2.0 rad with no ceiling
within 1 m) next to success.

### 1.2 The no-safe fallback
**Seen.** When no candidate is safe the frozen planner samples 16 + 48 + 16 candidates and streams the max-clearance
one. Its choices were deep crouches (pelvis 0.38, 0.23 m: dropped by the node) and steps backwards away from an
"obstacle" that followed the robot. Cost on the 4070, uncontended: 242 ms (lab twins) / 336 ms (dense room, 31 boxes);
0.42 s in real time, over the 0.32 s gate; the 48-candidate resample is ~46 % of it (`tools/profile_fallback.py`,
DEPLOY_KIT.md §10). Deploy side now: guards `calm_stand` / `calm_stop` and a calm latch (robustness 2318ae7).
**Training.** Give the student an explicit hold: no-safe states in the training set with the teacher's label "stand
still, settle to a symmetric stance"; train the stop pool to produce calm stands rather than crouches; a learned
clearance/value head to rank candidates would remove most of the resample (one batch instead of three).

### 1.3 Hesitation at narrow passages under a beam
**Seen.** Door twin (opening 1.0 m, beam underside 1.17 m, start lined up 1.4 m before it): the student either ducked
through within ~5 s (pelvis 0.55–0.59 m, no contact) or stood 0.6–0.7 m before the beam shuffling sideways for 20–43 s
and then walked around the outside of the frame, every plan "safe". With the rear half of the lidar blank it once
ducked through but brushed the frame (49 N). The real attempts never got that far (1.4).
**Training.** More doorway / beam-in-a-frame scenes with tight lateral clearance (opening 0.8–1.2 m, beam 1.05–1.25 m,
slightly rotated frames), detours made longer or impossible so the duck is required; reward progress and penalise
oscillation (net displacement over a 2–4 s window); arm/hand clearance terms while ducking (the brush was lateral).

### 1.4 Things attached to or hugging the robot
**Seen.** (a) The safety rope runs from the back hook to a spotter at 1.0–1.3 m height; the MID-360 sees a line of
cells from 0.19 m behind the pelvis, the height field makes it a column touching the robot, and it follows the robot:
0 of 80 candidates safe, best go clearance 0.04 m; re-planning the same inputs with the rear half blanked: 13–16/16
safe, chosen pelvis 0.41–0.50 m (it ducks). (b) A 3 cm self-filter margin leaked right-shoulder returns into two cells
at 1.06 m beside the robot; the first plan had 0/16 safe candidates. (c) A spotter or rope bar 0.15–0.25 m behind the
back while the robot hung in the rope stayed in the memory.
**Training.** Augment the perception input with near-body artefacts: 1–5 leaked self cells around shoulders/hands, a
thin tether line 0.2–0.8 m behind the back at 1.0–1.3 m that moves with the robot, a person 0.5–1.5 m behind. The
desired behaviour is to keep walking forward (they are behind and never in the swept volume), not to freeze or back
away.

### 1.5 Partial observability (a blind rear, head-shell bands)
**Seen.** Ignoring the rear half of the lidar (our tether workaround) made the frozen planner slower and crouchier:
13/13 arrived in real-time simulation, but with 2–12 envelope corrections per walk where full lidar had 0–1, and in
lock-step any rear sector (90–180°) broke the slalom (0–1/3 vs 2/2). Blanking only the near rear (< 1 m, rope and
spotter) behaved like full lidar (slalom 5 s clean, 0–5 corrections) while the static-rope course went from "never
walks, 273 no-safe plans" to "arrives in 2 s" (DEPLOY_KIT.md §11): the planner is sensitive to large unobserved
regions, not to a small near-body hole. The real head shell also blocks azimuth −45…−35° and +40…+50°.
**Training.** Occlusion augmentation so a missing sector is in-distribution: random unknown sectors (rear 90–180°,
the two head-shell bands), random max range 3–4 m, unknown cells marked as unknown (not free) exactly as the deploy
memory does.

### 1.6 Guidance boxes
**Seen.** Flow's `memory_boxes` wraps each connected component in one box. Real rooms connect: the lab became one
3.4 × 6.8 m box containing the robot's start; the authored closed room one 5.4 × 6.1 m box. The planner then had no
safe candidate. Deploy side: guards split components into rectangles of their own cells (`split_boxes`).
**Training.** Train and evaluate with the same split guidance the robot uses (or make the guidance robust to a box
containing the start), otherwise train and deploy see different guidance in every cluttered room.
**Status (Flow, 2026-10-25).** Flow reports this is already handled in its newer students' perception configuration (details with the Flow team); a bundle built from it absorbs the deploy-side split.

### 1.7 Odometry
**Seen.** Leg odometry under-counted real walks by 20–40 %: 1.5 m goal (leg 1.43 m, LIO 1.86 m), 2.0 m goal (leg
1.72 m, LIO 2.45 m, operator ~2.3 m). The robot overshoots its goals by the same amount. FAST-LIO validated standing
(0.09 cm drift in 30 s) but once diverged while people handled the robot.
**Training.** Odometry scale (±30 %) and yaw-drift noise in training and evaluation; arrival scored on the true pose.

### 1.8 Arrival stance and motor heat
**Seen.** Walks end in a staggered stance; SONIC holding it for ~10 minutes heated the right ankle motors from 45 to
83 °C (roll) and 76 °C (pitch) at ~20 Nm continuous.
**Training.** Arrival and stand plans should end in a symmetric double-support stance (penalise asymmetric final
stance, a short settle phase).

### 1.9 Planner speed
**Seen.** Normal plans 0.08–0.14 s p95 on the 4070 (gate 0.32 s); no-safe plans 0.24–0.34 s uncontended. The
real-time rehearsals show the success cliff right after the gate (READINESS.md).
**Distillation.** Consistency / progressive distillation of the denoiser to 2–4 steps, or fewer candidates with a
learned ranker (1.2); bf16 is already evaluated (85/98) if numerics allow.

## 2. Distillation (teacher → student)

- **DAgger on the deployment perception stack**, not an idealised map: the node's `CloudMemory` with the mesh self
  filter at 5 cm, the MID-360 scan pattern from the inverted head mount with per-point timing and deskew, mount errors
  of ±1° / ±2 cm, 10 Hz frames, leg-odometry noise (1.7), occlusion bands (1.5), near-body artefacts (1.4).
- **Distil the guards into the student** so deployment needs fewer corrections: the walk envelope (1.1), the calm
  fallback (1.2), split guidance (1.6), the anchor check. Track the *guard-intervention rate* as a training metric.
- **Start from standing** and re-plan from a stand (real robots always do; 94/98 standing starts).
- **Real scenes as validation**: twins rebuilt from the robot's own lidar (`tools/twin_from_recording.py`;
  `sim/courses/lab-door-aligned`, `lab-start-spot`, `lab-site-*`), plus the authored `rope-behind`, `door`,
  `door-rope`, `dense`, `room` courses (`tools/robust_panel.py`).

## 3. Evaluation we would trust for the next bundle

The real-time deploy chain (SONIC C++ runner, the node, MuJoCo, raw MID-360 on DDS), guards on, standing starts,
`goal_lookahead_m` 1.0; real-scene twins + the robustness courses; per run: arrival, contact, falls, envelope
violations, guard interventions, no-safe plans, hesitation time (no net progress), plan latency p95 on the deploy GPU.
`./g1 sim --layout NAME --guards deploy --goal-lookahead 1.0 --script ...` and `runs/sim-studio/run_experiments*.sh`
do this today.

## 4. Data on the deploy PC (ask before copying; sessions are 1–5 GB)

- Real sessions: `runs/robot-day-20261024/{h3,h4,h5-1,h5-2m,door-1,door-2}`: lowstate 1 kHz (35 motors), lowcmd,
  raw MID-360 frames + lidar IMU, LIO odometry (door-1/2), node status, every plan's exact inputs and output
  (`plans/`, `--debug-plans`), runner CSVs. Layout: `g1deploy/recording.py`.
- Offline tools: `tools/debug_plan.py` (re-plan a recorded state, draw all candidates), `tools/profile_fallback.py`,
  `tools/guard_mask_real.py`, `tools/lio_replay.py`, `tools/twin_from_recording.py`.
- Simulated experiments with videos: `runs/sim-studio/` (`videos/`), replayable in Deploy Studio.
- Code: `linjiw/g1-student-deploy` branch `integration`; guards on `robustness`.
