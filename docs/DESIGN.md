# G1 student deployment: design

Status: 2026-10-07, design + sim2sim build. Nothing here has run on a robot.

## 1. What is being deployed

**Policy = the round-n lidar student of `~/flow`** (`S2-guide10m8-grow06`): a flow-matching
whole-body planner (9.4 M params, DiT dim 256, depth 6) that proposes K = 16 one-second plans
from 4 frames of measured history, a 3×31×61 ego height/ceiling map built from a head-mounted
Livox MID-360, and a local goal; clearance-gradient guidance against boxes extracted from the
lidar memory (w = 1.0, 8 cm inflation, 0.6 m overhead fragment growth); selection by an 8 cm
height-field margin; leg-IK grounding. The released **GR00T SONIC** g1 encoder + decoder tracks
the plan.

| what | value | source |
|---|---|---|
| student checkpoint | `~/flow/artifacts/student-s2-kcsc_s0/model.pt` sha256 `ad9c5b32…` | receipt.json |
| planner source | frozen snapshot `~/flow/artifacts/closed-loop-v2-main/source` | panel v2 packet |
| planner config | `artifacts/closed-loop-test-v4/config.json` arm `student-guide10m8-grow06` | F37/F38 |
| SONIC | release step 41550, `last.pt` sha256 `e6bdab3f…`; fused ONNX 1570→29 `sonic_release_g1.onnx` sha256 `325031d7…` (ONNX vs torch 2.4e-6) | hoi-retarget `release_export_v2` |
| Isaac held-out | 83/98 (test-v1, privileged route); panel v2 partial 77/99 privileged route, 75/99 lidar route, 13 tilt falls each | F38, `closed-loop-v2-main` |

The student is a *planner*, not a joint-level policy: SONIC is the joint-level policy. Deploying
the student therefore means deploying three things that must agree: the planner, the SONIC
tracker, and the stream between them.

## 2. Runtime contract (taken from the Isaac callback, `closed_loop/callback.py`)

Per 50 Hz control tick:
1. measured world qpos (pelvis xyz, quat wxyz, 29 joints MuJoCo order) → history buffer.
2. lidar memory step (10 Hz captures, 0.1 s latency, 5 cm world grid, ±6 m).
3. every 8 ticks (0.16 s): route waypoint → goal_fn; `memory_boxes` → guidance boxes;
   `planner.plan(history, …)` → 50 frames (1.0 s) of world qpos + rotation + joint velocity,
   frame i at anchor + (i+1)·20 ms.
4. `feed.target(plan, anchor, j)` → (10, 67): rows at 0, 0.1, …, 0.9 s ahead (frame j+5k−1, clamped
   to the last plan frame), joint pos/vel, 6D of R_measured^T R_plan.
5. SONIC g1 encoder ← (joint pos, joint vel) in IsaacLab order + 6D; decoder ← token + 10-frame
   proprioception history; action → PD targets.

**SONIC never receives root position or height.** It tracks joint trajectories and pelvis
orientation relative to the measured pelvis; xy is open loop and corrected only by replanning
from measured odometry. The planner, not SONIC, needs world odometry.

## 3. Target architecture (hardware)

```
 MID-360 ─► cloud ─► body crop ─► CloudMemory (label-free support/ceiling grid) ─┐
    │                                                                              │
    └─► LIO odometry (FAST-LIO2, gravity aligned) ─► pose ─┐                      │
 rt/lowstate (joints, pelvis IMU) ───────────────────────► StateBuffer (50 Hz world qpos)
                                                             │                    │
                                                             ▼                    ▼
                                        StudentPlanner (async thread, replan 0.16 s, K16 guided)
                                                             │ plan: 50 frames, world frame
                                                             ▼
                          ReferenceStreamer: latency compensation, 50 Hz frames, frame_index sync,
                          tail padding (≥ 46 frames ahead), SONIC ZMQ protocol v1
                                                             │  tcp PUB bind :5556, topic "pose"
                                                             ▼
          SONIC C++ runner (lucid-g1-deploy, TensorRT, fused 1570→29) 50 Hz ─► rt/lowcmd 500 Hz ─► G1
  SafetySupervisor: tilt, stream age, plan sanity, odometry health ─► hold-stand stream / runner damping
```

Stream wire format (`zmq_packed_message_subscriber.hpp`, protocol v1): one ZMQ frame
`"pose"` + 1280-byte null-padded JSON header + little-endian fields `joint_pos [N,29]`,
`joint_vel [N,29]` (IsaacLab order), `body_quat [N,4]` wxyz, `frame_index [N]` i64, optional
`timestamp_monotonic`, `catch_up`, `heading_increment`. The runner consumes one stored frame per
50 Hz tick, reads frames k, k+5, …, k+45, hard-overwrites from the new chunk's first
`frame_index`, and **resets playback + re-captures heading** if a chunk starts at/before its
window start or jumps ahead. It stalls ("waiting following motion") when fewer than 46 frames are
ahead.

## 4. Two MuJoCo test paths

**Path A — lock-step evaluator (`sim/lockstep.py`).** MuJoCo 200 Hz physics, Python SONIC
(onnxruntime on the fused ONNX; observation builder ported from lucid `tools/mujoco_player.py`,
parity-checked against the export parity vectors), the frozen planner, and the Isaac
`TaskScorer` rules, on the same `test-v1` / `heldout-v2` task files. Deterministic, ~CPU only,
and it can inject deployment effects one at a time:
- `lidar`: `analytic` (= Isaac: labelled hits on task boxes) | `mjray` (MuJoCo ray casting on the
  real scene incl. the robot's own body, then label-free `CloudMemory` + body crop).
- `latency_s`: plan becomes available L later while the previous plan keeps playing; frames are
  shifted exactly as the real streamer would shift them.
- `odom`: exact | noisy (random-walk xy/yaw, as `StateEstimate` in flow).
- `start`: `recorded` (Isaac panel start mid-walk) | `stand` (default pose at rest, as on hardware).

**Path B — real-time rehearsal (`sim/realtime/`).** The lucid DDS MuJoCo simulator
(`sim/run_robot_sim.py`) with the task boxes added, publishing `rt/lowstate` + ground-truth
`rt/odostate` + a simulated MID-360 cloud; the unmodified-protocol SONIC C++ runner with
`--input-type zmq`; and the planner node running asynchronously at wall-clock latency. Same
processes and wire protocols as the robot except the sensor drivers.

## 5. Gates (each must pass before the next is trusted)

| gate | test | pass rule |
|---|---|---|
| G0 bundle | planner + SONIC files sha-match the evaluated artifacts; planner loads and plans on CPU and GPU | exact hashes; plan shapes |
| G1 SONIC parity | Python 1570-obs + ONNX vs export parity vectors | max abs action err ≤ 1e-4 |
| G2 tracker gap | open-loop replay of recorded Isaac `references` in MuJoCo from the same start | report pelvis path / tilt deviation; falls must not exceed Isaac's on the same episodes |
| G3 closed loop = Isaac protocol | Path A, analytic lidar, recorded start, test-v1 98 eps | report paired vs Isaac 83/98; a large drop means a MuJoCo/Isaac gap to fix before G4 |
| G4 deployment effects | Path A with mjray lidar, latency 0.16/0.3/0.6 s, odom noise, standing start | each effect reported separately and combined; pick the latency budget |
| G5 real-time rehearsal | Path B end to end | completes episodes with the C++ runner; outcomes consistent with G4 at the measured latency |
| G6 hardware readiness | supervisor unit tests, gantry checklist, network + lidar + LIO bring-up | checklist in `RUNBOOK.md` |

## 6. Risks found while designing

1. **Latency vs plan length.** Plans cover 1.0 s; SONIC reads 0.9 s ahead. Isaac ran lock-step
   (planner latency 0, physics paused). Measured planner latency: 0.29–0.32 s p50 on the RTX 5080
   (shared), 0.67 s on 8 CPU threads. At 0.3 s the last ~17 of 46 look-ahead frames are past the
   plan end → the stream must pad (= Isaac's clamp) or plan 3 chunks, and the robot decelerates.
   G4 measures this; options are 3 chunks, fewer ODE steps, TensorRT for the planner, or
   anchoring each plan at the *predicted* state at its arrival time.
2. **Frame sync.** The runner never reports its playback frame; a chunk that starts ≥ 5 frames
   too early triggers a catch-up reset (playback to 0, heading re-captured). Plan: patch the runner
   to publish `motion_frame` on its `g1_debug` output; until then derive it from `g1_debug` tick
   count.
3. **Heading frame.** The runner aligns the stream's yaw to the IMU yaw at stream start; odometry
   yaw and IMU yaw drift apart on hardware. Correct with `heading_increment` from the LIO−IMU yaw
   difference.
4. **Start state.** Every Isaac panel starts from a recorded SONIC walk at ~0.6 m/s. The robot
   starts standing. Untested regime until G4 `start=stand`.
5. **Tilt falls on deep ducks** (13/99 student episodes in panel v2 partial, 0.80–0.85 rad, beam
   families). First hardware runs should use ground obstacles only and a supervisor tilt limit.
6. **Lidar labels.** The Isaac memory knows which box each ray hit (ground / solid / overhead).
   A real cloud does not, and it contains the robot's own arms and shoulders. `CloudMemory` must
   classify cells by height structure alone; G4 compares it with the labelled memory.
7. **Mount.** The real MID-360 is upside down; the URDFs disagree on its pitch and offset
   (`allterrain/perception/visibility.py`). Calibrate on the robot.
8. **Odometry.** Unitree's odometry is not available after the controller is released; use
   FAST-LIO2 with the MID-360 IMU, gravity-aligned, ≥ 50 Hz after IMU propagation. History
   velocities are finite differences of pose, so odometry jitter enters the planner directly.
9. **Route.** The best Isaac arm plans its route on the true boxes. On hardware either the course
   is measured into the odometry frame, or the lidar route (`route_source: sensor`, 0.3 s
   pre-scan) is used: 75/99 vs 77/99 in the panel v2 partial read.

## 7. Repository layout

```
g1deploy/            runtime library (no Isaac dependency)
  bundle.py          locate + verify the frozen planner/SONIC bundle
  contract.py        joint orders, gains, default pose, action scale
  sonic.py           SONIC 1570 observation + onnxruntime policy (Python tracker)
  feed.py            plan → per-tick target; plan → 50 Hz stream frames
  zmq_stream.py      SONIC protocol v1 packer / publisher / debug subscriber
  planner_service.py student planner session: history, route, memory, sync + async replanning
  perception/        MID-360 pattern + mount, label-free CloudMemory, body crop
  scoring.py         TaskScorer rules (fall / contact / goal / deadline)
  safety.py          supervisor
sim/                 MuJoCo scene builder, ray-cast lidar, lock-step evaluator, real-time rehearsal
tools/               freeze_bundle.py, eval_panel.py, replay_isaac.py, compare.py, render
tests/               unit + parity tests
docs/                DESIGN.md (this), RUNBOOK.md, HARDWARE.md
```
