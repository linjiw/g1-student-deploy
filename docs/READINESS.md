# Deployment readiness review — lidar student on the G1 (2026-10-08)

**Verdict.** The student, its SONIC tracker and every interface the robot exposes are packaged, tested in a
real-time MuJoCo rehearsal that uses the robot's own protocols, and ready for the hardware ladder up to H5
(walking on open floor with the student in control, gantry/spotter, remote in hand). Obstacle runs (H6) are
reasonable for ground obstacles; **overhead obstacles (beams) are not ready for hardware**. Under the most
realistic simulated deployment (real MID-360 model, standing start, lidar route, LIO-level odometry, 0.32 s planner
latency) the student scores 75/98 (n.s. vs Isaac's 83/98), and 84/98 without the planner latency — after two
perception fixes found in this review. Hard gates: planner p95 ≤ 0.32 s and a calibrated lidar mount.

## 1. What is established (simulation evidence)

| claim | evidence |
|---|---|
| deployed files = evaluated files | bundle sha256 manifest; student `ad9c5b32…`, SONIC ONNX `325031d7…` (G0) |
| SONIC runs identically outside Isaac | ONNX parity < 1e-4 (G1); 98 Isaac reference streams replayed in MuJoCo: pelvis z RMSE 1.3 cm, no falls (G2) |
| the student transfers to MuJoCo | 81/98 vs Isaac 83/98 on the same tasks/seeds, p = 0.77 (G3); 83/98 with a 0.3 s lidar pre-scan |
| it starts from standing (as on the robot) | 94/98 (better than Isaac's mid-walk starts) |
| it needs no privileged route | lidar route 88/98 |
| planner latency is handled | 0.32 s latency: 67/98 naive → **79/98 with predictive anchoring** (n.s. vs Isaac); two plans in flight is harmful (32/98) |
| the C++ deploy stack streams it correctly | stream rows proven equal to the Isaac feed rows (unit test); a recorded beam episode replays end to end through the runner (TensorRT) + DDS sim; closed loop passes two beams with the robot's raw lidar topic |
| the robot's lidar is usable as published | `rt/utlidar/cloud_livox_mid360` decoded from the measured G1 layout (raw inverted frame, ns point times, placeholders, wrong clock), deskewed + registered by the node, self-filtered per 5 ms slice |
| the mount can be calibrated on the robot | floor fit recovers an injected 2° pitch + 1° roll + 2 cm to < 0.1° / < 5 mm |
| the robot's own body is not an obstacle | mesh-capsule filter per time slice: 0 returns left within 0.3 m of the pelvis; Dex3 or rubber hands |
| safety reactions | tilt > 0.75 rad → e-stop (all falls ≥ 0.80; ~1% of successes exceed 0.75); stale state → e-stop; lidar loss → stand (deferred near ceilings); arrival → stand; restart-safe (5 node tests with fake inputs) |
| operator override | wireless remote via runner `gamepad_manager`: D-up mid-walk at 1.38 m/s → SONIC stand in ~1.5 s; Select = e-stop |

## 2. What costs success (G4, lock-step MuJoCo, 98 held-out episodes, Isaac = 83)

| condition | goals | reading |
|---|---|---|
| idealised lidar (MuJoCo rays, 2° grid) | 83 | = Isaac |
| real MID-360 (pattern, per-point timing, torso mount), after the perception fixes | 79 | n.s. (was 73 before the along-ray self filter + 3.5 cm face push) |
| standing start + real MID-360 | 89 | the robot's real start is easier than Isaac's mid-walk start |
| standing start, slower walking (lookahead 1.0 m) | 93 | **slow mode costs nothing** — use it on the robot |
| **combined: real MID-360 + standing start + lidar route + LIO-level odometry + 0.32 s latency** | **75** | n.s. (p = 0.12) |
| … the same without planner latency | 84 | = Isaac: **latency is the remaining gap** |
| planner latency 0.16 / 0.32 / 0.36 / 0.50 s (predictive anchoring) | 79 / 79 / 70 / 52 | cliff right after 0.32 s → `g1 bench` p95 ≤ 0.32 s is a hard gate |
| LIO-level odometry noise | 77 | small |
| leg odometry (joints + IMU only) | 74 | −9 (p = 0.02): fine for short first walks, use LIO afterwards |
| head shell blocking steep-down front rays (±45°, ≥ 30° down) | 79 | no cost |
| **front ±45° fully blind** | **58** | large loss → measure the real occlusion (`g1 calibrate --png`) |
| mount error after calibration (0.5°, 1 cm) | 80 | no cost |
| **uncalibrated head, 3° pitch** | **15** | **catastrophic → `g1 calibrate` is mandatory** |

**Deploy PC measurement (this machine, RTX 5080, 2026-10-08, GPU shared at ~100% with other training jobs):**
deployed planner (K 16, 10 ODE steps) p50 0.31 s, p95 0.36 s, p99 0.38 s, max 0.42 s (n = 80); with two more
evaluation workers on the GPU p95 0.39 s. That is just past the cliff. TF32 does not help. Faster configs measured
under the same load: K 8 → p95 0.33 s, 6 ODE steps → 0.28 s, both → 0.24 s; their closed-loop cost is being
evaluated at those latencies (`jobs/g4v.sh`). Before robot runs: free the GPU (pause other training) and re-run
`./g1 bench`, or deploy a faster config once it passes the panel.

## 3. Hazards found in rehearsal (and what was done)

1. **Standing up under a beam.** Operator takeover (D-up) 1 m before a beam: SONIC's planner stands the robot
   up into it (566 N). → Operator rule: take over only in open space; near overhead obstacles let the student
   finish or e-stop with a gantry. The node's own soft stop now defers standing near remembered ceilings.
2. **Front blind sector.** One G1 unit had the head shell blocking ±45° of azimuth; a fully blocked front sector
   leaves no returns ahead. → Measure on our unit (`calibrate_mount.py --png`); G4d quantifies the cost.
3. **Python cloud publishing / CPU load.** Lidar frames > 1 s late under heavy load make the node stand
   (correct, but it ends the run). On the robot the driver is C++ on PC1; keep the node machine dedicated.
4. **Restart bugs** (perception thread, stale jobs, carried-over run state) — found by the supervisor tests, fixed.

## 4. Not established until on the robot

- `rt/utlidar/*` keeps publishing after `ReleaseMode()`; the 22-byte layout on our firmware; MID-360 vs MID-360S.
- Real mount pose and occlusion; real self-returns with our hands.
- LIO (Point-LIO on `/utlidar/*`) quality while walking; the ROS 2 bridge is untested here.
- Planner latency on the deploy machine (`tools/bench_planner.py`; p95 ≤ 0.32 s is the tested budget).
- Sim-to-real of SONIC itself on this G1 (lucid's runner has stood in a gantry once; no policy armed yet).
- Real foot slip (leg odometry drifts 12% in MuJoCo).

## 5. Go / no-go for the first student walk (H5)

Go only if all hold: bundle verifies; `pytest` green on the deploy PC; `bench_planner.py` p95 ≤ 0.32 s;
`check_robot.py` shows 500 Hz lowstate and 10 Hz lidar after `ReleaseMode()`; floor levelled by
`calibrate_mount.py` (|z| < 1 cm, < 0.5°) and no unexpected blocked sector ahead; self filter leaves no returns on
the robot; odometry chosen (Point-LIO bridge, or `--odom leg` for ≤ 3 m); the stream-replay test (H4) tracked in
the gantry; course = open floor, goal 2–3 m ahead, nothing overhead; gantry slack + spotter + remote (D-up / Select)
+ keyboard `O` all tested before the run.
