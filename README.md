# g1-student-deploy

Deploy the best `~/flow` **lidar student** (flow-matching whole-body planner, `S2-guide10m8-grow06`,
83/98 held-out in Isaac) on a Unitree G1 29-DoF, with the released **GR00T SONIC** tracker running in
the **lucid-g1-deploy C++ runner**, and test the whole chain in MuJoCo first.

```
MID-360 cloud ─► self filter ─► label-free height/ceiling memory ─┐
G1 joints + IMU (DDS rt/lowstate) + odometry ─► 50 Hz history ────┤
                                                                   ▼
          student planner (K=16 guided FM, fused/CUDA-graph, ~0.17 s p95, predictive anchoring, 1 in flight)
                                                                   │ ZMQ protocol v1 (port 5556)
                                                                   ▼
             SONIC C++ runner (TensorRT, fused release ONNX 1570→29) ─► rt/lowcmd ─► G1
```

Status: simulation only. Nothing has run on a robot. See [docs/DESIGN.md](docs/DESIGN.md),
[docs/RUNBOOK.md](docs/RUNBOOK.md) (commands + results), [docs/HARDWARE.md](docs/HARDWARE.md) (bring-up ladder).

## What is here

| path | what |
|---|---|
| `tools/freeze_bundle.py` → `bundle/` | byte-identical planner source, student checkpoint, SONIC ONNX, mounts, contract; sha256 manifest checked against the evaluated artifacts |
| `g1deploy/sonic.py` | SONIC 1570-observation tracker in Python (ONNX parity) |
| `g1deploy/route_compat.py` | deployment correction for same-cell start/goal routes, applied by `bundle.load_flow()` without changing frozen files |
| `g1deploy/planner_compat.py` | preserves maximum candidate clearance, prefers actual stop-conditioned candidates on fallback ties, and reports the selected pool |
| `sim/lockstep.py`, `tools/eval_panel.py` | Path A: MuJoCo counterpart of the Isaac callback; switches for latency (+ predictive anchoring), standing start, lidar route, ray-cast lidar, odometry drift |
| `g1deploy/perception/` | MID-360 pattern/mount, MuJoCo ray casting incl. self-hits, mesh-capsule self filter, label-free evidence memory |
| `g1deploy/node.py` | the deployable node: DDS state + odometry + cloud → planner processes → SONIC stream; supervisor |
| `g1deploy/fastplan.py` | the planner as deployed: same plans as the evaluated one, 2.3–2.9× faster (CUDA graphs, per-chunk model constants, compiled steps, vectorised selection); [docs/PLANNER_SPEED.md](docs/PLANNER_SPEED.md) |
| `g1deploy/zmq_stream.py` | SONIC ZMQ protocol v1; framing proven equal to the Isaac feed |
| `sim/realtime/` | Path B: DDS MuJoCo G1 (lucid gear_sonic_sim) + obstacles + cloud; `rehearse.py` runs sim + C++ runner + node |
| `g1deploy/adapters/` | LIO frame math through the waist + 50 Hz pose filter (tested), ROS 2 FAST-LIO bridge (untested) |
| `tools/check_robot.py`, `tools/stream_replay.py` | hardware ladder H1–H2 (read-only checks) and H4 (stream without the planner) |

## Quick start

Near-goal routing preserves both physical endpoints when A* returns a single
grid cell. This prevents an empty-segment projection in both `Route` and
`SensorRoute`; coincident endpoints remain a valid zero-length segment. The
frozen bundle/manifest, normal routes, occupancy rules, and goal distance/speed/
hold scoring are unchanged. CPU regression check (no DDS or hardware):
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_route.py -q`.

Fallback selection also resolves discrete clearance ties explicitly. Previously
`stop_max_clearance` could select the first moving GO candidate when GO and STOP
pools shared the same clearance. The deployment adapter prefers the STOP pool on
an equal-clearance tie, then the candidate with least mean squared translation.
Higher-clearance GO candidates remain eligible and are labelled
`fallback_go_max_clearance`. `no_safe_candidate` reports when every candidate
misses the existing margin; neither fallback label certifies safety. Model
weights, sampler, scorer and frozen manifest are unchanged. The optional
execution verifier retains its own selection. CPU source-selector checks:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_route.py tests/test_planner_selection.py -q`.

The shared node/rehearsal local-goal function also redirects an obstructed
pursuit ray to the furthest visible point on the existing local route. The old
1.5 m extrapolation could cross a bend despite a valid route; just capping that
bearing at the wall could then leave a tiny goal and stall. The visible point
stays within the existing route arc lookahead; if none exists, the free ray
prefix is used. Occupied starts retain the existing escape behavior and are
marked uncertified. This is a route-grid check using its existing unknown-as-free
convention, not whole-body safety or observed free-space certification. Plan
diagnostics include the actual first requested/returned local goal and adjustment
counts (`tests/test_goal_guard.py`).

Sensor routing now defaults to a **0.10 m uncertainty buffer** around observed
obstacles, grown only into adjacent unknown support cells using the existing
`SensorRoute.grow_unknown_m` mechanism. Partial wall surfaces otherwise suggested
short gaps that later scans closed, forcing long detours. Observed free support
cells block this growth; the existing robot-radius inflation remains separate.
This buffer is a planning assumption, not measured occupancy: CloudMemory,
LiDAR returns and observed/unknown masks are unchanged. `student_config()` returns
the effective value for both the node and rehearsal, and native run configuration
records it. An explicit `grow_unknown_m: 0` in the bundle configuration retains
the previous behavior. The frozen bundle itself is never rewritten.

```bash
./g1 test                                  # 26 tests (bundle, SONIC parity, stream, perception, node safety, fast planner)
./g1 sim --layout beams2-100               # operator console + MuJoCo G1 + C++ SONIC runner + student node
                                           # keys: s start SONIC, g GO, h stand, t takeover, n goal, o E-STOP, q quit
./g1 robot --iface eth0 --goal 3 0 --odom leg --mount-calib mount.json      # the same chain on the G1
./g1 check | calibrate | bench | replay    # robot checks, lidar mount calibration, planner latency, stream replay
```

Guides: [docs/OPERATOR_GUIDE.md](docs/OPERATOR_GUIDE.md) (setup, keys, procedure, logs) ·
[docs/HARDWARE.md](docs/HARDWARE.md) (bring-up ladder) · [docs/G1_PERCEPTION.md](docs/G1_PERCEPTION.md) (the robot's
lidar, odometry, calibration) · [docs/READINESS.md](docs/READINESS.md) (what is established, what is not) ·
[docs/RUNBOOK.md](docs/RUNBOOK.md) (simulation gates and results).
Video: `runs/video/g1_deploy_tutorial.mp4` (built by `tools/make_tutorial.py` from recorded `g1 sim` sessions).


For position-only navigation (`bearing_near`), arrival heading latches the existing
route/guard approach direction at the observed robot pose once it enters the navigation position tolerance:
the task's `goal_radius_m` in lockstep, or the node's `--goal-radius`. All future
candidates then target this fixed world heading while retaining their requested XY
point. This avoids chasing a final segment that flips around a nearly reached
point, without suppressing approach turns before arrival or allowing a sampled
future to activate arrival. The source approach direction is retained rather than
replaced by measured yaw that may still lag a requested turn. Diagnostics record
both the requested heading basis and the measured entry yaw. The latch survives fresh worker callbacks, releases
outside the existing half-lookahead region (or radius, if larger), and resets
for new goals and independent runs. Node jobs retain the observed pose separately
from latency-predicted history. Explicit `path` mode keeps its path orientation,
including after route-goal redirection. Goal radius, speed and continuous-hold
scoring remain unchanged; heading latching does not declare goal completion.

Focused adapter tests (no model inference):
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_route.py tests/test_planner_selection.py tests/test_goal_guard.py tests/test_route_config.py tests/test_arrival_latch.py -q`.

Positive uncertainty buffers that round to zero support-grid cells are rejected
at `SensorRoute` construction: SciPy otherwise interprets zero dilation
iterations as unbounded growth. For example, 0.01 m with a 0.05 m support grid,
or 0.10 m with a 0.25 m grid, is unsupported; choose an explicit buffer that
rounds to at least one cell, or explicit zero to disable growth. The default
0.10 m buffer with the deployed 0.05 m grid remains two iterations.
