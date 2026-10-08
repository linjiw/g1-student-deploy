# Hardware bring-up plan (G1 29-DoF + head MID-360)

Nothing in this repo has run on a robot. lucid-g1-deploy's runner has been on a G1 once (INIT + fixed
stand in a gantry, 2026-09-11, policy never armed). Read its `docs/HARDWARE_RUNS.md`,
`DEPLOY_SEQUENCE.md` and `ETHERNET_AND_SDK.md` first; this plan builds on that procedure.

## 1. What runs where

```
G1 PC1 (192.168.123.161, closed): motion board + Unitree lidar_driver
   rt/lowstate (500 Hz)  rt/utlidar/cloud_livox_mid360 (10 Hz, raw)  rt/utlidar/imu_livox_mid360 (200 Hz)
        │ DDS domain 0, wired
deploy PC (192.168.123.x, NVIDIA GPU) ── or the Orin NX later (JetPack 6.2 + TensorRT 10.7; latency unmeasured)
   SONIC C++ runner  --input-type gamepad_manager --planner-file bundle/sonic/planner/V2/planner_sonic.onnx
                     (CRC check ON, never --disable-crc-check on hardware)   <── ZMQ pose :5556 ──  g1deploy/node.py
   node.py  --cloud dds --mount-calib mount.json --odom zmq|leg --goal-frame start --goal <ahead> <left>
   [Point-LIO on /utlidar/* + adapters/ros2_lio_bridge.py → ZMQ odom :5561]   (Orin or PC, ROS 2 + rmw_cyclonedds)
```

Details of the robot's sensors and the LIO configuration: [G1_PERCEPTION.md](G1_PERCEPTION.md).

## 2. Operator procedure (rehearsed in sim: `rehearse.py --operator remote`)

1. Robot in the gantry, feet on the floor. Start the runner; wait for "Init Done" (INIT ramp to the stand).
2. **Remote Start**: SONIC takes control with its own planner (IDLE, standing). Slacken the gantry.
3. Start the node (it pre-scans the lidar while standing, then plans from the stand).
4. **Remote F1**: control passes to the node's stream; the student walks to the goal and settles to a stand.
5. Override at any time:
   - **D-up** → SONIC's own planner, standing (from 1.38 m/s walking: standing still in ~1.5 s). **Not near an
     overhead obstacle**: standing up within ~1 m of a beam drove the head into it (566 N) in rehearsal.
   - **Select** → e-stop (damping: the robot drops; only with the gantry/spotter).
   - The node's own events: tilt > 0.75 rad → `estop`; lidar or state stale → stand (deferred near ceilings).

## 3. Measure on the robot before any walking

| check | tool | pass |
|---|---|---|
| planner latency on the deploy machine | `tools/bench_planner.py` (fused planner) | p95 ≤ 0.32 s (the latency tested with predictive anchoring); measured 0.16–0.18 s on the loaded deploy PC |
| DDS state and lidar after releasing Unitree's controller | `tools/check_robot.py --iface <nic> --perception --cloud dds` | 500 Hz lowstate, 10 Hz lidar, no returns left on the robot |
| MID-360 mount + occlusion | `tools/calibrate_mount.py --iface <nic> --out mount.json --png coverage.png` | floor |z| < 1 cm, < 0.5° after calibration; no blocked sector ahead |
| hands of the self filter | `G1_BODY_MODEL=dex3` if Dex3 hands | no returns within 0.3 m of the pelvis while the arms move |
| odometry | Point-LIO + bridge (`--odom zmq`), or `--odom leg` for ≤ 3 m tests | walk the robot by hand 5 m out and back: < 5 cm, < 1° |
| clocks | chrony on the PC/Orin against PC1 | only needed for LIO (the node times points on arrival) |

## 4. Ladder (stop at the first failure; gantry/spotter and the remote in hand every time)

| step | what | pass |
|---|---|---|
| H0 | desk: `freeze_bundle` copy verifies, `pytest`, `bench_planner.py`, one rehearsal on the deploy PC | numbers as in RUNBOOK |
| H1 | read-only network check (`check_robot.py`) | §3 rows 2 |
| H2 | perception standing still: calibrate, coverage, self filter, odometry | §3 |
| H3 | runner in the gantry: INIT → Start → SONIC planner stand | stable stand, no aborts |
| H4 | stream without the planner: `tools/stream_replay.py <recorded empty-floor episode> --slow 2`, gantry slack | tracked; one catch-up reset at the start only |
| H5 | student live, open floor, goal 2–3 m (`--goal-frame start`), gantry slack then spotter | arrives and stands |
| H6 | ground obstacles (pillar, wall detour), measured course | as G4 in sim |
| H7 | overhead beams — only after the deep-duck envelope (tilt up to ~0.7 rad) is checked in place, and with a beam that yields | — |

## 5. Known gaps (decide before H5)

- **Walking speed.** From standing the student averages ~0.9 m/s (peaks 1.4 m/s; goal lookahead 1.5 m). A slower mode
  (`--goal-lookahead 1.0`) is being evaluated (G4e).
- **Planner latency.** Lock-step assumed 0; with 0.32 s + predictive anchoring + two planners in flight the
  sim cost is measured in G4. A faster planner (TensorRT/torch.compile, fewer ODE steps) is the main lever.
- **Start state.** Isaac panels start mid-walk; the robot starts standing (G4 `--start stand`).
- **Falls.** Supervisor: tilt > 0.75 rad → `estop` event → runner `O` (damping: the robot drops, so only
  with a gantry/spotter). There is no in-process recovery.
- **Moving people/obstacles** are outside the student's training distribution; the memory never forgets.
