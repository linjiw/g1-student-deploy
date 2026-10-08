# Operator guide — running the student on a G1 (simulation first, then the robot)

Everything goes through one command, `./g1`. It launches the **same deployment chain in simulation and on the
robot**: the SONIC C++ runner (TensorRT, `gamepad_manager` input: keyboard + Unitree wireless remote + the student's
ZMQ stream) and the student node (lidar → memory → planner → stream). In simulation a MuJoCo G1 stands in for the
robot on the same DDS topics, including the MID-360 point cloud in the robot's own format.

```
              ./g1 sim | ./g1 robot
                     │  (operator console: status + keys)
     ┌───────────────┼───────────────────────────────────────────┐
  [sim only]      SONIC runner (C++)                     student node (Python)
  MuJoCo G1  <──> rt/lowcmd · rt/lowstate (DDS) <────── rt/lowstate, rt/utlidar/cloud_livox_mid360
  + obstacles                 ^                                   │
  + MID-360                   └──── ZMQ pose stream :5556 ◄───────┘    control :5562 ◄── console
```

## 1. Install (deploy PC: Ubuntu 22.04, x86_64, NVIDIA GPU)

```bash
# the SONIC runner (once): lucid-g1-deploy
git clone https://github.com/linjiw/lucid-g1-deploy ~/lucid-sonic/lucid-g1-deploy
cd ~/lucid-sonic/lucid-g1-deploy && bash setup.sh && source env.sh && bash build.sh

# this repo + its Python env
cd ~/code/g1-student-deploy
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
CYCLONEDDS_HOME=$HOME/opt/sonic-deploy-toolchain/cyclonedds uv pip install --python .venv/bin/python -r requirements.txt
# the frozen policy bundle: copied from the dev box (bundle/, ~0.9 GB) or rebuilt there with tools/freeze_bundle.py
./g1 test            # 22 tests: bundle hashes, SONIC parity, stream format, perception, node safety
./g1 bench           # planner latency on this machine: p95 must be <= 0.32 s
```

The first start of each SONIC model builds a TensorRT engine (policy ~30 s, SONIC's planner ~7 min); later starts
load the cached engine.

## 2. Simulation: drive it with the keyboard

```bash
./g1 sim --layout beams2-101          # any layout in ~/flow/artifacts/panels/test-v1: empty-100, beam-101, pillars-101, ...
./g1 sim --task my_course.json --goal 4 0
```

| key | does | runner / node side |
|---|---|---|
| `s` | start SONIC: the runner takes control, the robot stands on SONIC's reference (sim: the gantry releases 1.5 s later) | runner `#` `]` |
| `g` | GO: hand control to the student; it pre-scans the lidar 0.5 s, plans from the stand and walks to the goal | runner `#` Enter (stream on), node `go` |
| `h` | student stand: the stream settles to a stand (waits until clear of a ceiling) | node `stand` |
| `t` | takeover: SONIC's own planner, standing (= remote D-up) | runner `@`, node `stand` |
| `n` | new goal "ahead left" (m, relative to the robot now), then `g` | node `goal` |
| `o` / Esc | E-STOP: damping — the robot drops | runner `o` |
| `r` | (sim) release the virtual gantry now | — |
| `q` | quit (on the robot: asks for confirmation) | — |

The screen shows: runner state (booting / INIT done / CONTROL / E-STOP), who drives the robot (SONIC reference,
student stream, SONIC planner), the student's phase (WAIT, PRESCAN, RUN, ARRIVED, STOPPED), goal distance, plan
count and latency, lidar age, pelvis height and tilt, and the event log. Every session is saved under
`runs/console/` (runner/node/sim logs, plans, the node's lidar memory).

**Scripted (tests, CI, videos):**

```bash
./g1 sim --layout beams2-101 --record runs/console/demo \
  --script "wait:init,sleep:1,key:s,wait:control,sleep:2.5,key:g,wait:run,wait:arrived,sleep:2,key:t,sleep:4"
MUJOCO_GL=egl .venv/bin/python tools/render_session.py runs/console/demo --out demo.mp4   # console + robot video
```

Script steps: `wait:init|control|stream|run|arrived|standing`, `sleep:S`, `key:K`, `goal:AHEAD:LEFT`.

**Batch rehearsals** (scored with the Isaac rules): `jobs/g5.sh <name> --lidar-io dds` runs 14 held-out layouts
through the full chain; `tools/eval_panel.py` runs the 98-episode lock-step panel (see RUNBOOK.md).

## 3. Robot setup (once per robot / site)

1. **Network.** Deploy PC on the G1's switch, wired, static `192.168.123.222/24` (any free .x). DDS domain 0. Never Wi-Fi.
2. **Unitree services.** In the Unitree app make sure the lidar service (`lidar_driver`) is on. Do not run your own
   Livox driver (it steals the lidar stream from `rt/utlidar/*`).
3. **Time.** Only for LIO: chrony on the PC/Orin against PC1 (`server 192.168.123.161 iburst prefer`).
4. **Hands.** `--body dex3` (default) or `--body rubber`: the self filter must match the robot.
5. **Calibrate the head lidar** (robot standing still on open floor, in its stand or gantry):
   `./g1 calibrate --iface eth0 --out mount.json --png coverage.png` → floor level within 0.5°, |z| < 1 cm; look at
   `coverage.png` for blocked sectors ahead.
6. **Odometry.** First walks: `--odom leg` (joint encoders + IMU, no extra software; fine for ≤ 3 m). For longer
   courses run Point-LIO on the robot's lidar topics + `g1deploy/adapters/ros2_lio_bridge.py` and use `--odom zmq`
   (configuration in G1_PERCEPTION.md §4).

## 4. Robot procedure (every session)

Pre-flight, robot in damping, hanging in the gantry, feet on the floor:
```bash
./g1 check --iface eth0 --perception        # 500 Hz lowstate, 10 Hz lidar, no returns on the robot, floor at 0
./g1 replay runs/<recorded empty-floor session>/ --slow 2     # H4 only: the stream path without the planner
```

Run:
```bash
./g1 robot --iface eth0 --goal 3 0 --odom leg --mount-calib mount.json
```

1. The console asks for `yes` (robot in gantry, remote in hand). The runner releases Unitree's controller and
   ramps the joints to the stand ("Init Done").
2. `s` (or remote **Start**) — SONIC stands the robot. Slacken the gantry.
3. `g` (or remote **F1** + `g` for the node) — the student walks to the goal and settles to a stand.
4. Override at any moment: `t` / remote **D-up** → SONIC's planner, standing (not within ~1 m of anything
   overhead); `o` / remote **Select** → damping (the robot drops into the gantry).
5. New target: `n`, type "2 0", then `g`.
6. To end: `t` (SONIC stands), tighten the gantry, `o` (damping), `q`.

Rules: no overhead obstacles until the ladder's H7; nobody within 2 m of the robot's path; the remote operator
watches the robot, the console operator watches the screen; any surprise → Select.

## 5. Reading the logs

| message | meaning |
|---|---|
| `node: plan_dropped (pelvis_z …)` | a plan went below the 0.35 m pelvis floor and was rejected; the previous plan keeps playing |
| `node: stand_deferred` | a stand was asked for near a remembered ceiling; the current plan finishes first |
| `node: stand (cloud stale …)` | no lidar frame for 1 s: the student stops (check the lidar service / network / CPU load) |
| `node: estop (tilt)` | pelvis tilt above 0.75 rad: the console sends damping |
| runner `Catch-up: Reset to frame 0` | normal once per stream start; repeated = the stream fell behind (planner too slow) |
| runner `waiting following motion` | the stream ran out (end of a stand, or the node stopped streaming) |
| sim `selected interface "lo" is not multicast-capable` | normal for the loopback simulation |
