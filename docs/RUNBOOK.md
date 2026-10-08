# Runbook

All commands from the repo root. `source env.sh` in every shell (CycloneDDS library, lucid SDK,
runner path). The planner uses the GPU (~0.4 GB per planner); everything else runs on CPU.

## 0. Setup (once per machine)

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
CYCLONEDDS_HOME=$HOME/opt/sonic-deploy-toolchain/cyclonedds uv pip install --python .venv/bin/python -r requirements.txt
python3 tools/freeze_bundle.py            # on the dev box: copies + hashes the evaluated artifacts into bundle/
.venv/bin/python tools/make_stand_motion.py  # the runner's pre-loaded stand motion
```

The C++ runner comes from `~/lucid-sonic/lucid-g1-deploy` (`bash setup.sh && bash build.sh` there;
TensorRT 10.13 on x86, 10.7 on Orin). A deploy machine needs `bundle/` copied over: it is
self-contained and `g1deploy.bundle.verify()` checks every hash against `bundle/MANIFEST.json`.

## 1. Tests (seconds)

```bash
.venv/bin/python -m pytest -q
```

Covers: bundle/contract joint orders, SONIC ONNX parity vectors, observation layout, stream wire
format, **stream rows == Isaac feed rows**, MID-360 mount/pattern = flow sensor model, cloud memory on a
wall + beam, LIO frame math through the waist, 50 Hz pose filter.

## 2. Simulation gates (Path A, lock-step, CPU + GPU planner)

```bash
.venv/bin/python tools/replay_isaac.py $(ls -d ~/flow/artifacts/closed-loop-test-v4/student-guide10m8-grow06__*) \
    --out runs/g2_replay_test_v4.json                                         # G2, ~1 min
.venv/bin/python tools/eval_panel.py --name g3-base --workers 3               # G3, ~5 min
jobs/g4.sh                                                                     # G4 matrix, ~45 min
.venv/bin/python tools/compare.py runs/g3-base                                # vs Isaac (paired)
.venv/bin/python tools/compare.py runs/g3-base runs/g4-deploy                 # two MuJoCo conditions
```

One episode with any deployment effect:

```bash
.venv/bin/python sim/lockstep.py --task ~/flow/artifacts/panels/test-v1/beams2-101/task.json --seed 8871 \
    --out runs/one --lidar mjray --start stand --route sensor --prescan-s 0.3 \
    --latency-s 0.32 --anchor predict --inflight 2
```

## 3. Real-time rehearsal (Path B: sim + C++ runner + node, same processes as the robot)

```bash
source env.sh
$PYTHON sim/realtime/rehearse.py --task ~/flow/artifacts/panels/test-v1/beam-100/task.json --out runs/rt/beam-100
```

What it does, in the operator's order: start `robot_sim.py` (G1 on DDS, task boxes, MID-360 cloud on
:5560, pelvis held = gantry) → start the runner (`--input-type zmq`, fused release ONNX, TensorRT) →
"Init Done" → `]` (SONIC takes the stand) → release the gantry → node up → Enter (runner switches to
ZMQ streaming) → node pre-scans 0.5 s, plans from standing, streams → `O` on an estop event.
Outputs in the run dir: `score.json` (Isaac rules), `sim.npz` (100 Hz root pose + contact force),
`node.log` (JSON events), `node_plans.json` (per-plan latency / lateness), `runner.log` (time-stamped).

Ports: 5556 pose stream (node binds, runner connects), 5557 runner debug out, 5560 sim cloud,
5561 LIO bridge (hardware). Only one rehearsal at a time per machine (fixed ports, DDS domain 0).

## 4. Results so far (2026-10-08)

| gate | result |
|---|---|
| G0 bundle | 137 files, 97 MB; student `ad9c5b32…` and SONIC ONNX `325031d7…` match the evaluated artifacts |
| G1 SONIC parity | Python 1570-obs + ONNX vs 16 export parity vectors: < 1e-4 |
| G2 tracker gap | 98 recorded Isaac episodes replayed open-loop in MuJoCo: pelvis z RMSE 1.3 cm median (p90 1.7), tilt within ±0.03 rad, no MuJoCo falls; joint tracking 0.052 vs 0.033 rad in Isaac |
| G3 closed loop (Isaac protocol) | MuJoCo 81/98 vs Isaac 83/98 on the same tasks + seeds (discordant 7:5, McNemar p = 0.77) |
| G4 pre-scan 0.3 s | 83/98 (= Isaac) |
| G4 others | running (`runs/g4.log`): 0.32 s latency measured / predictive / pipelined, mjray lidar, standing start, lidar route, odometry drift, combined |
| G4a label-free lidar | MuJoCo ray casting incl. self-hits → mesh-capsule self filter (removes ~100% of self hits) → evidence memory: beam ceilings 156/156 and 307/307 cells exact, pillars/slalom/detour within a few cells of the labelled memory |
| G5 real-time | end to end works: empty-100 goal 3.4–4.1 s, no runner stalls during the run; plan latency 0.42–0.74 s p50 on the shared, saturated GPU (≈ 2× the lock-step number) |
