# Hardware bring-up plan (G1 29-DoF + head MID-360)

Nothing in this repo has run on a robot. lucid-g1-deploy's runner has been on a G1 once (INIT + fixed
stand in a gantry, 2026-09-11, policy never armed). Read its `docs/HARDWARE_RUNS.md`,
`DEPLOY_SEQUENCE.md` and `ETHERNET_AND_SDK.md` first; this plan builds on that procedure.

## 1. What runs where

```
G1 motion board (192.168.123.161)  <-- rt/lowcmd / rt/lowstate (DDS, domain 0) -->  deploy PC (wired, 192.168.123.x)
MID-360 (192.168.123.120)  --> Livox driver --> FAST-LIO2 --> ros2_lio_bridge.py  --(ZMQ odom :5561, cloud)-->
                                                                                  node.py --(ZMQ pose :5556)--> runner
```

Default: everything on an external PC with an NVIDIA GPU on the robot switch (SONIC's own docs run the
deploy this way; Wi-Fi is not acceptable). The Orin NX is possible later (runner needs JetPack 6.2 +
TensorRT 10.7; planner latency on the Orin is unmeasured and likely 2–4× the desktop).

## 2. Interfaces the hardware side must provide

| input | sim (today) | robot (to build/verify) |
|---|---|---|
| joints + pelvis IMU | rt/lowstate from robot_sim | rt/lowstate (same IDL) |
| pelvis pose, floor z = 0 | rt/odostate ground truth | FAST-LIO2 → `adapters/lio.py` frames (tested) → `PoseFilter` 50 Hz (tested) → ZMQ `odom` via `adapters/ros2_lio_bridge.py` (**untested**) |
| lidar cloud, world frame | ZMQ `cloud` from robot_sim | `/cloud_registered` → bridge → ZMQ `cloud` (**untested**) |
| SONIC | C++ runner, `--input-type zmq` | same binary, `--iface <robot NIC>`, CRC check ON (never `--disable-crc-check` on hardware) |

## 3. Things to measure on the robot before any walking

1. **MID-360 extrinsic.** URDFs disagree (inverted, pitch 0.0511 vs 0.040 vs 3.101 rad, z 0.416–0.428 m on
   torso_link; Unitree's page says −2.3° and 0.476 m). With the robot standing still, fit the floor plane in
   the first clouds: it must come out at z = 0 ± 1 cm and level within 0.5°. Adjust `mid360.POS_TORSO/RPY`
   (or `LioFrames(t_torso_lidar=...)`) until it does. Then put a box of known size 2 m ahead and check its
   height and position in `CloudMemory`.
2. **Self filter.** The G1 MJCF used for the filter must match the robot's hands (rubber hands vs Dex3).
   Count cloud points within 0.3 m of the pelvis axis after filtering while the arms move: must be ~0.
3. **LIO.** Gravity initialisation standing still; check drift by walking the robot by hand (gantry) 5 m out
   and back: < 5 cm, < 1°. FAST-LIO's world is the upside-down sensor start frame — `LioFrames.initialise`
   re-levels it with the pelvis IMU and FK; verify the levelled floor (1.).
4. **Clocks.** Lidar timestamps vs host: use host receive time for everything the node fuses (the node does).
5. **Heading.** The runner re-captures heading against the pelvis IMU at stream start; IMU yaw drifts against
   LIO yaw. Log both for 60 s standing; if the drift is > 2°/min, feed `heading_increment` corrections.

## 4. Ladder (stop at the first failure; every step with gantry/spotter and the remote's e-stop in hand)

| step | what | pass |
|---|---|---|
| H0 | desk: bundle verify, `pytest`, G3/G4/G5 in sim on the deploy PC itself | same numbers as on the dev box; planner p95 latency measured on this GPU |
| H1 | network read-only: `tools/check_robot.py --iface <nic>` (lowstate rate, mode_machine, IMU, joint order sanity) | 500 Hz lowstate, sane joints |
| H2 | perception standing still: driver + FAST-LIO + bridge; `tools/check_robot.py --perception` | §3.1–3.3 pass |
| H3 | lucid procedure in the gantry: runner INIT → stand → `]` SONIC tracking the stand motion | stable stand, no runner aborts |
| H4 | stream path without the planner: `tools/stream_replay.py` replays a recorded rehearsal plan sequence (empty floor, slow) into the runner, gantry slack | tracked like in sim; no catch-up resets after the first |
| H5 | planner live, empty floor, gantry slack then free with spotter, `--max-speed 0.8` and a short goal | arrives, settles to stand |
| H6 | ground obstacles (pillar, wall detour), measured course | as G4 in sim |
| H7 | overhead beams — only after the tilt envelope of deep ducks is checked in place (13/99 student episodes fell at 0.80–0.85 rad tilt in Isaac panel v2) | — |

## 5. Known gaps (decide before H5)

- **Walking speed.** The student walks the panels at ~1.2–1.6 m/s (goal lookahead 1.5 m). Slower operation
  (shorter lookahead) is untested — evaluate in sim (G4) before using it on the robot.
- **Planner latency.** Lock-step assumed 0; with 0.32 s + predictive anchoring + two planners in flight the
  sim cost is measured in G4. A faster planner (TensorRT/torch.compile, fewer ODE steps) is the main lever.
- **Start state.** Isaac panels start mid-walk; the robot starts standing (G4 `--start stand`).
- **Falls.** Supervisor: tilt > 0.75 rad → `estop` event → runner `O` (damping: the robot drops, so only
  with a gantry/spotter). There is no in-process recovery.
- **Moving people/obstacles** are outside the student's training distribution; the memory never forgets.
