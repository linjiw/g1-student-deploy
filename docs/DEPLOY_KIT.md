# Deployment kit: rehearse, deploy, record

One command, `./g1`, runs the same chain in simulation and on the robot, records everything the robot, runner and
node exchange, and ends every session with a report. This page is the robot-day runbook; the operator keys are in
[OPERATOR_GUIDE.md](OPERATOR_GUIDE.md), the ladder's rationale in [HARDWARE.md](HARDWARE.md).

```
./g1 doctor --robot lab-g1         pre-flight: desk + network + DDS rates + controller mode + lidar + calibration
./g1 lidar start --robot lab-g1    robots without Unitree's lidar service: Livox driver + adapter (for check/calibrate)
./g1 lio replay SESSION            offline: FAST-LIO on a recorded session's lidar + IMU, drift PASS/FAIL (§7)
./g1 lio start|check|stop --robot lab-g1    FAST-LIO + LIO bridge for --odom zmq (Studio goals), standing check
./g1 check / calibrate / remote    H1 state, H2 perception + mount, remote buttons (read-only)
./g1 robot --robot lab-g1 --goal 2 0 [--replay EPISODE]     H3-H5 (H4 with --replay), recorder + runner CSV on
./g1 sim --layout NAME [--mission course] [--odom leg|lio]  the same chain in MuJoCo (courses: ./g1 courses)
./g1 sessions | report SESSION | note latest "tape: 1.94 0.05"
```

## 1. What runs, and what is recorded

| process | sim | robot | log / data in the session directory |
|---|---|---|---|
| recorder (`g1deploy/recorder.py`), subscribe-only | ✓ | ✓ | `rec/`: lowstate (all 35 motors, IMU, remote), lowcmd, secondary IMU, raw lidar frames, lidar IMU, `pose` / `g1_debug` / `odom` streams, node status at 5 Hz |
| SONIC runner | ✓ | ✓ | `runner.log`; `runner_csv/` per-tick CSVs (robot default; sim `--runner-csv`) |
| student node | ✓ | ✓ | `node.log` (events), `node_plans.json`, `node_memory.npz` (map at exit) |
| MuJoCo G1 | ✓ | | `sim.npz` (truth: pose, contact) |
| Livox driver + adapter (`--lidar livox`) | | ✓ | `livox.log`, `lidar_adapter.log` |
| LIO stand-in (`--odom lio`) | ✓ | | `lio.log` |
| console | ✓ | ✓ | `session.json` (code, bundle, runner, calibration, robot, operator, note), `console_frames.jsonl`, `console_summary.json` |
| report (at exit) | ✓ | ✓ | `report.md`, `report.json`: walks, planner latency vs the 0.32 s gate, stream rates and losses, tilt, motor temperatures and torques, PD tracking, flags |

All times are `time.monotonic()` of the deploy PC, so the recorder, the node events and the runner log line up.
Read a session in Python:

```python
from g1deploy import recording as R
rec = R.open_session("runs/console/robot-20261024-101500-lab-g1")
low = rec.stream("lowstate")                 # t, q (N, 35), dq, tau, temp, imu_quat, remote, ...
for t, pts, t_s in rec.lidar_frames(): ...   # raw MID-360 frames (sensor frame, per-point time)
status = rec.node_status()                   # phase, goal, odometry xy, tilt, latency, ...
```

A robot session records about 3 MB/s (lidar ~1.5 MB/s compressed; `--no-record-lidar` skips it). Copy session
directories off the machine after each robot day (lucid's HARDWARE_RUNS.md lesson: the first hardware run left no data).
Write what the robot cannot measure with `./g1 note latest "..."`: tape-measured stop positions, who was spotting,
floor, lighting, anything unusual.

## 2. Robot profile

`robots/<name>.json` holds what is specific to one robot (port, lidar path, hands, odometry, mount calibration) and
the facts measured on it ([robots/README.md](../robots/README.md)). `--robot NAME` applies it to `doctor`, `lidar`
and `robot`; command-line flags win.

## 3. Robot day

Stop at the first failure. Gantry or spotter, the wireless remote and the physical E-stop in hand every time. Two
people: the remote operator watches the robot, the console operator watches the screen.

**H0 desk (the day before or that morning).** `./g1 doctor` all ✓; `./g1 test`; one `./g1 sim --layout open-3m`
and one `./g1 sim --layout nav-room --mission course` (also builds the TensorRT engines and planner kernels).

**Network and lidar.** Cable into the robot's switch, `nmcli con up g1-robot` (192.168.123.222/24,
never-default). `./g1 doctor --robot NAME`: link, pings, lowstate ≥ 400 Hz, lidar ~10 Hz. On robots without
Unitree's lidar service, `./g1 lidar start --robot NAME` first (the console starts it by itself later).

**H1 read-only state.** `./g1 check --iface IFACE --seconds 5` (no `--perception`: that mode under-counts the
rate). Joints inside limits, tilt small.

**H2 perception, standing still.** `./g1 check --iface IFACE --perception --cloud dds --odom leg --png h2.png` and
`./g1 calibrate --iface IFACE --out robots/NAME/mount.json --png coverage.png` with the robot standing still on open
floor. Floor |z| < 1 cm and < 0.5° after calibration; no blocked sector in the front ±35°. Confirm the forward
direction once per robot: an object in front must appear ahead (+x) in `h2.png`.

**LIO (only for `--odom zmq` / Studio's real-robot goals).** Robot standing still: `./g1 lio start --robot NAME`
(FAST-LIO + bridge; it starts `g1 lidar` if needed), `./g1 lio check --robot NAME` (30 s: < 3 cm, < 0.5°). Then walk
the robot 5 m out and back by hand with `g1 lio status` running (< 5 cm, < 1°) before trusting it for a walk (§7).

**Remote.** `./g1 remote --iface IFACE`: press Start, Select, F1, D-up; all four must register. Only with the robot
in the gantry, since Unitree's own controller may react to buttons.

**H3 runner in the gantry.** `./g1 robot --robot NAME --goal 2 0`, type `yes`. The runner releases Unitree's
controller and ramps to the stand ("Init Done"). `s`: SONIC stands the robot. Watch the record line (lowstate
~1000 Hz on lab-g1, lowcmd ~500 Hz from the runner). Re-run `g1 calibrate` now, standing on SONIC, and keep that
file. End: `t`, tighten the gantry, `o`, `q`.

**H4 stream without the planner.** `./g1 robot --robot NAME --goal 2 0 --replay EPISODE` (a recorded empty-floor
episode, e.g. `runs/lockstep/open-floor-s0` from `sim/lockstep.py --start stand`), `s`, slacken, `g` streams it at
half speed. Pass: tracked, one catch-up reset.

**Site check before every walk (new start pose or goal).** `./g1 site --robot NAME --goal 2 0 --png site.png
--json site.json` with the robot standing at the start: the node's own memory and the frozen planner's own boxes and
route. NO-GO (start or goal inside a box, no route) means move the start; CHECK (merged giant box, ceiling near
start/goal, mostly unseen route) means look at `site.png` first. Then rehearse that exact walk on its twin:
`.venv/bin/python tools/site_to_course.py site.json --name here-2m && ./g1 sim --layout here-2m --odom leg`.

**H5 student, open floor.** `./g1 robot --robot NAME --goal 2 0`, `s`, slacken, `g`. With leg odometry the robot
can walk past the goal (simulation: up to ~25 %); keep that much clear floor beyond it. Nothing within 0.8 m of the
start or goal between 0.8 and 1.35 m high (table tops, the gantry frame): the node files those as ceilings and
defers its stand there. New goal: `n` then `g`.

**After each session** the console prints the report; read the flags first. `./g1 note latest "..."`.

**Captures for research (record them when the step comes up; the recorder runs in every console session).**
- H3, SONIC standing the robot in the gantry: 20–30 s standing still with lidar. Real self-returns at SONIC's stand
  pose: leak and mask counts offline with `tools/guard_mask_real.py SESSION` (so far only the hanging-limp pose was
  measured).
- With the gantry strap where it was on the NO-GO day: `g1 site ... --json site.json`, keep the session (memory
  dump). Confirms on real data that the strap stays visible (the hanging-limp recordings say yes, §10).
- Optional: a person walks past at 1–2 m while the robot stands (does the memory forget what moved away?).
- LIO: the 5 m out-and-back hand walk with `g1 lio start` running (§7), recorded: the first moving real-data LIO
  check (`g1 lio replay` has only had standing data).

**Never run a simulation on the deploy PC during a robot session**: both use ZMQ :5556/:5557/:5561/:5562 on this
machine, and a sim's streams end up in the robot's recording (seen on 2026-10-23). `g1 doctor` flags busy ports.

## 4. Navigation goals

| how | where | needs |
|---|---|---|
| `--goal AHEAD LEFT` at launch | sim, robot | — |
| `n` + `g` on the console (relative to the robot now) | sim, robot | the robot standing |
| `--mission "A,L;A,L"` / `--mission course` (scripted legs) | sim | — |
| Studio Navigate, simulated robot (click a goal on its lidar map) | Studio | — |
| Studio Navigate, real robot (Studio checks and stages the goal, operator presses `g`) | robot | `--odom zmq` (LIO); with leg odometry Studio refuses goals (frame mismatch) |

`jobs/rehearse_studio_link.sh` rehearses the real-robot Navigate workflow against the simulated chain: `g1 sim
--odom lio` (LIO stand-in on :5561), Studio's gateway and robot link, a goal clicked through the API, read back from
the node, the console's `g`, and a check that the robot arrived where Studio sent it.

## 5. Simulation rehearsals

`./g1 courses` lists the course library (`sim/courses`): `open-2m`, `open-3m` (H5), `pillar`, `box-low`,
`wall-detour`, `slalom` (H6), `table-edge` (the 0.8–1.35 m ceiling trap), `nav-room` (multi-goal mission).
`--odom leg` rehearses first-day odometry, `--odom lio [--lio-noise JSON]` the LIO path. `jobs/g5.sh` and
`tools/eval_panel.py` (RUNBOOK.md) are the scored panels; they need the dev box's Isaac panels.

## 6. Lidar on G1s without Unitree's lidar service

Some firmware has no `lidar_driver` service (lab-g1: `robot_state` lists 20 services, none for the lidar), so nothing
publishes `rt/utlidar/*`. Unitree's G1 lidar instructions and GalaxyGeneral's
[Click-and-Traverse](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse) (`deploy/Click-and-Traverse-SLAM`)
read the MID-360 on the host instead: Livox-SDK2 / `livox_ros_driver2` with `host_net_info` = the host's IP
(192.168.123.222), `/livox/lidar` (CustomMsg) + `/livox/imu`, then FAST-LIO2 (`mid360.yaml`: Livox type 1, blind
0.5 m, extrinsic_T [-0.011, -0.02329, 0.04412]) → ground filter → OctoMap. Unitree's page adds a noise rule: keep a
point only when the low four bits of its Livox `tag` are 0.

This kit keeps the node's own, evaluated perception and only changes where the raw frames come from:

```
MID-360 ─UDP─► livox_ros_driver2 (this PC) ─/livox/lidar─► g1deploy/adapters/livox_to_utlidar.py
                 frames cut every 0.1 s on lidar time, tag filter, G1 raw layout (x y z intensity ring time)
             ─► rt/utlidar/cloud_livox_mid360 + rt/utlidar/imu_livox_mid360 ─► node.py --cloud dds (unchanged)
                 [--custom-out /livox/lidar_10hz: the same frames as 10 Hz CustomMsg for FAST-LIO]
```

`./g1 robot --robot NAME` with `"lidar": "livox"` starts the driver (unless one runs) and the adapter (unless one
runs: two adapters would publish every frame twice). `./g1 lidar start|status|stop` does the same outside a console
session, for `g1 check --perception` and `g1 calibrate`. The driver build on this PC publishes one message per UDP
packet (~2 kHz × 96 points) whatever `publish_freq` says; the adapter does not care, FAST-LIO does (hence
`--custom-out`).

## 7. LIO odometry (needed for Studio's real-robot goals)

Leg odometry is the first-day default; Studio's real-robot Navigate needs the node on LIO (`--odom zmq`). On this PC
the ROS python cannot import the deploy stack, so `g1deploy/adapters/dds_lio_bridge.py` reads FAST-LIO's
`/Odometry` (+ `/cloud_registered`) directly from DDS with the same frame math and 50 Hz filter as
`ros2_lio_bridge.py`. `g1 lio` runs the pieces:

```bash
./g1 lio replay runs/robot-probe/rec2 [--loops 6]   # offline, any time: FAST-LIO fed a recorded session (below)
./g1 lio start --robot NAME     # robot standing still: g1 lidar (if needed) + FAST-LIO + bridge -> :5561
./g1 lio check --robot NAME     # 30 s standing: FAST-LIO's /Odometry inside 3 cm / 0.5 deg, else FAIL
./g1 robot --robot NAME --odom zmq --goal 2 0          # the console warns if no bridge runs
./g1 lio status | stop          # logs in runs/lio/
```

FAST-LIO is the Click-and-Traverse `lidar_ws` build with `config/lio/fastlio_mid360_g1.yaml` (input: the adapter's
10 Hz `/livox/lidar_10hz` + `/livox/imu`). Like every FAST-LIO run, it rewrites its own debug logs under that
workspace's `src/FAST_LIO/Log/` (`mat_pre.txt` etc.; its ROOT_DIR is compiled in).

**Offline check on real data (`g1 lio replay`, `tools/lio_replay.py`).** Plays a recorded session's `rec/lidar`
frames (as CustomMsg, the adapter's `--custom-out` format) and `rec/lidar_imu` into FAST-LIO on a private DDS domain
(77) on loopback, nothing on the robot NIC; `--loops N` repeats a short recording with shifted stamps (a standing
robot stays put). A fresh FAST-LIO per replay: reusing one across recordings sends its IMU propagation across the
gap between them and it diverges (kilometres). Results on lab-g1's two standing recordings (2026-10-23, robot in the
gantry, Unitree controller holding; RTX 4070 Ti SUPER PC):

| recording | frames, IMU | replay | poses | drift max (end) | yaw drift | largest step | verdict |
|---|---|---|---|---|---|---|---|
| `runs/robot-probe/rec2` | 80 frames, 1655 IMU (8.3 s) | ×6, 49.5 s | 477 at 9.6 Hz | 0.25 cm (0.15) | 0.10° | 0.17 cm | PASS |
| `runs/robot-probe/rec3` | 92 frames, 2020 IMU (10.2 s, 0.4 s lidar gaps) | ×6, 60.0 s | 549 at 9.1 Hz | 0.47 cm (0.27) | 0.05° | 0.27 cm | PASS |

So the config, the adapter's 10 Hz frames, the Livox IMU in g and the lidar-clock stamps work with FAST-LIO on this
robot's data. Not yet covered: motion (extrinsic and IMU-gyro errors only show when the head moves) and the bridge's
world initialisation on real lowstate. Next robot day: `g1 lio start` + `g1 lio check` standing, then the 5 m hand
walk (HARDWARE.md §3: < 5 cm, < 1°), recorded, before using it for a walk.

## 8. lab-g1: what the read-only bring-up measured (2026-10-23)

| item | measured |
|---|---|
| network | `eno1` ↔ robot switch, NetworkManager profile `g1-robot` (192.168.123.222/24, never-default); PC1 .161 0.1 ms, Orin .164 0.2 ms, MID-360 .120 |
| `rt/lowstate` | **1000 Hz** (tick +1 per sample; docs assume 500), 0 missing, 5 % duplicate deliveries |
| controller | mode `ai`; Unitree's `humanoid` service writes `rt/lowcmd` at 1 kHz (zero gains when limp, kp 40–200 when holding) until the runner releases it |
| DDS | 18 participants, all Unitree services on PC1 (incl. `g1_dex_protocol_v2.0`: Dex3 hands, and `g1_arm_example`); nothing on the Orin |
| lidar | no Unitree lidar service; Livox driver on this PC (Click-and-Traverse workspace) → adapter: 10.0 Hz, ~14 k points/frame, 1.4 % tag-dropped; lidar clock within a few ms of this PC |
| H2 (nominal mount) | floor +4.0 cm, 1.41° → calibrated 0.03 cm, 0.06° (Δroll −1.1°, Δpitch −1.3°, Δz −4.3 cm); forward = +x confirmed (wall behind the robot) |
| occlusion | blind bands at azimuth −45…−35° and +40…+50° (head shell sides), thin lines at ±90°, ±157°; front −35…+40° clear |
| ceiling cells | gantry frame 0.4–0.5 m behind the pelvis at 1.13–1.17 m (stand deferred at the start pose); furniture tops 0.83–1.03 m |
| remote | connected (header present); buttons not yet tested |

## 9. What the rehearsals found, and what was fixed (2026-10-23)

Debugging a walk: `./g1 sim|robot ... --debug-plans` makes the node save every plan's exact inputs and result;
`tools/debug_plan.py SESSION --plans 0 1 2` re-runs the planner on them and draws every candidate (paths, pelvis
heights, safety, the live plan, the map the network sees). `./g1 sim --viewer` shows the robot live;
`tools/render_session.py` renders a session to video.

| finding | evidence | status |
|---|---|---|
| a first plan dropped by the sanity check left the node in RUN forever (robot standing, console waiting) | kit-nav-room session | **fixed** `g1deploy/node.py` (re-request with no active plan) + `tests/test_node_supervisor.py` |
| `--odom zmq` (LIO) could never start with `--goal-frame start`: the odometry input was created after the anchoring wait | Studio-link rehearsal, node log "no robot state to anchor the goal" | **fixed** `g1deploy/node.py`; Studio's real-robot workflow then passed: goal read back within 0.2 cm, arrived 0.20 m from it |
| more than 32 guidance boxes left the fast planner's CUDA graphs: a torch recompile per new box count, then the recompile limit, an **11 s plan** mid-walk | lab twin (36 boxes); the real lab has 29 boxes after 5 s | **fixed** `g1deploy/fastplan.py` buckets (8…128), recompile limit 64, padded fallback; eager vs fused on the twin: same rule in 49/49 plans, same final distance; p95 0.28 s fused vs 1.16 s eager |
| node control and pose sockets listened on every interface (Wi-Fi included) | `tcp://*:5562` | **fixed** the console binds them to 127.0.0.1 |
| Flow's `memory_boxes` merges connected walls/furniture into one box; around the robot it contains the start and every plan is a crouch (dropped) | nav-room: one 5.4 × 6.1 m box, 452 plans dropped; lab twin: a 2.9 × 7.1 m box | frozen planner behaviour: **`g1 site` detects it**; choose open start poses |
| in the twin of the lab site the student crouched, lunged, backed off and sometimes fell on a clear 2 m path | `runs/debug/twin-try*`, `tools/debug_plan.py` | **root cause found and fixed**: the self filter leaked `right_shoulder_pitch_link` returns (sphere in the filter model, cylinder in the simulator, 2 cm range noise vs a 3 cm margin) into two cells beside the robot at 1.06 m; the first plan then had 0 of 16 safe forward candidates and fell back to max-clearance plans. Margin 3 → 5 cm (`cloud_memory.py`): 0 leaks in 170 noisy frames, 3/3 walks arrived upright with 0 contact. Real robot: returns near the pelvis 605 → 405 |
| the recorder lost the lidar (1 frame) in walled rooms; this PC dropped 5.26 M UDP packets for "receive buffer errors" (0.3-0.9 MB lidar bursts, 208 KB socket cap) | rec/stats.json, `netstat -su` | code requests 16 MB buffers (`g1deploy/dds.py`); **needs `sudo sysctl` once** (`g1 doctor` shows the command) |
| leg odometry off by up to 0.77 m on a 2 m walk (~1 m on 3 m) | h5-open2m-leg, rehearse leg-odom-pillar | known (READINESS: −9/98); keep 25 % clear floor beyond the goal; move to LIO |
| planner p95 0.26–0.29 s near obstacles on this RTX 4070 Ti SUPER (0.06 s on open floor) | pillar, wall-detour, slalom, table-edge | within the 0.32 s gate, close to it; `--planner-precision tf32` is the evaluated faster option (82/98, p95 0.142 s) |
| obstacles 0.8–1.35 m tall and the gantry frame are filed as ceilings → deferred stand | table-edge trap, lab site | `g1 site` reports them |
| `g1 check` exited 245 after its report (leg-odometry thread vs DDS teardown) | H2 runs | **fixed** clean exit |

## 10. Deploy-PC measurements (2026-10-24, RTX 4070 Ti SUPER)

**Where a no-safe plan's time goes** (`tools/profile_fallback.py`, fused fp32, exact node inputs, CUDA-synced
stages, uncontended GPU; `runs/profile/`). With no safe candidate the frozen planner samples K = 16 go, 3K = 48
resampled and K = 16 stop candidates:

| state | guide boxes | total | go (16) | resample (48) | stop (16) | clearance (80) | grounding | rest |
|---|---|---|---|---|---|---|---|---|
| lab-site twins, 4 states (median) | 6–9 | 242 ms | 41 | 104 | 40 | 52 | 2 | 2 |
| dense room (`rt-4070/kit/dense__s2` memory) | 31 | 336 ms | 58 | 154 | 50 | 53 | 20 | 2 |
| as deployed, lab twins (K go only) | 6–9 | 54 ms | 41 | – | – | 10 | 1 | 1 |

In real time (runner + simulator on the same GPU and CPU) the dense room's no-safe plans took 0.42 s: about 1.25×
the uncontended number. The 3K resample is ~46 % of a no-safe plan, its share of the clearance another ~10 %:
skipping it would bring the dense case to ~0.15 s uncontended. That is a candidate-set change (a planner or
`planner_compat` decision with its own panel), not a speed-up of the same plan. Clearance is CPU-bound and linear in
the candidates (~0.65 ms each).

**Lock-step panel, remaining courses** (`tools/robust_panel.py`, 3 seeds; `runs/robust/ls-4070-b`): kit 9/12 (3
collisions: pillar s3, slalom s1, s2), kit-walls-slow 11/12 (1 collision: slalom s3), 0 falls. Guard view 9.0 ms p50,
9.6 p95, 12.3 max per plan on this CPU; guard selection p95 0.5 ms (max 19 ms on an envelope reselect).

**Self mask on real lidar** (`tools/guard_mask_real.py`: lab-g1 rec2 and rec3 re-fused through the node's path, leg
odometry, the preliminary mount calibration; robot hanging in the gantry, controller holding). At the deployed 5 cm
margin: 1 near-only cell inside the 10 cm band, 0.20 m behind the pelvis at 1.17 m, where the harness strap leaves
the back. It is joined to evidence outside the band, so the mask hides nothing (0 masked, 0 refused) and the strap
stays visible to the planner. At 3 cm: 2 such cells (0.15 and 0.20 m behind), still 0 masked. SONIC's stand pose
(H3) is not measured yet (captures, §3).

## 11. The safety rope and spotter behind the robot (lab-g1, 2026-10-24)

**What failed.** Both door attempts planned nothing safe from the first plan: the rope from the back hook ran back to the
spotter at 1.0-1.3 m, the MID-360 saw it as a line of cells from 0.19 m behind the pelvis, and the height field made it a
column touching the robot (it follows the robot, so the memory never clears it). Re-planning the exact inputs with the
rear half blanked: 13-16/16 safe candidates, chosen pelvis 0.41-0.50 m (it ducks).

**Next robot day.** Rope vertical from overhead (>= 1.8 m, above the back hook; the inverted MID-360 barely sees above head
height), spotter >= 1 m behind and to the side; Arm -> Start SONIC -> rope up, spotter back -> **Clear map** (panel `c`)
-> check the map around the robot -> GO. Beams and doors: the rope has to be passed over the beam.

**Rear blind (`--rear-blind-deg 180`, off by default).** Real-time deploy-kit simulation, deploy guards, lookahead 1.0,
one run each (runs/sim-studio, videos in runs/sim-studio/videos):

| run | full lidar | rear 180 deg blind |
|---|---|---|
| door twin, goal 2.5 m | arrived 25 s, around the left post | arrived 5 s, around the left post |
| door, 2.6 m / 0.5 m left | arrived 43 s, around the left post | arrived 34 s, under the beam, **49 N brush**, 8 no-safe |
| door, 3.0 m / 0.5 m right | arrived 5 s, under the beam | arrived 9 s, under the beam |
| stop before the door 0.9 m | 1.8 s | 5.4 s |
| start spot 1.5 / 2.0 / 2.0 + 0.5 left | 1.6 / 2.0 / 2.1 s, no guard events | 2.0 / 2.7 / 2.3 s, 2-12 envelope corrections each |
| slalom 3.4 m | 5.6 s, 1 correction | 10.2 s, 12 reselects + 2 drops, no contact |
| pillar, wall detour | 3.7 s, 4.2 s | 4.2 s, 4.1 s |

13/13 arrived either way; the blind rear half makes the frozen planner's first choices crouchier and slower (more envelope
corrections), presumably because an unseen rear half is outside what it was trained on. In lock-step (no real-time
chain) any rear sector broke the slalom (0-1/3). Use it for straight tests with a tether, not for weaving. Next software
step to try: blank only the *near* rear (< ~1 m: rope, spotter), keeping the far rear observed.

**Near-rear blind (`--rear-blind-deg 180 --rear-blind-range-m 1.0`, 2026-10-25).** Only rear returns closer than 1 m
(horizontal, from the torso) are dropped: the rope and the spotter, while the far rear stays observed. Same real-time
runs (one each unless noted; `corr` = guard envelope corrections; `under`/`around` = how the robot passed the door):

| scene | full lidar | rear 180° blind | near rear < 1 m blind |
|---|---|---|---|
| **rope behind** (static 2 cm rope 0.2-0.8 m behind, 1.0-1.3 m high) | **never walks**: 273 plans, all no-safe | 2 s, 8 corr | **2 s, clean** |
| **door + rope** | **never walks**: 265 no-safe | 4 s, under | **5 s, under, clean** |
| door (authored, 1.0 m, beam 1.17 m) | 5 s, under | 5 s, under | 5 s, under |
| lab door twin 2.5 m | 25 s, around | 5 s, around | 8 s, under |
| lab door twin 2.6 m / 0.5 L | 43 s, around | 34 s, under, 49 N | 5 s, around |
| lab door twin 3.0 m / 0.5 R (3 runs) | 3/3 arrived, 0 N (2 under) | 9 s, under | 3/3 arrived, 2 clean under, **1 touch 191 N** |
| stop before the door 0.9 m | 2 s | 5 s | 4 s |
| start spot 1.5 / 2.0 / 2.0 + 0.5 L | 2 / 2 / 2 s | 2 / 3 / 2 s, 3-12 corr | 2 / 2 / 2 s, 0-5 corr |
| slalom | 6 s, 1 corr | 10 s, 14 corr | **5 s, clean** |
| pillar / wall detour | 4 / 4 s | 4 / 4 s | 4 / 4 s |

The static rope reproduces the robot's failure exactly (full lidar: no motion at all); the near-rear blind removes it
and behaves like full lidar elsewhere (slalom included), unlike the full rear blind. One 0.1 s 191 N touch in three
runs of the irregular scanned door (crouched at the end of a partly seen beam) against none with full lidar: too few
runs to call, so validate on the robot with the 2 m open walk before any door. On the robot (rope attached):
`--rear-blind-deg 180 --rear-blind-range-m 1.0` (robot profile keys `rear_blind_deg`, `rear_blind_range_m`).
**Default for lab-g1 since 2026-10-25** (`robots/lab-g1.json`); `--rear-blind-deg 0` turns it off for a session.

## 12. Other bundles, deploy rules plugins, offline replays (2026-10-26)

**Another bundle.** `./g1 sim|robot --bundle DIR` runs the node on that bundle (`G1_STUDENT_BUNDLE`; SONIC files must
match the deployed ones). The session records the bundle path and its MANIFEST hash. A bundle whose student config is
not the deployed format (no `route_variants`) also needs its **deploy rules plugin**: `--rules FILE.py`
(`G1_DEPLOY_RULES`). The node never runs such a bundle without one. The plugin interface is in
`g1deploy/bundle.load_rules`: it supplies the route, extra memory layers for the planner's view, a planner class built
on the fast planner, and the planner job itself. The node's guards are passed in, and its state is carried from one
replan to the next. Plugins for unpublished bundles are kept outside this public repository (`private/` is
git-ignored).

**Fast planner, per-replan margins.** `guide_margin_m` is now an input of the captured CUDA graphs. A plugin may
change it per replan; before this, a graph kept the margin it was captured with (`tests/test_fastplan_margin.py`).
Graph buckets now go up to 512 guidance boxes: finer box decompositions of a real room give 40–320 boxes within
2 m. A pool whose candidates × padded boxes exceed `MAX_PAIRS` runs off the graphs in one pass. Splitting it into
graph-sized candidate batches was tried and reverted: it changes the denoiser's GEMM shapes, and the normalised
guidance amplifies the last-bit differences up to 4e-3 in the raw state on 48 candidates. Unsplit, 48 candidates ×
512 boxes need more than 13 GB of GPU memory. A bundle that needs that must bound its guidance memory another way, in
its plugin's planner class. Guidance cost is linear in the box count: about 0.5 s per 16-candidate pool at about 300
boxes on the 4070 Ti SUPER, against about 0.04 s at 16.

**Offline replays** (`tools/replay_plans.py`). These re-plan a recorded session's states with the node's own job
function (`node.plan_job`), for any bundle or plugin. Inputs are the `--debug-plans` dumps (exact history and goal)
and the node's lidar memory, rebuilt from the recorded raw frames:
- on the node's grid;
- each frame deskewed and registered with the pose of its own time;
- the node's pelvis position from its status stream and the measured plan histories;
- IMU orientation and joints from rt/lowstate;
- the session's mount calibration.

Each row reports how well the rebuild matches the node's own memory at that plan. On lab-g1 h5-1: occupied-cell IoU
0.98 / 0.96 standing, 0.87 / 0.74 late in the walk, where 5 Hz status poses replace the node's 200 Hz leg odometry.
Kcsc's offline rules then match the live ones. (Registering every frame with the pose of the replay instant instead
gave IoU 0.32: frames from the first seconds land tilted.) `tools/compare_replays.py` tabulates runs (no-safe,
fallbacks, advance, pelvis, latency, rebuild IoU, plugin flags); `--rear-blind-deg / --rear-blind-range-m` re-fuse
with a lidar blind.

```
G1_STUDENT_BUNDLE=$PWD/bundle-x G1_DEPLOY_RULES=path/to/rules.py .venv/bin/python tools/replay_plans.py \
    runs/robot-day-20261024/h5-1 --every 3 --out runs/replays/h5-1_x.json
.venv/bin/python tools/compare_replays.py runs/replays/*.json
```

**Door risk: torso pitch near the tilt E-stop (sim, 2026-10-26).**
- In the deploy-kit door scenes (lab door twin, authored door, door + rope, all rear-blind settings), 24 kcsc door
  passes arrived. Each run's peak real pelvis tilt was 0.49 rad median, 0.63 p90 and 0.73 at most: most of the 0.75 rad
  node E-stop (`--tilt-estop`).
- One run in 25 crossed it: lab door twin, goal 3.0 / −0.5. For 15 s at the door, the chosen plans stayed "safe" while
  their own planned tilt climbed from 0.49 to 0.81 rad (knees up to 2.48 rad).
- The guards' walk envelope (tilt ≤ 0.6) is waived within 2 m of a low ceiling, so the student may duck. It cannot
  simply stay on: successful passes stream plans planned up to 0.8–0.99 rad.
- The E-stop then damps the joints, and the robot falls.
- **Contacts:** across all 19 kcsc runs on the lab door twin (3 goals; full, rear and near-rear lidar), 6 touched the
  frame or the furniture beside it (49–289 N); one more is the E-stop above.
- **Not latency:** kcsc plans in about 0.1 s here. With the node logging each plan switch (`switch`, `offsets` in
  node_plans.json), its arms run 0.3–0.7 rad (median) from the reference, before and after a switch alike.
- **On the robot at a door:** expect forward-pitched ducks close to the limit, and side brushes. The gantry or rope must
  be ready to catch a damped collapse there. Stop the pass, with the remote's Select or the panel, if the robot rocks
  back and forth under the beam for more than a few seconds.
