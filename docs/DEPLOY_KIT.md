# Deployment kit: rehearse, deploy, record

One command, `./g1`, runs the same chain in simulation and on the robot, records everything the robot, runner and
node exchange, and ends every session with a report. This page is the robot-day runbook; the operator keys are in
[OPERATOR_GUIDE.md](OPERATOR_GUIDE.md), the ladder's rationale in [HARDWARE.md](HARDWARE.md).

```
./g1 doctor --robot lab-g1         pre-flight: desk + network + DDS rates + controller mode + lidar + calibration
./g1 lidar start --robot lab-g1    robots without Unitree's lidar service: Livox driver + adapter (for check/calibrate)
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
`ros2_lio_bridge.py`:

```bash
g1 lidar start --robot NAME            # adapter; then (ROS env, rmw_cyclonedds on the robot NIC):
ros2 launch fast_lio mapping.launch.py config_path:=$PWD/config/lio config_file:=fastlio_mid360_g1.yaml rviz:=false
.venv/bin/python g1deploy/adapters/dds_lio_bridge.py --iface eno1 --forward-cloud    # robot standing still
./g1 robot --robot NAME --odom zmq --goal 2 0
```

Status: the bridge and the 10 Hz input are built and unit-tested; FAST-LIO on this robot has **not** been run yet
(the link dropped before the test). Validate it standing still (odometry must not drift), then by walking the robot
5 m out and back by hand (HARDWARE.md §3: < 5 cm, < 1°), before using it for a walk.

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
