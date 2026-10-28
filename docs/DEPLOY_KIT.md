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
| recorder (`g1deploy/recorder.py`), subscribe-only | ✓ | ✓ | `rec/`: lowstate (all 35 motors, IMU, remote), lowcmd, secondary IMU, raw lidar frames, lidar IMU, `pose` / `g1_debug` / `odom` streams, the LIO registered `cloud` (when a bridge runs), node status at 5 Hz |
| SONIC runner | ✓ | ✓ | `runner.log`; `runner_csv/` per-tick CSVs (robot default; sim `--runner-csv`) |
| student node | ✓ | ✓ | `node.log` (events), `node_plans.json`, `node_memory.npz` (map at exit) |
| MuJoCo G1 | ✓ | | `sim.npz` (truth: pose, contact) |
| Livox driver + adapter (`--lidar livox`) | | ✓ | `livox.log`, `lidar_adapter.log` |
| LIO stand-in (`--odom lio`) | ✓ | | `lio.log` |
| console | ✓ | ✓ | `session.json` (code, bundle, runner, calibration, robot, operator, start confirmations, note), `console_events.jsonl` (every act: source, operator, result, reason), `attempts.jsonl`, `console_frames.jsonl`, `console_summary.json` |
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

**Recording health and evidence (since 2026-10-28).** The recorder rewrites `rec/stats.json` every second: its pid
and heartbeat (`t_unix`), disk free, and per stream the samples, the rate over the last 5 s against the expected rate,
the last sample and the last chunk write, gaps (silences longer than max(3 expected periods, 0.1 s)), samples dropped
and why, the writer's queue, chunks, bytes, write errors. It never stops for a fault, it reports it: a chunk that
cannot be written (disk full, I/O error) is dropped and counted; chunks are written on their own thread with a bounded
queue, so a slow disk shows as a busy writer and then as dropped samples; below 1 GB free (`--min-free-gb`) or past
`--quota-gb` it keeps its heartbeat but drops (and counts) samples. `meta.json` gets the lidar fields at the first frame
(a cut recording stays decodable) and `rec/finalized.json` is written last, at a clean exit: without it the recording
was cut (kill, power loss) and everything up to the last whole chunk is still usable (`g1deploy/recording.py` skips a
cut chunk and lists it in `rec.skipped`). If its console disappears, the recorder notes it, records 60 s more (the
aftermath) and finalizes itself. The console's own files on a full disk lose lines, never the session (a child's
stdout is always drained, so a full disk cannot block the runner).
`g1deploy/evidence.py`: `recording_health(dir)` (live: each stream's required / active / stalled state, rates, last
write age, drops, gaps, bytes; disk free; errors; problems; finalized; recorder alive) and `manifest(dir)` v2 (each
record present / partial / missing / n/a, required or not, span, gaps, drops, reason). Required: lowstate, lowcmd,
lidar_raw, node_status, node_events, plans, console_events, attempts (+ the LIO odometry and cloud with zmq / fused
odometry, and lio in simulation; + the sim truth for a rehearsal; + node memory, start checks, scan, identity, the
robot checklist and the rehearsal set with a robot attempt where they apply); optional: video, runner log, report. A
record that does not apply to the session as it went (no Arm, no GO) is n/a and not required; whether the runner ran
is the console's own record (a done Arm or Start SONIC act, or a console started without `--arm`), never the optional
runner log. `complete` only when
every required record is present (not partial), the recording is finalized and the console closed the session; a
running session is `recording`. `python -m g1deploy.evidence DIR --health`.

A robot session records about 3 MB/s (lidar ~1.5 MB/s compressed; `--no-record-lidar` skips it, and makes the session
incomplete: raw lidar is required). Copy session directories off the machine after each robot day (lucid's HARDWARE_RUNS.md lesson: the first hardware run left no data).
Write what the robot cannot measure with `./g1 note latest "..."`: tape-measured stop positions, who was spotting,
floor, lighting, anything unusual.

## 2. Robot profile

`robots/<name>.json` holds what is specific to one robot (port, lidar path, hands, odometry, mount calibration) and
the facts measured on it ([robots/README.md](../robots/README.md)). `--robot NAME` applies it to `doctor`, `lidar`
and `robot`; command-line flags win.

## 3. Robot day

Stop at the first failure. Gantry or spotter, the wireless remote and the physical E-stop in hand every time. Two
people: the remote operator watches the robot, the console operator watches the screen.

**The lab hub and the panel (since 2026-10-27).** `./g1 lab` serves the day's home page (at `/legacy` since 2026-10-28) on
127.0.0.1:8769 (reached from the operator's tailnet devices like the panel). It lists the deploy profiles, starts a
sim rehearsal (`g1 sim --profile P --arm --panel 8771 --layout COURSE`) or, after a four-item checklist (gantry,
E-stops, rope and spotter, clear area), a robot session (`g1 robot --robot NAME --profile P --arm --panel 8770`),
and shows the day's sessions (`runs/lab-day-YYYYMMDD/`) with their attempts, report flags, report and a Deploy Studio
replay link (the day directory is linked into Studio's `robot-sessions`). The hub never sends anything to the
robot: the console it starts runs in its own process group (it outlives the hub), the runner still waits for the
panel's Arm, and a robot session with the runner up is ended from its panel only. The panel (`g1deploy/panel.html`)
keeps the sticky E-STOP and adds the robot day around it:
- *Next step*: the ladder (systems, Arm, stand ramp, Start SONIC, planner warm, Clear map, goal, GO, the walk, its tag)
  with the button for the step to do now (`g1deploy/workspace.py`). GO is refused, with the reasons on the button, by
  the console's GO gate (below).
- *What the planner sees*: the node's memory around the robot (`{"cmd": "map"}` on the node's control port, 10 cm
  cells, 0.7 s refresh), robot-centred and facing up, with the rear blind, the active plan and the goal; tap to pick a
  goal. Cells seen only from up close (the self-return signature the guards mask) have their own colour.
- *Checks*: PC quiet, recording, lowstate, lidar age, remote (robot), motor temperatures (robot: the remote watcher
  reports the hottest motor, ankles and knees every second; warn at 60 °C, NO-GO at 75 °C), obstacle cells within
  0.5 m outside the blind and tall cells 1.0–1.6 m behind (standing only), node loop lag, plan latency.
- *Attempts* (`attempts.jsonl`): every GO opens one; arrival, stop, E-STOP, takeover or a stand closes it with its
  extremes (tilt, pelvis height, latency, no-safe plans, distance walked). The operator tags it (clean, touched,
  stumble, caught, aborted, setup issue) and adds a note.
The terminal console and its keys work as before; the panel and the keys drive the same actions.

**Acts and the GO gate (since 2026-10-28).** Every operator action is an *act*, whoever sends it: the panel page, the hub
and Deploy Studio (through the hub) post `POST /act {act, id, source, operator, expect?, ...}` to the console's panel;
the terminal keys, the remote and `--script` steps are acts too. E-STOP runs at once on the request thread and answers
`{ok, key_sent, runner_alive, runner, confirmed, t_mono}` (`confirmed`: the runner's log shows its EMERGENCY STOP).
Every other act needs a client `id` (a repeated id returns its first result, so a retry never acts twice), is refused
when a field of `expect` (`session, runner, phase, goal_seq, attempts`) no longer matches, and is answered within 1 s:
`{ok, id, result: done | refused | queued, reason, session}`. Each one is a line in `console_events.jsonl` (source,
operator, arguments, result, reason; an act answered `queued` has a `queued` line there before its reply leaves);
`/state.acts` shows the last 20. `GET /act_result?id=` answers what became of any act of the last 500 (any source,
recorded the moment the console receives it): `{known, id, act, result: done | refused | queued | pending, reason,
t_mono, t_done, source, operator, session}`; a relay whose reply was lost asks it instead of sending again. Goals: `goal {ahead, left}` (the panel) or
`goal_odom {x, y, seq, check_id}` (the hub's Send goal, converted to the node's odometry frame after its start check)
count only once the node's status echoes their seq (within 0.5 s; otherwise "goal not confirmed by the node").
GO (`g`) is refused, with every reason, unless: the runner is in CONTROL and not in E-STOP, 2 s have passed since
CONTROL, no rehearsal runs beside the robot, the node is ready and not walking, LIO is not lost, a goal was read back in
this cycle (after the latest of CONTROL, the last attempt's end and the last Clear map; the node still has it) and the
10 s Clear-map scan is over. All of it is checked before the runner is switched to the student's stream. A robot
console's launch goal (`--goal`, anchored by the node at Arm) never counts: send the goal. A sim console counts its
launch goal for the first GO only (scripted rehearsals and `--mission` keep working); `--require-sent-goal` makes a sim
behave like the robot (practice). `/state.go_locks` lists the reasons (empty iff GO would be accepted);
`go_locks_panel` adds the operator lease: while Deploy Studio holds it (`lease {holder, until}` from the hub), GO from
the phone panel is refused (E-STOP, Stand and Takeover always work). Each attempt row keeps what was in force at GO:
goal seq and odometry position, the start check and its `check_id`, the rehearsal set, the odometry mode and health;
`session.json` keeps the operator and the start confirmations (`--confirm JSON`).

**The node's live view (since 2026-10-28).** `{"cmd": "view", "cloud_after": n, "mem_after": n, "max_points": 4000}`
on the node's control port (base + 6) answers at once, in strict JSON (no NaN): the state (`qpos` 36, `joint_names`,
`state_age_s`, the last 0.5 s of poses), the goal, the active plan's frames 0.4 s and 0.9 s from the current tick and its
path, the route, odometry health (`odom {mode, lio_age_s, doubt, lost, watchdog}`; also in the node's status), the
lidar mount, the last lidar frame (int16 cm relative to its origin, one class per point: 0 fused, 1 removed by the self
filter, 2 dropped by the rear blind; decimated per class) and the full 241 x 241 memory grid (uint8 support / ceiling /
near layers; cell (i, j) centred at lo + (i, j)·res; a new `seq` at most once per second and only on a change, `epoch`
+1 on Clear map). Cloud and memory are null unless newer than the cursors. The perception thread builds these parts
only while someone polls (the hub's 5 Hz poller), so the 50 Hz loop pays nothing otherwise. The recorder also keeps
the LIO bridge's registered cloud (`rec/zmq_cloud`) when it runs.

**Scan, goal, rehearse, deploy (the hub's main card).** One map, robot-centred at its start pose and facing up:
1. *Scan now* (and every robot session start, unless the scan its goal was picked on is under 15 min old) runs
   `tools/scan_scene.py`: `g1 site` (read-only, 8 s of the node's own lidar memory with the robot profile's mount and
   rear blind; `--dump-memory`), Deploy Studio's scan-to-twin fit of that memory (`studio.twin`, run with Studio's
   Python) saved as a Studio arena over Studio's local API, and a course of the twin in the start frame
   (`site_to_course.py --out`). The map shows the scan's heights and the twin's boxes.
2. *Goal*: tap the map (ahead / left of the start pose); the page flags a goal inside a twin box or on unseen floor.
3. *Rehearse this goal in the twin*: `g1 sim --task TWIN --goal A L --script ...` (the ladder runs by itself, the panel
   shows it live); the planned and walked pelvis paths and contacts are drawn on the same map.
4. *Start the robot session* with the same profile and goal, after the four-item checklist; the runner waits for Arm.
Rehearsals run beside a live robot session: the hub starts every sim as `nice -n 10 g1 sim ... --port-base 5656`, an
isolated chain (its own stream, runner debug, cloud, odometry and control ports; DDS on loopback while the robot's
chain is on its own interface). While it runs, `runs/.rehearsal.lock` holds its pid and the robot console refuses GO
(and the hub's panel proxy refuses `g`); it is not started while the robot walks, nor beside a robot console older
than this interlock. Two sims never run together (both on loopback DDS).
Every rehearsal is a pair: the exact one (simulator odometry), then, when it ends, the same goal on true odometry with
the step-tied error lab-g1's leg odometry showed (`--odom lio --lio-noise '{"motion": "lab-g1-leg"}'`,
g1deploy/odom_error.py). The results list marks the pair *same with drift*, *drift: other way (x m)* (the
drifted walk left the exact path by more than 0.5 m) or *drift: did not arrive*; a goal that only works with exact
odometry is flagged before GO (§13).
*Start check* (`g1deploy/registration.py`, since 2026-10-27): within a live session the hub registers the node's live
map to the scan's (obstacle cells, heading search ±30° plus an FFT over translation, 0.1 s on the CPU) and shows how far
the robot moved and turned since the scan (a match below 60 % is not trusted: rescan). *Send goal to the robot* then
sends the scan's goal as the same spot in the room, from the robot now; every check is appended to the session's
`start_check.jsonl`. On 2026-10-27 every walk started 16–31° right of its scan's heading and 0.16–0.26 m off (the
robot hangs in the gantry until the runner takes over), so an unconverted goal pointed up to 1 m beside the rehearsed
one. When the robot stands more than 0.10 m or 5° off, *Rehearse from where the robot stands* moves the twin into the
robot's frame (`registration.repose_task`, `runs/.../reposed/`); it needs the GPU room beside the live session.
Every robot session that ends also gets a twin of its final node map (`scan_scene.py --from-session`). Studio accepts
loopback requests only, so its links open on the deploy PC itself.

**The deploy service for Deploy Studio (API v1, since 2026-10-28).** `./g1 lab --headless` is the same hub without a
page to open: Deploy Studio's Deploy workspace is the operator surface, and its gateway calls the hub's JSON API on
127.0.0.1 (`/` says so; the page above stays at `/legacy`). The token is `G1_PANEL_TOKEN`, else
`~/.config/g1-deploy/panel_token` (0600, created once; the consoles read the same file, so a restarted hub still
reaches them). A restarted hub re-attaches to every console still running from any `lab-day-*` directory beside its
day, with the token in that session's `panel_url.txt`. A development hub: `./g1 lab --headless --port 8789 --day
SCRATCH/lab-day-x --no-studio-link` (never 8769). Every reply is strict JSON (non-finite numbers are `null`).
- `GET /api/v1/state` (computed at 1 Hz in the background, served from a cache): versions, PC, robot link with the
  lidar rate from the adapter's stats log and the LIO pose age, profiles, the live robot / sim session with its panel
  port, the path of its token file and the console's heartbeat age, scans (goal-independent reasons only), robot
  sessions uncapped (others paged: `/api/v1/sessions?offset=&limit=`), rehearsal sets, GPU, the last doctor run
  (kept in the day's `doctor.json`). `GET /api/v1/readiness?scan=&slot=`: every check with its evidence kind, when it
  was measured, its maximum age, the step after which it can be measured, and its 3D anchor.
- Scans: `POST /api/v1/scan`, `GET /api/v1/scan/live` (the scan so far, from `g1 site --live-out`: robot, the last
  frame with each point's fate (fused, self-filtered, rear-blind dropped) and the memory, 2 Hz),
  `GET /api/v1/scan/<name>/scene` (memory, twin, course and the frames between them).
- Practice, no robot: `POST /api/v1/practice/scan {course | scan}` runs `robot_sim` alone (unarmed, held standing) and
  scans it like the robot (`g1 site --iface lo --odom dds` with the robot profile's mount and rear blind), then stops
  it: a normal scan named `practice-*`. `POST /api/v1/start {mode: "sim", practice: true, scan, start_offset}` starts
  a practice session on that scan's course reposed by a start offset (default the 2026-10-27 one: 20° right, 0.2 m),
  with Arm and the panel, so the start check and the goal conversion run for real. Practice and robot sessions never
  run together; every sim runs alone.
- `POST /api/v1/rehearse {scan, goal_scene, profile}`: a set of three runs in sequence in the sim slot: exact, drift
  (`lab-g1-leg`) and start offset; verdict as before plus `offset_ok`. `GET /api/v1/rehearsal/<run>/track` after a
  run: sim truth in the scene (29 joints picked by name from the sim's 43-DoF model, 25 Hz) and contacts over 5 N.
- `POST /api/v1/start_check {scan, slot}` and `POST /api/v1/send_goal {slot, scan, goal_scene, check_id}`: the goal
  goes to the console as `goal_odom` only with a goal check's id, after a start check and a coarse check on the live
  node memory. `GET /api/v1/live/<slot>/view`: the node's `view` (one 5 Hz poller per live slot on the node's control
  port) with the frame to draw it in. `POST /api/v1/act` relays an operator act to the console (audited to the day's
  `hub_actions.jsonl`); `POST /api/v1/stop` ends a session only before Arm, after an E-STOP or standing after an
  attempt; a robot start refuses a stale scan instead of scanning again. `GET /api/v1/session/<name>/manifest`: the
  session's evidence (`g1deploy/evidence.py`, also written when a session ends).

*Contract v3 (after the 2026-10-28 review).* A page must describe the session that actually runs, a rehearsal counts
only for what it rehearsed, and a session reopens from its directory alone. `state.capabilities` names what a hub has
(`session_receipt`, `rehearsal_identity`, `rehearsal_match`, `passage`, `act_status`, `session_detail`, `session_tag`,
`profile_route`, `session_bound_acts`, `go_rehearsal_gate`, and `recording_health` / `manifest_v2` when the evidence
module provides them); an older hub answers 404 "unknown endpoint". `memory_sha` (scan rows, scan scene, receipts,
identities) is the SHA-256 of the scan's memory.npz:
- **Session receipt**: `live.<slot>.receipt` in the state and `GET /api/v1/session/<name>/receipt`: kind (robot,
  practice, rehearsal), robot, operator, start, the profile and the SHA-256 of its file, the bundle's MANIFEST hash,
  the rules plugin's hash, odometry (practice: the robot odometry it practises, `sim_odom` beside), the mount
  calibration's hash, the scan and its memory hash, the start offset, the confirmations, the kit / private / runner
  commits; from session.json (what the console ran), hub_start.json before the console has written it (`source`).
  Acts, `stop`, `start_check` and `send_goal` carrying `session` or `expect.session` for another session are refused
  (409) before they reach a console; every act reply names the session it went to.
- **Rehearsal sets** (`state.rehearsals`, newest first by start time): an `identity` (scan + its memory hash, goal,
  profile name + file hash, bundle and rules hashes, odometry, the three starts, the drift model), a `state` (running,
  partial: stopped or interrupted, complete, failed: a run that did not start or ended without sim truth) and per run:
  the node's result, `arrival` (the node ended ARRIVED **and** the simulator's truth ends within 0.5 m), the true final
  distance, `passage`, the largest contact, a fall, `timing_valid` (node loop lag ≤ 0.35 s and fewer than 3 overruns,
  from report.json) and `passed` (arrival, through the gate, no contact over 5 N, no fall). The verdict exists only for
  a complete set (robust, other way, fails with drift, missed passage: the exact run arrived around the gate, fails). `POST /api/v1/rehearse` takes `passage {a, b}`: a gate segment in the scene (e.g. between two
  towers' inner faces) the true path must cross; a detour that arrives is a failure (a walk that starts past the gate:
  its passage is `n/a`, "started past the gate"). `GET /api/v1/rehearsal/match?
  scan=&x=&y=&profile=&odom=` (or `&session=` / `&slot=` for a session's own receipt; `&offset=dx,dy,dyaw`,
  `&pa=x,y&pb=x,y`): the newest complete set whose identity matches (goal within 0.08 m), the newest running one,
  and every other set with its mismatched or missing fields. A set from before identities never matches (its scan's
  memory hash is unknown).
- **GO in robot mode** (the operators' decision): the hub relays a robot session's GO only with a complete set matching
  the goal sent through `send_goal`, the session's profile contents, bundle, rules and odometry **whose verdict is
  robust**, or with the operator's `rehearsal_override {operator, reason}`: any other verdict (fails, fails with drift,
  missed passage, other way) needs the override whatever the odometry (FAST-LIO only changes the words). Either way
  the rehearsal in force (and the override) goes to the console with GO and is stored with the attempt; readiness
  blocks the same way. Practice: advisory, recorded the same way, never refused.
- **Start check and the rehearsed start offsets**: `POST /api/v1/start_check {scan, slot, goal_scene?}` (and
  send_goal's own check) judges `within_offset_envelope` against the set in force for that goal (the goal given, else
  the goal last sent in this session on this scan): only when that set's offset case passed; `null` without a goal,
  without such a set, or when its offset case failed.
- `GET /api/v1/act/<slot>/<id>`: what became of an act whose reply was lost: the console's act record, else its last 20
  acts, else its event log, beside the hub's own audit of the relay.
- `GET /api/v1/session/<name>` (any session endpoint: the name is looked up in the hub's day, then in the other
  `lab-day-*` directories beside it, newest first, so a Review link from an earlier day opens; an unknown name is a
  404 saying where it looked): the receipt, ended or not, the attempts (labels with their history, check, start check,
  rehearsal in force, goal in the scene, scan, passage gate and the result through it: on the simulator's truth in
  practice, on odometry mapped by the start check on the robot), Studio's Replay id, the manifest in brief and the
  recording health. `POST /api/v1/session/<name>/tag {n, tag, note, operator}`: a live session's console takes the
  label; an ended session's `attempt_tags.jsonl` gets an append-only row, merged over the console's rows (the newest
  wins) wherever the hub reads attempts.
- `GET /api/v1/profile/<name>/route`: the route parameters the node uses with that profile (route margin, step-over,
  unknown growth, the node's ceiling deferral, lookahead, guards; the profile's rules overrides on a rules bundle),
  numbers and names only, with the profile file's hash, so a goal check can use the walk's own configuration.
- `GET /api/v1/session/<name>/recording`: `evidence.recording_health` (expected vs active streams, rates, gaps, disk,
  finalized), `{supported: false}` with a kit that has none.

*Frames* (`g1deploy/frames.py`). Poses are [x, y, yaw]; "A_from_B" is B's origin in A. A scan's memory grid (and its
twins) are in *memory_S*, where the robot stood at `start_S` (leg odometry carries the IMU yaw: 1.2–1.4 rad on several
10-27 scans). The *scene* is the scan robot frame (`start_S` at the origin, x ahead; the course is in it):
`scene_from_memory = inverse(start_S)`. A start check returns `theta, t` (robot now from scene) taken with the node at
`P_c` in its odometry, so `scene_from_odom = inverse(P_c ∘ [t, theta])`. Goals cross the API as `goal_scene`; the hub
sends `inverse(scene_from_odom)` of it. A course reposed by a start offset O starts the robot at O in the scene:
`scene_from_course = O`. On scan-145338 (start yaw 1.381 rad) the twin maps onto the course within 1.3 mm and the
scan's goal lands on its typed 2.5 / −0.3 m.

**The wireless remote: two buttons only.** Select = E-STOP (the console's watcher turns it into the runner's damping in
every input mode) and D-up = takeover (the runner's standing motion set). The SONIC runner reads every D-pad press
itself, also while the student streams, and switches to its own planner in that motion set: D-down squat/crawl,
**D-left boxing** (arms up in a guard; then L1/R1/L2/R2 are punches and hooks), D-right styled walking. Stock
Unitree combos (L2 + ...) mean something else here. On 2026-10-27 an L2 then D-left during a walk put lab-g1 in the
boxing set mid-turn, and a later L2 threw a left hook. The console logs every press, and a D-down / left / right
during a walk stands the student and raises an alarm; the panel names the runner's motion set.

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

**Deploy profiles.** `--profile NAME` (`profiles/<name>.json`, or `private/profiles/` for bundles kept outside this
repository, or a path) sets the student bundle, rules plugin, guards, lookahead, lidar blind and planner options in
one go: `./g1 sim --profile deploy --layout lab-door-aligned`, `./g1 robot --robot lab-g1 --profile deploy ...`. The
command line wins over the profile and the profile over the robot profile. `deploy` is the robot's current setup
(deploy guards, lookahead 1.0, near-rear blind, plan dumps); `deploy-full-lidar` is the same without the blind. Every
`session.json` records the profile and a deploy identity (one hash over the effective settings, the bundle MANIFEST
and the rules file), and the report prints both (`profiles/README.md`).

**A quiet PC.** Real-time numbers depend on the PC being otherwise idle: replays run beside a batch on 2026-10-26
doubled the node's loop lag and caused an E-stop 2.9 s into a walk (§11). Before every session the console takes a
snapshot of the load, the GPU, other deploy chains and heavy kit jobs (`g1deploy/machine.py`) and stores it in
`session.json` ("machine"). `--busy warn` (default) goes on and logs it, `wait` waits up to `--busy-wait-s` for the
PC to settle, `refuse` stops, `off` skips the check; another deploy chain already running is refused unless `off`.
The report flags a session as timing-suspect when the PC was busy at its start or the node's loop lag passed
0.35 s (clean runs ~0.2 s): rerun it before trusting its latency, contact or fall numbers. For batches use
`--busy wait`; add your own heavy jobs with `G1_HEAVY_JOBS=regex,regex`.

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

With `--odom zmq` the node also runs leg odometry alongside as a watchdog (`OdomWatchdog`, node event `odom_watchdog`
at start): the robot stands (`odom_disagree`) when LIO says it moves faster than 1.2 m/s, or the two speeds differ by
more than 0.4 m/s, for 0.5 s (leg odometry under-counts 5–40 %; speeds only, the frames need no alignment).
`--no-odom-watchdog` turns it off.

**`--odom fused` (FAST-LIO + legs, since 2026-10-28).**
- What runs: the node runs `g1deploy/odom_fusion.py`'s filter at 200 Hz.
  - Leg-odometry steps and the IMU's yaw predict, with extra noise while turning (leg odometry breaks in turns).
  - The bridge's poses correct (one per 0.1 s). The filter learns the leg correction while LIO is good.
- Three checks stop the walk (`lio_doubt`). Any of them also marks LIO lost (`lio_lost`): no further walk, not even after a new GO; the robot never walks on legs alone.
  - A discontinuity: LIO's step between two poses more than 0.25 m off the filter's (a jump).
  - LIO and leg speeds disagreeing (OdomWatchdog): a runaway, or a frozen LIO.
  - LIO refused by the filter's χ² gate for 0.5 s; this one marks LIO lost after 1 s.
  - LIO silent for 0.5 s also marks it lost.
- Tuned on 3 real walks (FAST-LIO replays) and 8 sim walks. With no fault, nothing stopped. Faults injected 2 s into the walk:

  | Fault | Caught | How fast |
  |---|---|---|
  | 0.3 m jump | 11/11 | within 0.1 s |
  | 0.5 m/s runaway | 11/11 | mostly within 1–1.4 s (one 3 s) |
  | 0.3 m/s runaway | 10/11 | 1.2–5.5 s |
  | Frozen pose | every walk that was still moving | about 1 s |

- Slow drift is the limit of a leg-based cross-check: legs are only good to ±10–15 % per second. In a tight gap, walk slowly.
- Offline, on the three lab walks of 10-27, it follows FAST-LIO within 2–7 cm RMS.
- Starting it: `g1 robot --odom fused`, or the hub's *Odometry* choice. The hub refuses it until `g1 lio start` runs.
- In sim: `g1 sim --odom fused` (the LIO stand-in), and `tools/drift_gate.py --levels "fused;fused+jump;..."` for injected LIO faults.

**LIO pose age.** FAST-LIO stamps each pose with its scan's end time, on the lidar clock, which runs within a few ms of this PC's clock.
- The fix: the bridges give the pose filter that time (`adapters/lio.py fix_time`), so it predicts the pose forward to now.
- Before: a pose a scan plus processing old (50–100 ms) counted as current, so at walking speed every frame was registered 2.5–5 cm behind the robot.
- `dds_lio_bridge` prints the mean pose age with its stats. Read it on the next robot day.

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

**Rear-blind study, kcsc (2026-10-26).** Real-time sim, deploy guards, lookahead 1.0, the three settings interleaved:
10 scenes x 3 runs, plus 6 more runs of the door twin (3 goals) and the authored beam door, 162 runs (deploy PC
`runs/rear-blind/`, Deploy Studio sessions `rb-*`). The robot-day sessions were re-planned offline with each setting
(`tools/replay_plans.py`):

| | full lidar | near rear < 1 m blind | rear 180° blind |
|---|---|---|---|
| lab walks (3 goals) + slalom, 12 runs | 12/12 arrived | 12/12 | 12/12 |
| rope behind, beam door + rope, 6 runs | 1/6 arrived | 6/6, no contact | 6/6, 2 contacts |
| door twin + beam door, 36 runs | 28 arrived, 5 tilt E-stops | 26, 9 | 29, 5 |
| robot day door 1 / door 2, safe re-plans | 0/52, 1/22 | 52/52, 14/22 | 52/52, 14/22 |
| robot day walks 1 / 2, safe re-plans | 23/23, 13/13 | 23/23, 13/13 | 23/23, 13/13 |

- 5 vs 9 E-stops is within chance (Fisher p = 0.37). Paired check: 1,036 recorded door/beam states, each re-planned
  with the memory rebuilt from the same lidar under each setting (same planner seed), give the same plans. Near
  minus full planned pelvis tilt is -0.002 rad (95 % CI -0.006 to +0.002); plans backing up 33.8 % vs 33.9 %. The
  blind does not change what kcsc plans. Its door falls are its own: ducking under the overhead with the envelope
  waived, or backing away fast.
- The inverted MID-360 never sees the floor within about 1 m of the robot, with or without the blind, so the near-rear
  blind removes only tall returns there (rope, spotter, shelf). Returns farther than 1 m behind stay mapped.
- The rear 180° blind at all ranges deletes real obstacles behind the robot (the gantry frame at door 1, shelving
  beside a lab turn) and gains nothing over the near-rear blind. Keep the near-rear setting.
- kcsc backs up in about a third of its door plans, most of them faster than 1 m/s, into a zone it cannot see with
  the blind: the spotter stands to the side and at least 1 m back.
- `tools/replay_plans.py` rows now also report `max_tilt`, `backward_m` and `max_backward_speed` of the chosen plan.
- Real-time sim runs are sensitive to other load on the PC: 17 runs that overlapped replays (node lag up to 0.7 s
  against about 0.2 s) were discarded and run again. Run replays and renders after a batch, not during it.

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

## 13. Odometry smear: the stuck walk at a tower gap (lab-g1, 2026-10-27)

**What happened.** The goal was through a 0.69 m gap between two block towers. Walking on leg odometry, the robot turned
toward it, then turned ~90° away; no safe plan followed. A remote press (L2, D-left) then put the runner in its
boxing set (§3).

**Why.**
- *The map smeared.* Leg odometry counted 0.62–0.85 of the distance FAST-LIO measured on that walk (FAST-LIO replayed
  offline on the recorded lidar and IMU, `g1 lio replay --window`), and turned its steps 5–11° right. The node's
  memory only adds evidence, so each frame registered with the drifting pose widened the towers: the gap closed from
  0.62 m standing to 0.45 m after 1 m of walking. FAST-LIO's map kept it at 0.60–0.65 m.
- *The planner turned away.* Re-planning the 7 plans before the turn with the same everything but FAST-LIO's pose and
  map, none turns away (as deployed: all 7 turn 70–82° right). A memory of the last 2 s still turns away (the error
  grows ~0.15 m/s while walking); the last 0.6 s does not, but forgets the robot's blind sides. Clearing the map where
  newer rays pass through (`--memory-clear`, experimental) widens the gap back to 0.55–0.60 m and still turns away 6/7:
  the smear beside the robot, in the body-blocked lidar sectors, cannot be seen through.
- *The rehearsal could not show it.* Sim odometry is exact; and the robot stood 16° right of its scan's heading at GO
  (every walk that day: 16–31°, 0.16–0.26 m; it hangs in the gantry until the runner takes over).

**Rehearsed from the robot's real start** (`tools/repose_course.py`, the scan's twin moved to the start the robot had):
exact odometry and the FAST-LIO stand-in go through the gap (4/4); every leg odometry, the sim's own at 1 or 3 cm or
with the robot's error injected, goes around the left tower instead (8/8).

**How good must odometry be for that gap?** True odometry (the sim LIO stand-in) with an injected error, from the same
start, one run per row unless noted:

| odometry error | through the gap |
|---|---|
| none | 4/4 (with the exact runs) |
| stride ×0.95 | 3/3 |
| steps turned 1° (stride ×1.0 or ×0.97) | 2/2 |
| steps turned 2° | 1/2 |
| stride ×0.9 or ×0.8; steps turned 4° or 8°; ×0.95 and 2°; ×0.9 and 4° | 0/6 (around the left tower) |
| random walk 2 cm/√s in xy and 0.01 rad/√s in yaw | 0/2 |

So about 5 % in distance and 1–2° in direction over the approach. Leg odometry (0.85–0.99 and 5–11°) is outside it.
FAST-LIO standing drifted under 0.5 cm and 0.1° in 50–60 s (§7); walking accuracy is the next robot-day check (< 3 % and < 1° on a
tape-measured 3 m walk).

**What changed.** Leg odometry's contact threshold 3 → 1 cm (0.93 of FAST-LIO's travel on three walks, was 0.81; not
enough on its own). The hub's start check and scene-anchored goals, and exact + drifted rehearsal pairs (§3). The
FAST-LIO watchdog (§7). Error injection for experiments: `G1_LEG_ODOM='{"contact_m": .., "scale": .., "rotate_deg":
..}'` (sim only; the robot refuses scale / rotate_deg) and the sim LIO stand-in's `--lio-noise '{"scale": ..,
"rotate_deg": ..}'` (an error on top of true odometry; the node registers the raw frames with it, so its map smears as
on the robot).

**Margin vs drift (drift gate, 2026-10-28).** The low-margin profile went through the gap with good odometry. Across all odometry levels, though, it touched a tower in 12 of 54 walks (up to 476 N), mostly under odometry error. The default margins never went through, never touched anything (0 of 24) and always arrived. With odometry worse than FAST-LIO, a low route margin trades detours for contacts: use it only with FAST-LIO (or fused) odometry, and slowly.

**For gaps: walk on FAST-LIO** (`g1 lio start`, `g1 lio check`, `--odom zmq`; §7), after a standing check and a
tape-measured walk.

**Tools for odometry robustness (2026-10-28).**
- `tools/drift_gate.py --profile P --course gap-069 [--task T.json] --levels "exact;lio;0.95,0;1.0,-4;motion;leg"`:
  - The full real-time chain, one walk per odometry level and repeat, on isolated ports.
  - Each walk is scored on the simulator's true path:
    - arrival;
    - the task's `gates` (segments the walk must cross: through an opening, not around it);
    - contact.
  - Courses `gap-069` (0.69 m tower gap, straight) and `gap-069-diag` (the same at 30°, the real approach).
  - Levels:
    - `exact`, `lio`: the simulator's odometry and the LIO stand-in;
    - `scale,rot`: an error on true odometry;
    - `motion[:lab-g1-leg|training]`: the step-tied error below, a new draw per repeat;
    - `leg`: the sim's own leg odometry.
- `g1deploy/odom_error.py` (`MotionOdomError`): leg-odometry-like error tied to each step (stride scale, displacement
  heading bias, per-second jitter, turn slip; none while standing), fitted to lab-g1's walks.
  - Its `err` is the lidar memory's `pose_error`, so the memory smears as on the robot.
  - Used by `sim/lockstep.py --odom-noise '{"motion": "lab-g1-leg"}'` and the LIO stand-in (`--lio-noise '{"motion": ...}'`).
- `g1deploy/odom_fusion.py` (`LegLioFilter`): an EKF on (x, y, yaw, leg correction scale and turn).
  - Leg-odometry steps and the IMU's yaw predict; LIO poses correct, behind a χ² gate.
  - LIO rejected for 0.5 s is flagged lost (`lio_lost`): the filter carries on, on calibrated legs, so the robot can stop safely.
  - Offline so far: `tests/test_odom_fusion.py`, and the recorded walks (private analysis).
