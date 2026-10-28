#!/usr/bin/env python3
"""g1 — operator console for the student planner on the G1 (MuJoCo rehearsal or the real robot).

Runs the real deployment chain: SONIC C++ runner (lucid-g1-deploy, `gamepad_manager` input: keyboard + wireless
remote + ZMQ stream) and the student node (g1deploy/node.py), plus, in `sim` mode, the DDS MuJoCo G1 with the task
obstacles and a realistic MID-360 on the robot's own lidar topic. Every session also runs the passive recorder
(g1deploy/recorder.py: lowstate, lowcmd, lidar, streams, node status) and ends with a report (tools/session_report.py).
One screen shows the runner, node, robot and recording state; single keys drive everything:

    s  start SONIC (runner takes control; the robot stands on SONIC's reference)
    g  GO: hand control to the student (runner streams the node's plans; node pre-scans, plans, walks)
    h  student stand (the stream settles to a stand; deferred near ceilings)
    t  takeover: SONIC's own planner, standing (= remote D-up; NOT near overhead obstacles)
    n  new goal ("ahead left" in metres, relative to the robot now), then g
    o  E-STOP: runner damping (the robot drops: gantry/spotter only)   [also the remote's Select]
    r  (sim) release the virtual gantry          q  quit          ?  help

usage:
  g1 sim   [--profile deploy] [--layout open-3m | --task task.json] [--goal AHEAD LEFT] [--mission "A,L;A,L"]
           [--odom dds|leg|lio] [--script STEPS] [--record DIR] [--replay EPISODE] [--busy warn|wait|refuse|off]
  g1 robot [--robot NAME] [--profile deploy] --iface eno1 --goal 3 0 [--odom leg|zmq] [--lidar dds|livox]
           [--mount-calib mount.json] [--body dex3|rubber] [--replay EPISODE]
  (--profile: profiles/README.md; the command line wins over the profile, the profile over the robot profile)
  g1 lab [--port 8769] [--robot lab-g1]                 the robot day's web hub: profiles, sim rehearsal, robot session,
                                                         today's sessions (g1deploy/hub.py)
  g1 check|calibrate|bench|replay|rehearse|test|report|discover|remote|doctor|sessions|note|courses|lidar|lio ...
                                                         (wrappers around tools/, see docs/DEPLOY_KIT.md)

--script runs without a keyboard (tests, videos): comma-separated steps
  wait:init | wait:control | wait:ready | wait:stream | wait:run | wait:arrived | wait:standing | sleep:S | key:K | goal:A:L
  (wait:ready = the student node is up and waiting: GO and goals are refused before that)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import select
import signal
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import OrderedDict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import machine, profiles, service_token, workspace  # noqa: E402
from g1deploy.attempts import TAGS, Attempts  # noqa: E402

LUCID = Path(os.environ.get("LUCID_DEPLOY_ROOT", Path.home() / "lucid-sonic/lucid-g1-deploy"))
RUNNER = LUCID / "runner/target/release/g1_deploy_onnx_ref"
OBS_CONFIG = LUCID / "config/observation_config_lucid_g1_1570.yaml"
PANELS = Path(os.environ.get("G1_TASK_PANELS", "/home/linjiw/flow/artifacts/panels/test-v1"))
COURSES = ROOT / "sim/courses"
ROBOTS = ROOT / "robots"
# The runner's gamepad map (lucid-g1-deploy gamepad_manager.hpp): a D-pad press selects a motion set and leaves the
# student stream, in every input mode; in the boxing set the face and shoulder buttons are punches.
RUNNER_SETS = {"Up": "standing", "Down": "squat/crawl", "Left": "BOXING", "Right": "styled-walking"}
BOXING_KEYS = {"X": "walk boxing", "Y": "RANDOM PUNCHES", "B": "boxing idle", "L1": "LEFT PUNCH", "R1": "RIGHT PUNCH",
               "L2": "LEFT HOOK", "R2": "RIGHT HOOK"}
PORT_BASE = 5556  # the chain's local ports: stream +0, runner debug +1, cloud +4, LIO odometry +5, node control +6
REHEARSAL_LOCK = ROOT / "runs/.rehearsal.lock"  # an isolated rehearsal is running (pid): the robot's GO waits for it


def chain_ports(base=PORT_BASE):
    return {"stream": base, "debug": base + 1, "cloud": base + 4, "odom": base + 5, "control": base + 6}


def rehearsal_running():
    """The pid of an isolated rehearsal running on this PC, or None (a stale lock is ignored)."""
    try:
        pid = int(REHEARSAL_LOCK.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return pid if Path(f"/proc/{pid}").exists() else None


GO_SETTLE_S = 2.0  # GO waits this long after SONIC takes control (sim: the virtual gantry lets go 1.5 s after Start)
CLEAR_MAP_SCAN_S = workspace.CLEAR_MAP_SCAN_S  # GO waits for this much fresh scan after Clear map (node.CLEAR_MAP_SCAN_S)
# Acts (POST /act, keys, the remote, scripts): every one gets an id, a result and a line in console_events.jsonl
ACTS = ("s", "g", "h", "t", "c", "r", "?", "goal", "goal_odom", "arm", "quit", "lease", "tag")  # and estop
ACT_SOURCES = ("panel", "studio", "hub", "remote", "key", "script", "node")
ACT_WAIT_S = 1.0  # an HTTP act waits this long for the main loop, then answers result "queued"
ACT_ID_TTL_S = 600.0  # a repeated act id within this returns its first result (a retry never acts twice)
ACT_RECEIPTS = 500  # GET /act_result?id= answers for the last this many acts of the session, any source (CONTRACT v3 D)
GOAL_READBACK_S = 0.5  # a goal counts once the node's status echoes its seq within this
CLEAR_READBACK_S = 1.5  # Clear map counts once the node reports map_cleared within this
ESTOP_CONFIRM_S = 0.15  # an E-STOP reply looks this long for the runner's EMERGENCY STOP line (`confirmed`)
ESTOP_LINES = ("EMERGENCY STOP", "Emergency Stop")
SESSIONS = Path(os.environ.get("G1_SESSIONS", ROOT / "runs/console"))
# ROS 2 side of the lidar adapter (robots without Unitree's lidar service: livox_ros_driver2 on this PC)
ROS_SETUP = os.environ.get("G1_ROS_SETUP", "/opt/ros/humble/setup.bash")
LIVOX_WS = os.environ.get("G1_LIVOX_WS", str(Path.home() / "click-and-traverse-replicate/Click-and-Traverse/deploy/"
                                                 "Click-and-Traverse-SLAM/lidar_ws/install/setup.bash"))
HELP = __doc__.split("\n\n")[1]


def _env():
    env = dict(os.environ)
    dds = Path(env.get("CYCLONEDDS_HOME", Path.home() / "opt/sonic-deploy-toolchain/cyclonedds"))
    raw = subprocess.run(["bash", "-c", f"source {LUCID}/env.sh >/dev/null && env -0"], capture_output=True).stdout
    for kv in raw.split(b"\0"):
        if kv.startswith((b"onnxruntime_ROOT=", b"LD_LIBRARY_PATH=", b"CUDA_HOME=")):
            k, v = kv.decode().split("=", 1)
            env[k] = v
    env["LD_LIBRARY_PATH"] = f"{dds / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{ROOT}:{LUCID / 'sdk'}"
    # compiled planner kernels (g1deploy/fastplan.py) persist here: /tmp is wiped on reboot -> ~1 min recompile
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache/inductor"))
    return env


def resolve_task(layout):
    """A course of this repo (sim/courses/<name>/task.json) or a panel layout (G1_TASK_PANELS/<name>/task.json)."""
    for base in (COURSES, PANELS):
        if (base / layout / "task.json").exists():
            return str(base / layout / "task.json")
    known = sorted(p.parent.name for p in COURSES.glob("*/task.json"))
    raise SystemExit(f"no layout '{layout}' in {COURSES} or {PANELS}; local courses: {', '.join(known)}")


def mission_script(mission):
    """'A,L;A,L' -> a scripted multi-goal walk (simulation): start, walk the launch goal, then each leg."""
    legs = [tuple(float(v) for v in leg.split(",")) for leg in mission.split(";") if leg.strip()]
    steps = ["wait:init", "sleep:1", "key:s", "wait:control", "wait:ready", "sleep:2.5", "key:g", "wait:run",
             "wait:arrived"]
    for a, l in legs:
        steps += ["sleep:1.5", f"goal:{a}:{l}", "key:g", "wait:run", "wait:arrived"]
    return ",".join(steps + ["sleep:2", "key:t", "sleep:3"])


def _git(*args, cwd=ROOT):
    try:
        return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def session_meta(args, iface):
    """Provenance of a session: what code, bundle, runner, calibration and machine produced it."""
    broot = Path(getattr(args, "bundle", None) or os.environ.get("G1_STUDENT_BUNDLE") or ROOT / "bundle")
    manifest = broot / "MANIFEST.json"
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv[1:], "host": socket.gethostname(),
            "iface": iface, "robot": getattr(args, "robot", None), "operator": args.operator, "note": args.note,
            "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
                    "describe": _git("describe", "--tags", "--always"),
                    # _git strips the output, and with it the first line's leading status column
                    "dirty": [l[2:].strip() for l in (_git("status", "--porcelain", "--untracked-files=no") or "").splitlines()]},
            "bundle": str(broot.resolve()),
            "deploy_rules": getattr(args, "rules", None) or os.environ.get("G1_DEPLOY_RULES"),
            "profile": getattr(args, "profile", None), "deploy_identity": profiles.identity(args),
            "machine": getattr(args, "machine", None),
            "bundle_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
            "runner_commit": _git("rev-parse", "HEAD", cwd=LUCID), "runner": str(RUNNER)}
    mc = getattr(args, "mount_calib", None)
    if mc and Path(mc).exists():
        meta["mount_calib"] = {"path": str(Path(mc).resolve()), "content": json.loads(Path(mc).read_text())}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        meta["gpu"] = gpu or None
    except Exception:
        meta["gpu"] = None
    return meta


def cyclone_xml(iface, path):
    path.write_text(f'<?xml version="1.0" encoding="UTF-8" ?>\n<CycloneDDS><Domain><General><Interfaces>\n'
                    f'  <NetworkInterface name="{iface}" priority="default" multicast="default" />\n'
                    f'</Interfaces></General>\n'
                    f'  <Internal><SocketReceiveBufferSize min="default" max="16MB"/></Internal>\n'
                    f'</Domain></CycloneDDS>\n')
    return path


class Proc:
    """A child process in its own session, stdout stamped with time.monotonic() into a log file."""

    def __init__(self, name, cmd, log, env, stdin=False, on_line=None):
        self.name, self.log_path, self.on_line = name, Path(log), on_line
        self.key_lock = threading.Lock()
        self.f = open(self.log_path, "w", buffering=1)
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT),
                                  stdin=subprocess.PIPE if stdin else subprocess.DEVNULL, start_new_session=True)
        self.lines, self.count, self.lines_lock = [], 0, threading.Lock()  # count: lines ever (lines keeps the last)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        self.log_errors = 0
        for raw in iter(self.p.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip("\n")
            t = time.monotonic()
            try:  # a full disk loses log lines, never the pump: an undrained pipe would block the child (the runner)
                self.f.write(f"{t:.3f} {line}\n")
            except (OSError, ValueError):
                self.log_errors += 1
            with self.lines_lock:
                self.lines.append((t, line))
                self.count += 1
                if len(self.lines) > 4000:
                    del self.lines[:2000]
            if self.on_line is not None:
                try:
                    self.on_line(line)
                except Exception as exc:  # a bad line must not stop the pump (it also feeds the log)
                    try:
                        self.f.write(f"{t:.3f} [console] on_line error: {exc!r}\n")
                    except (OSError, ValueError):
                        self.log_errors += 1

    def new_lines(self, seen):
        """(lines after the first `seen` ever printed, the count now): trimming the buffer never hides or repeats one."""
        with self.lines_lock:
            k = min(self.count - seen, len(self.lines))
            return (self.lines[len(self.lines) - k:] if k > 0 else []), self.count

    def key(self, s):
        with self.key_lock:
            if self.p.poll() is None:
                self.p.stdin.write(s.encode())
                self.p.stdin.flush()

    def alive(self):
        return self.p.poll() is None

    def stop(self, sig=signal.SIGINT, timeout=5.0):
        for s_ in (sig, signal.SIGTERM, signal.SIGKILL):  # children may have inherited SIGINT ignored
            if self.p.poll() is not None:
                return
            try:
                os.killpg(self.p.pid, s_)
                self.p.wait(timeout)
            except subprocess.TimeoutExpired:
                continue
            except ProcessLookupError:
                return


class Session:
    def __init__(self, args):
        self.args, self.env = args, _env()
        if getattr(args, "bundle", None):  # the node's planner bundle (G1_STUDENT_BUNDLE); SONIC files are shared
            self.env["G1_STUDENT_BUNDLE"] = str(Path(args.bundle).resolve())
        if getattr(args, "rules", None):  # its deploy rules plugin (bundle.load_rules)
            self.env["G1_DEPLOY_RULES"] = str(Path(args.rules).resolve())
        if getattr(args, "rules_overrides", None):  # the plugin's own settings (profiles.KEYS rules_overrides)
            self.env["G1_RULES_OVERRIDES"] = json.dumps(args.rules_overrides, sort_keys=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        label = f"-{args.robot}" if getattr(args, "robot", None) else ""
        self.dir = Path(args.record) if args.record else SESSIONS / f"{args.mode}-{stamp}{label}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.procs, self.events = {}, []
        self.runner_state = {"init": False, "control": False, "iface": "GAMEPAD", "stream": False, "estop": False}
        self.node_status = {}
        self.rec_stats, self.rec_read = {}, 0.0
        self.frames = open(self.dir / "console_frames.jsonl", "w", buffering=1)
        self.req = None
        self.t0 = time.monotonic()
        self.inbox = queue.Queue()  # panel buttons and remote takeover -> the main loop (E-stop never waits here)
        self.remote = {"connected": False, "samples": 0, "last": None}
        self.panel_url = None
        self.ports = chain_ports(getattr(args, "port_base", None) or PORT_BASE)
        self.ctl = f"tcp://localhost:{self.ports['control']}"
        # the robot-day workspace (panel): attempts, motor temperatures, the node's map, the ladder's timestamps
        prof = getattr(args, "profile", None)
        self.attempts = Attempts(self.dir / "attempts.jsonl", prof["name"] if prof else None,
                                 on_error=lambda exc: self.write_failed("attempts.jsonl", exc))
        self.temps, self.map_lock, self.map_cache = None, threading.Lock(), (0.0, None)
        self.cleared_t = self.control_t = None
        try:
            self.identity = profiles.identity(args).get("short")
        except Exception:
            self.identity = None
        self.init_acts(args)

    def init_acts(self, args):
        """The act and GO-gate state (CONTRACT §4.1). Times are session time (monotonic - t0), like control_t.
        The goal in force: goal_t = when the node read it back (None: no confirmed goal), goal_seq its seq, goal_xy the
        node's goal then. The launch goal (--goal, anchored by the node at its start) is in force with goal_seq None;
        it never counts on a robot console; a sim console counts it for the first cycle (start_goal_counts) unless
        --require-sent-goal (practice sessions: GO behaves as on the robot)."""
        self.goal_rel = tuple(args.goal) if getattr(args, "goal", None) else None
        self.goal_t, self.goal_seq, self.goal_xy, self.goal_ctx, self.seqs = None, None, None, {}, set()
        self.start_goal_counts = args.mode == "sim" and not getattr(args, "require_sent_goal", False)
        self.act_lock, self.log_lock = threading.Lock(), threading.Lock()
        self.acts_by_id, self.acts_recent, self.pending, self.n_acts = {}, deque(maxlen=20), [], 0
        self.receipts, self.receipt_lock = OrderedDict(), threading.Lock()  # id -> act record, the last ACT_RECEIPTS
        self.write_errors = {}  # file -> [count, last error]: the console's own records on a failing disk
        self.lease, self.quit_requested, self.estop_logged = None, False, False
        self.node_seen, self.node_events = 0, []
        self.events_log = open(self.dir / "console_events.jsonl", "a", buffering=1)

    # -- processes ----------------------------------------------------------------------------------
    def start(self):
        a, py = self.args, str(ROOT / ".venv/bin/python")
        planner = ROOT / "bundle/sonic/planner/V2/planner_sonic.onnx"
        onnx = ROOT / "bundle/sonic/sonic_release_g1.onnx"
        motions = ROOT / "sim/realtime/motions"
        sim = a.mode == "sim"
        iface = "lo" if sim else a.iface
        if a.record_data:  # first, so it sees the runner's INIT ramp and the very first lowcmd
            P = self.ports
            moved = [] if P["stream"] == PORT_BASE else [  # an isolated rehearsal: the recorder follows its ports
                "--zmq", f"pose=pose=tcp://127.0.0.1:{P['stream']}", "--zmq", f"debug=g1_debug=tcp://127.0.0.1:{P['debug']}",
                "--zmq", f"odom=odom=tcp://127.0.0.1:{P['odom']}", "--zmq", f"cloud=cloud=tcp://127.0.0.1:{P['odom']}",
                "--status", f"tcp://127.0.0.1:{P['control']}"]
            self.procs["rec"] = Proc("rec", [py, "-u", "g1deploy/recorder.py", "--iface", iface, "--out",
                                             str(self.dir / "rec")] + ([] if a.record_lidar else ["--no-lidar"]) + moved,
                                     self.dir / "recorder.log", self.env)
            self.event("recorder: lowstate, lowcmd, lidar, streams, node status -> rec/")
        if sim:
            self.task = a.task or resolve_task(a.layout)
            (self.dir / "remote").mkdir(exist_ok=True)
            self.procs["sim"] = Proc("sim", [py, "-u", "sim/realtime/robot_sim.py", "--task", self.task,
                                             "--log", str(self.dir / "sim.npz"), "--lidar-io", "dds",
                                             "--release-file", str(self.dir / "release.flag"),
                                             "--remote-dir", str(self.dir / "remote")] + (["--viewer"] if a.viewer else []),
                                     self.dir / "sim.log", self.env)
            self.event("sim: MuJoCo G1 on DDS (lo), " + Path(self.task).parent.name)
            if a.odom in ("lio", "fused"):  # stands in for Point-LIO + adapters/ros2_lio_bridge.py: odom + registered cloud :5561
                self.procs["lio"] = Proc("lio", [py, "-u", "tools/sim_lio_bridge.py", "--iface", "lo",
                                                 "--bind", f"tcp://*:{self.ports['odom']}"]
                                         + (["--noise", a.lio_noise] if a.lio_noise else []),
                                         self.dir / "lio.log", self.env)
                self.event("sim: LIO stand-in on :5561 (odom + registered cloud)")
        elif a.lidar == "livox":
            self.start_livox(iface)
        if not sim:  # the remote's Select / D-up in every runner input mode (g1deploy/remote_watch.py)
            self.procs["remote"] = Proc("remote", [py, "-u", "g1deploy/remote_watch.py", "--iface", iface],
                                        self.dir / "remote_watch.log", self.env, on_line=self.on_remote)
            self.event("remote watcher: Select = E-STOP and D-up = takeover, also while the student streams")
        if not sim and a.odom in ("zmq", "fused") and subprocess.run(["pgrep", "-f", "^[^ ]*python[^ ]* -u [^ ]*adapters/dds_lio_bridge.py"],
                                                          capture_output=True).returncode:
            self.event("odom zmq: NO LIO bridge on :5561, the node will wait for odometry (`g1 lio start`, standing)")
        meta = session_meta(a, iface)
        meta.update(mode=a.mode, task=getattr(self, "task", None), goal=list(a.goal), t0=self.t0,
                    replay=a.replay, odom=a.odom, lidar=None if sim else a.lidar, body=a.body, guards=a.guards,
                    goal_lookahead=a.goal_lookahead, rear_blind_deg=a.rear_blind_deg,
                    rules_overrides=getattr(a, "rules_overrides", None),
                    rear_blind_range_m=a.rear_blind_range_m,
                    # the operator's start confirmations (the hub's checklist: gantry, E-stops, remote, rope, ...)
                    confirm=getattr(a, "confirm", None),
                    console_pid=os.getpid(),  # g1deploy/evidence.py: is this session still running?
                    go_gate={"start_goal_counts": self.start_goal_counts, "settle_s": GO_SETTLE_S,
                             "clear_map_scan_s": CLEAR_MAP_SCAN_S, "goal_readback_s": GOAL_READBACK_S})
        (self.dir / "session.json").write_text(json.dumps(meta, indent=1) + "\n")
        self.launch = (iface, onnx, motions, planner, py, sim)
        if getattr(a, "arm", False):  # the runner waits for the panel's Arm: recorder, lidar, remote, panel run first
            self.event("ARM to start the runner: it releases Unitree's controller and ramps to the stand pose")
            return
        self.start_runner()

    def start_runner(self):
        a = self.args
        iface, onnx, motions, planner, py, sim = self.launch
        if "runner" in self.procs:
            return
        runner_cmd = [str(RUNNER), iface, str(onnx), f"{motions}/", "--obs-config", str(OBS_CONFIG),
                      "--input-type", "gamepad_manager", "--planner-file", str(planner),
                      "--zmq-host", "localhost", "--zmq-port", str(self.ports["stream"]), "--zmq-topic", "pose"]
        if self.ports["debug"] != PORT_BASE + 1:
            runner_cmd += ["--zmq-out-port", str(self.ports["debug"])]
        if sim:
            runner_cmd.append("--disable-crc-check")  # the simulator computes no CRC; never on the robot
        if a.runner_csv:  # the runner's own per-tick CSVs (lucid run.sh records these on every hardware run)
            runner_cmd += ["--enable-csv-logs", "--logs-dir", str(self.dir / "runner_csv")]
        self.procs["runner"] = Proc("runner", runner_cmd, self.dir / "runner.log", self.env, stdin=True)
        self.event(f"runner: SONIC C++ deploy, gamepad_manager on {iface}" + (", CSV logs on" if a.runner_csv else ""))
        if a.replay:  # H4: the recorded stream starts on [g]; no planner node
            self.event(f"replay: {Path(a.replay).name} at 1/{a.replay_slow:g} speed starts on [g] (no planner)")
            return
        node_cmd = [py, "-u", "g1deploy/node.py", "--iface", iface, "--cloud", "dds", "--wait-go",
                    "--goal-frame", "start", "--goal", str(a.goal[0]), str(a.goal[1]),
                    "--log", str(self.dir / "node_plans.json"), "--dump-memory", str(self.dir / "node_memory.npz"),
                    # loopback only: everything that talks to the node runs on this PC; never the robot LAN or Wi-Fi
                    "--stream-bind", f"tcp://127.0.0.1:{self.ports['stream']}",
                    "--control", f"tcp://127.0.0.1:{self.ports['control']}",
                    "--odom-connect", f"tcp://localhost:{self.ports['odom']}"]
        if not sim:
            node_cmd += ["--odom", a.odom] + (["--mount-calib", a.mount_calib] if a.mount_calib else [])
        elif a.odom in ("leg", "lio", "fused"):
            node_cmd += ["--odom", {"leg": "leg", "lio": "zmq", "fused": "fused"}[a.odom]]
        if a.goal_lookahead:
            node_cmd += ["--goal-lookahead", str(a.goal_lookahead)]
        if a.candidates:
            node_cmd += ["--candidates", str(a.candidates)]
        if a.ode_steps:
            node_cmd += ["--ode-steps", str(a.ode_steps)]
        node_cmd += ["--planner-impl", a.planner_impl, "--planner-precision", a.planner_precision]
        if a.debug_plans:
            node_cmd += ["--debug-dump", str(self.dir / "plans")]
        if a.guards != "off":
            node_cmd += ["--guards", a.guards]
        if a.self_margin is not None:
            node_cmd += ["--self-margin", str(a.self_margin)]
        if a.memory_clear:
            node_cmd += ["--memory-clear"]
        if a.rear_blind_deg:
            node_cmd += ["--rear-blind-deg", str(a.rear_blind_deg)]
            if a.rear_blind_range_m:
                node_cmd += ["--rear-blind-range-m", str(a.rear_blind_range_m)]
        env = dict(self.env, G1_BODY_MODEL=a.body)
        self.procs["node"] = Proc("node", node_cmd, self.dir / "node.log", env)
        self.event(f"node: student planner, goal {a.goal[0]:.1f} m ahead {a.goal[1]:+.1f} m left, odom {a.odom}")
        import zmq

        self.zmq = zmq
        self.req = zmq.Context.instance().socket(zmq.REQ)
        self.req.setsockopt(zmq.LINGER, 0)
        self.req.connect(self.ctl)

    def start_livox(self, iface):
        """Robots without Unitree's lidar service: livox_ros_driver2 on this PC (started here unless one already runs)
        + g1deploy/adapters/livox_to_utlidar.py -> rt/utlidar/cloud_livox_mid360 for the node."""
        xml = cyclone_xml(iface, self.dir / f"cyclonedds_{iface}.xml")
        ros = f"source {ROS_SETUP} >/dev/null && source {LIVOX_WS} >/dev/null && " \
              f"export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://{xml} && "
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
        running = subprocess.run(["pgrep", "-f", "^[^ ]*livox_ros_driver2_node"], capture_output=True).returncode == 0
        if running:
            self.event("lidar: livox_ros_driver2 already running on this PC (reused)")
        else:
            self.procs["livox"] = Proc("livox", ["bash", "-c", ros + "exec ros2 launch livox_ros_driver2 "
                                                 "msg_MID360_launch.py"], self.dir / "livox.log", env)
            self.event("lidar: started livox_ros_driver2 (MID-360 -> this PC)")
        if subprocess.run(["pgrep", "-f", "^/usr/bin/python3 .*livox_to_utlidar"], capture_output=True).returncode == 0:
            # a second adapter would publish every frame twice and the node would fuse each one twice
            self.event("lidar: livox_to_utlidar adapter already running (reused; `g1 lidar stop` ends it)")
            return
        self.procs["lidar"] = Proc("lidar", ["bash", "-c", ros + "exec /usr/bin/python3 -u "
                                             f"{ROOT}/g1deploy/adapters/livox_to_utlidar.py --stats-s 5 --custom-out /livox/lidar_10hz"],
                                   self.dir / "lidar_adapter.log", env)
        self.event("lidar: /livox/lidar -> rt/utlidar/cloud_livox_mid360 (10 Hz frames, tag filter)")

    def event(self, text):
        self.events.append((time.monotonic() - self.t0, text))
        self.events = self.events[-12:]

    # -- monitoring -------------------------------------------------------------------------------------
    def poll(self):
        r = self.procs.get("runner")
        if r:
            for _, line in r.lines[-300:]:
                st = self.runner_state
                if "Init Done" in line and not st["init"]:
                    st["init"] = True
                    self.event("runner: Init Done (standing ramp complete)")
                if "transitioning to CONTROL" in line and not st["control"]:
                    st["control"] = True
                    self.control_t = time.monotonic() - self.t0
                    self.event("runner: SONIC in CONTROL")
            for _, line in r.lines[-60:]:
                st = self.runner_state
                if "Switched to: ZMQ" in line:
                    st["iface"], st["set"] = "ZMQ", None
                elif "Switched to: GAMEPAD" in line:
                    st["iface"], st["stream"] = "GAMEPAD", False
                elif "ZMQ STREAMING MODE: ENABLED" in line:
                    st["stream"] = True
                elif "STREAMING MODE: DISABLED" in line or "FORCE DISABLED" in line:
                    st["stream"] = False
                elif any(s_ in line for s_ in ESTOP_LINES):
                    st["estop"] = self.estop_logged = True
        n = self.procs.get("node")
        if n and not self.args.replay:
            lines, self.node_seen = n.new_lines(self.node_seen)
            for _, line in lines:
                if not line.startswith("{\"event\""):
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.node_events = (self.node_events + [e])[-500:]
                if e["event"] not in ("stall_risk", "dq_clipped"):
                    self.event("node: " + e["event"] + ("" if "reason" not in e else f" ({e['reason']})"))
                if e["event"] == "map_cleared":
                    self.cleared_t = float(e.get("t", time.monotonic())) - self.t0
                if e["event"] == "estop" and not self.runner_state["estop"]:
                    # the node's supervisor asked for a stop (tilt / stale state): damping now, no operator needed
                    self.procs["runner"].key("o")
                    self.runner_state["estop"] = True
                    self.event("AUTO E-STOP: runner damping (node supervisor)")
                    self.log_act({"id": self.new_id("node"), "act": "estop", "source": "node", "operator": None,
                                  "args": {"reason": e.get("reason")}, "t_mono": time.monotonic(), "wall": time.time()},
                                 "done", f"node supervisor: {e.get('reason')}")
        if self.req is not None:
            try:
                self.req.send_string(json.dumps({"cmd": "status"}), self.zmq.NOBLOCK)
                if self.req.poll(150):
                    self.node_status = json.loads(self.req.recv())
                else:  # REQ socket stuck without a reply: rebuild it
                    self.req.close(0)
                    self.req = self.zmq.Context.instance().socket(self.zmq.REQ)
                    self.req.setsockopt(self.zmq.LINGER, 0)
                    self.req.connect(self.ctl)
            except self.zmq.ZMQError:
                pass
        self.check_pending()
        self.attempts.update(time.monotonic() - self.t0, self.node_status, estop=self.runner_state["estop"])
        if "rec" in self.procs and time.monotonic() - self.rec_read > 1.0:
            self.rec_read = time.monotonic()
            try:
                self.rec_stats = json.loads((self.dir / "rec/stats.json").read_text())
            except (OSError, json.JSONDecodeError):
                pass

    NODE_READY = ("WAIT", "PRESCAN", "RUN", "ARRIVED", "STOPPED")  # node phases with its command socket up

    def node_ready(self):
        """The node has finished building its planner (warm-up: ~25 s for the deployed bundle, longer for bundles
        with larger graphs) and answers commands. Before that a goal or GO would be lost."""
        return "node" in self.procs and self.node_status.get("phase") in self.NODE_READY

    def go_settle_left(self):
        if self.control_t is None:
            return 0.0
        return max(0.0, GO_SETTLE_S - (time.monotonic() - self.t0 - self.control_t))

    def node_cmd(self, **msg):
        try:
            s = self.zmq.Context.instance().socket(self.zmq.REQ)
            s.setsockopt(self.zmq.LINGER, 0)
            s.connect(self.ctl)
            s.send_string(json.dumps(msg))
            ok = s.poll(1000)
            if ok:
                s.recv()
            s.close(0)
            return bool(ok)
        except Exception:
            return False

    # -- acts: every operator action, from every source (CONTRACT §4.1) --------------------------------------
    def new_id(self, prefix):
        with self.act_lock:
            self.n_acts += 1
            return f"{prefix}-{os.getpid()}-{self.n_acts}"

    def make_act(self, act, source, operator=None, act_id=None, expect=None, **args):
        """A new act record, findable by its id (GET /act_result) from this moment: before it runs, before any reply."""
        rec = {"id": str(act_id) if act_id is not None else self.new_id(source), "act": act, "source": source,
               "operator": operator if operator is not None else getattr(self.args, "operator", None),
               "expect": expect or None, "args": args, "t_mono": time.monotonic(), "wall": time.time(),
               "result": None, "reason": None, "state": "queued", "t_done": None, "done": threading.Event()}
        with self.receipt_lock:
            self.receipts.pop(rec["id"], None)  # a lease renewal reusing its id: the newest record answers
            self.receipts[rec["id"]] = rec
            while len(self.receipts) > ACT_RECEIPTS:
                self.receipts.popitem(last=False)
        return rec

    def act_result(self, act_id):
        """GET /act_result?id=: what became of an act (a reply lost in transit, a timed-out relay): {known, id, act,
        result: done | refused | queued (not handled yet) | pending (waiting for the node to confirm it), reason,
        t_mono (when the console received it), t_done, source, operator, session}. Never acts."""
        with self.receipt_lock:
            rec = self.receipts.get(str(act_id))
        if rec is None:
            return {"known": False, "id": act_id, "act": None, "result": None, "t_mono": None, "t_done": None,
                    "reason": f"no act with this id in this console session (it never arrived, or it is older than the "
                              f"last {ACT_RECEIPTS} acts)", "source": None, "operator": None, "session": self.dir.name}
        res = rec["result"] or rec["state"]
        why = rec["reason"] if rec["result"] else ("waiting for the node to confirm it" if res == "pending"
                                                   else "the console's main loop has not handled it yet")
        return {"known": True, "id": rec["id"], "act": rec["act"], "result": res, "reason": why,
                "t_mono": round(rec["t_mono"], 3), "t_done": rec["t_done"], "source": rec["source"],
                "operator": rec.get("operator"), "session": self.dir.name}

    def write_failed(self, name, exc):
        """One of the console's own files could not be written (disk full): counted, shown once, never fatal."""
        n = self.write_errors.setdefault(name, [0, None])
        n[0] += 1
        n[1] = f"{type(exc).__name__}: {exc}"
        if n[0] == 1:
            self.event(f"cannot write {name} ({n[1]}): the session goes on, this record has a hole")

    def log_act(self, rec, result, reason, final=True):
        """One line per act in console_events.jsonl (whatever its source) and the last 20 in /state.acts. final=False:
        the line of an act answered "queued" (its HTTP reply left before its result; the result line follows)."""
        row = {"t_mono": round(rec["t_mono"], 3), "wall": round(rec["wall"], 3), "id": rec["id"],
               "source": rec["source"], "operator": rec.get("operator"), "act": rec["act"], "args": rec.get("args") or {},
               "result": result, "reason": reason, "t_done": round(time.monotonic(), 3) if final else None}
        if final:
            rec["t_done"] = row["t_done"]
        with self.log_lock:
            if not final:  # checked under the lock: a result line never precedes its queued line
                if rec["result"] is not None or rec.get("queued_logged"):
                    return
                rec["queued_logged"] = True
            try:
                self.events_log.write(workspace.dumps(row) + "\n")
            except ValueError:  # the log is closed at shutdown: the act itself stands
                pass
            except OSError as exc:  # disk full: the act stands, its receipt answers from memory
                self.write_failed("console_events.jsonl", exc)
            if final:
                self.acts_recent.append({k: row[k] for k in ("id", "act", "source", "operator", "result", "reason",
                                                             "t_mono")})

    def finish(self, rec, result, reason=None):
        rec["result"], rec["reason"], rec["state"] = result, reason, result
        self.log_act(rec, result, reason)
        if result == "refused":
            self.event(f"{rec['source']} [{rec['act']}] refused: {reason}")
        rec["done"].set()
        return rec

    def act_reply(self, rec, **extra):
        res = rec["result"] or "queued"
        return {"ok": res != "refused", "id": rec["id"], "act": rec["act"], "result": res,
                "reason": rec["reason"] if rec["result"] else "the console's main loop has not handled it yet",
                "session": self.dir.name, **extra}

    @staticmethod
    def validate_act(msg):
        """What is wrong with an HTTP act before it is queued (None: fine)."""
        act = msg.get("act")
        if act not in ACTS:
            return f"unknown act {act!r}"
        if msg.get("id") in (None, ""):
            return "id is required (a client uuid; a retry with the same id never acts twice)"
        if msg.get("source", "panel") not in ACT_SOURCES:
            return f"source must be one of {', '.join(ACT_SOURCES)}"
        if msg.get("expect") is not None and not isinstance(msg["expect"], dict):
            return "expect must be an object"

        def num(*keys):
            for k in keys:
                v = msg.get(k)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    return f"{act} needs a finite number {k}"
        if act == "goal":
            return num("ahead", "left")
        if act == "goal_odom":
            return num("x", "y") or (None if msg.get("seq") not in (None, "") and msg.get("check_id") not in (None, "")
                                     else "goal_odom needs seq and check_id")
        if act == "tag" and not isinstance(msg.get("n"), int):
            return "tag needs the attempt number n"
        if act == "lease" and msg.get("holder") and msg.get("until") is None and msg.get("ttl_s") is None:
            return "lease needs until (unix seconds) or ttl_s"
        return None

    def submit(self, msg):
        """POST /act from the panel's HTTP thread (the panel page, the hub, Studio's gateway: through the hub, or
        straight here for E-STOP, Stand and Takeover). E-STOP runs here and now (no id needed); every other act is
        validated, a repeated id returns its first result (except `lease`, re-applied), then it is queued for the main
        loop and the reply waits up to ACT_WAIT_S for its result. Returns (HTTP code, reply)."""
        if msg.get("act") == "estop":
            src = msg.get("source") or "panel"
            return 200, self.estop(f"{src} E-STOP", operator=msg.get("operator"), act_id=msg.get("id"),
                                   kind=src if src in ACT_SOURCES else "panel")
        bad = self.validate_act(msg)
        if bad:
            return 400, {"ok": False, "id": msg.get("id"), "act": msg.get("act"), "result": "refused", "reason": bad}
        args = {k: v for k, v in msg.items() if k not in ("act", "id", "source", "operator", "expect")}
        key = str(msg["id"])
        with self.act_lock:
            now = time.monotonic()
            for k in [k for k, r in self.acts_by_id.items() if now - r["t_mono"] > ACT_ID_TTL_S]:
                del self.acts_by_id[k]
            rec = self.acts_by_id.get(key)
            # a lease renewal may reuse its id: setting the lease is idempotent, so it is applied again (new until)
            repeat = rec is not None and msg["act"] != "lease"
            if not repeat:
                rec = self.acts_by_id[key] = self.make_act(msg["act"], msg.get("source") or "panel", msg.get("operator"),
                                                           key, msg.get("expect"), **args)
        if not repeat:
            self.inbox.put(("act", rec, rec["source"]))
        if not rec["done"].wait(ACT_WAIT_S):  # the reply will say queued: its line is on disk first
            self.log_act(rec, "queued", "replied before the main loop handled it", final=False)
        return 200, self.act_reply(rec, repeat=repeat)

    def act(self, key, source="key", **args):
        """A key (terminal, script, the remote) on the main loop, now; returns the act record (None for E-STOP)."""
        if key == "o":
            self.estop("operator key" if source == "key" else f"{source} o", kind=source)
            if not self.args.replay:
                self.node_cmd(cmd="quit")
            return None
        return self.run_act(self.make_act(key, source, **args))

    def run_act(self, rec):
        """Main loop: one act -> done | refused (with its reason) | pending (a goal or Clear map waits for the node to
        confirm it; check_pending ends it). Every result goes to console_events.jsonl."""
        why = self.expect_mismatch(rec.get("expect"))
        if why and rec["act"] in ("h", "t"):  # a stop is never refused over bookkeeping: stand now, note the mismatch
            self.event(f"{rec['source']}: {rec['act']} despite a stale page ({why})")
        elif why:
            return self.finish(rec, "refused", why)
        handler = {"goal": self.act_goal, "goal_odom": self.act_goal_odom, "arm": self.act_arm, "quit": self.act_quit,
                   "lease": self.act_lease, "tag": self.act_tag}.get(rec["act"], self.act_key)
        try:
            out = handler(rec)
        except (KeyError, TypeError, ValueError) as exc:
            out = ("refused", f"bad arguments: {type(exc).__name__}: {exc}")
        if out is not None:
            self.finish(rec, *out)
        elif rec["result"] is None:
            rec["state"] = "pending"  # waits for the node (goal read-back, Clear map); check_pending ends it
        return rec

    def expect_mismatch(self, expect):
        """An act's `expect` against the console now (a stale page must not act on a state that has changed)."""
        if not expect:
            return None
        now = {"session": self.dir.name, "runner": self.runner_label(), "phase": self.node_status.get("phase"),
               "goal_seq": self.goal_seq, "attempts": len(self.attempts.rows)}
        for k, v in expect.items():
            if k not in now:
                return f"unknown expect field {k!r} (session, runner, phase, goal_seq, attempts)"
            if now[k] != v:
                return f"the state changed: {k} is {now[k]!r} now, the page expected {v!r}; look again"
        return None

    def check_pending(self):
        """Main loop (poll): end the acts that wait for the node (goal read-back, Clear map)."""
        now = time.monotonic()
        for p in list(self.pending):
            rec, deadline, check, timeout_reason = p
            out = check()
            if out is None and now > deadline:
                out = ("refused", timeout_reason)
            if out is not None:
                self.pending.remove(p)
                self.finish(rec, *out)

    def pending_goal(self):
        return next((p[0] for p in self.pending if p[0]["act"] in ("goal", "goal_odom")), None)

    # -- the keys ---------------------------------------------------------------------------------------------
    def act_key(self, rec):
        key = rec["act"]
        if key == "?":
            for line in HELP.splitlines():
                self.event(line.strip())
            return "done", None
        if "runner" not in self.procs:
            return "refused", "the runner is not started (Arm first)"
        r, st = self.procs["runner"], self.runner_state
        if key == "s":
            if st["estop"]:
                return "refused", "the runner is in E-STOP: end the session"
            if st.get("control"):
                return "refused", "SONIC is already in CONTROL"
            if not st.get("init"):  # Start SONIC during the stand ramp dropped the practice robot (2026-10-28)
                return "refused", "the runner is still booting (stand ramp): Start SONIC after INIT done"
            if self.args.mode == "sim" and self.args.remote_start:
                (self.dir / "remote" / "start").touch()  # virtual remote Start (enables SONIC's planner)
                self.event("operator: remote Start")
            else:
                r.key("#")  # ZMQ interface (keyboard path), then start control
                time.sleep(0.2)
                r.key("]")
                self.event("operator: start SONIC control")
            if self.args.mode == "sim":
                threading.Timer(1.5, lambda: (self.dir / "release.flag").touch()).start()
                self.event("sim: gantry releases in 1.5 s")
            return "done", "SONIC start sent"
        if key == "g":
            return self.act_go(rec)
        if key == "h":
            if self.args.replay:
                return "refused", "replay: no student node to stand"
            ok = self.node_cmd(cmd="stand")
            self.event("operator: student stand" if ok else "STAND NOT DELIVERED: the node did not answer")
            self.attempts.close(time.monotonic() - self.t0, "operator stand", self.node_status)
            return ("done", "student stand") if ok else ("refused", "the node did not answer: Takeover or E-STOP")
        if key == "t":
            r.key("@")  # SONIC's own planner (safety reset; streaming off)
            if self.args.replay and "node" in self.procs:
                self.procs["node"].stop()
            elif not self.args.replay:
                self.node_cmd(cmd="stand")
            self.event("operator: TAKEOVER -> SONIC planner")
            self.attempts.close(time.monotonic() - self.t0, "takeover", self.node_status)
            return "done", "takeover: SONIC's planner, standing"
        if key == "c":
            return self.act_clear(rec)
        if key == "r" and self.args.mode == "sim":
            (self.dir / "release.flag").touch()
            self.event("sim: gantry released")
            return "done", "gantry released"
        return "refused", f"unknown key {key!r}"

    def act_go(self, rec):
        """GO: refused with every reason the gate has (go_locks), checked BEFORE the runner is switched to the ZMQ
        stream (a refused GO leaves the runner as it was). Opens an attempt with the goal, start check and rehearsal set
        in force."""
        locks = self.go_locks(rec["source"])
        if locks:
            return "refused", "; ".join(locks)
        r, st = self.procs["runner"], self.runner_state
        # the stream is never on before SONIC controls the robot (go_locks): SONIC would take control tracking plans
        # anchored on the hanging robot (seen in simulation 2026-10-24: the robot ended crouched)
        if st["iface"] != "ZMQ":
            r.key("#")
            time.sleep(0.2)
        if not st["stream"]:
            r.key("\n")  # ZMQ streaming on
        if self.args.replay:
            if "node" not in self.procs:
                py = str(ROOT / ".venv/bin/python")
                self.procs["node"] = Proc("node", [py, "-u", "tools/stream_replay.py", self.args.replay,
                                                   "--slow", str(self.args.replay_slow),
                                                   "--bind", f"tcp://127.0.0.1:{self.ports['stream']}"],
                                          self.dir / "node.log", self.env)
                self.event("operator: GO (stream on, recorded reference plays)")
            return "done", "replay: the recorded reference plays"
        if not self.node_cmd(cmd="go"):
            self.event("GO NOT DELIVERED: the node did not answer (stream is on, the student is not planning)")
            return "refused", "the node did not answer (the stream is on, the student is not planning): Stand or Takeover"
        self.event("operator: GO (stream on, student plans)")
        ctx, args = self.goal_ctx, rec["args"]
        self.attempts.start(time.monotonic() - self.t0, self.node_status, goal_rel=self.goal_rel,
                            goal_seq=self.goal_seq, goal_kind=ctx.get("kind", "launch"), goal_odom=ctx.get("odom"),
                            check_id=args.get("check_id") or ctx.get("check_id"),
                            start_check=args.get("start_check") or ctx.get("start_check"),
                            rehearsal=args.get("rehearsal") or ctx.get("rehearsal"),
                            # the operator's explicit "walk without a matching rehearsal" (robot GO), the goal in the
                            # scan's scene and the gate it must pass: kept with the attempt for review and evidence
                            rehearsal_override=args.get("rehearsal_override"), goal_scene=ctx.get("goal_scene"),
                            scan=ctx.get("scan"), passage=args.get("passage") or ctx.get("passage"),
                            odom=getattr(self.args, "odom", None), odom_state=self.node_status.get("odom"),
                            operator=rec.get("operator"), source=rec["source"], act_id=rec["id"])
        return "done", f"GO: attempt {len(self.attempts.rows)}"

    def act_clear(self, rec):
        """Clear map: done when the node reports map_cleared (then GO waits CLEAR_MAP_SCAN_S of fresh scan)."""
        if self.args.replay:
            return "refused", "replay: no planner"
        if not self.node_ready():
            return "refused", "the student node is not up yet (it starts on Arm)"
        if self.node_status.get("phase") in ("PRESCAN", "RUN"):
            return "refused", "stand first: the walk plans on this map"
        t_sent = time.monotonic()
        if not self.node_cmd(cmd="clear_map"):
            return "refused", "the node did not answer"
        self.event("operator: CLEAR MAP (the node maps again from here; walks wait 10 s)")

        def check():
            for e in reversed(self.node_events):
                if float(e.get("t", 0.0)) < t_sent - 0.01:
                    break
                if e["event"] == "map_cleared":
                    self.cleared_t = float(e["t"]) - self.t0
                    return "done", f"map cleared; GO waits {CLEAR_MAP_SCAN_S:.0f} s of scan"
                if e["event"] == "command_refused" and e.get("cmd") == "clear_map":
                    return "refused", f"the node refused: {e.get('reason')}"
            return None
        self.pending.append((rec, time.monotonic() + CLEAR_READBACK_S, check, "the node did not confirm Clear map"))
        return None

    # -- goals --------------------------------------------------------------------------------------------------
    def next_seq(self):
        return max([s for s in self.seqs if isinstance(s, int) and not isinstance(s, bool)] + [0]) + 1

    def send_goal(self, rec, seq, ctx, **cmd):
        """Deliver a goal with `seq`; it counts (goal_t) only once the node's status echoes the seq within
        GOAL_READBACK_S, else the act is refused "goal not confirmed by the node". Until then no goal counts."""
        if self.args.replay:
            return "refused", "replay: no planner"
        if not self.node_ready():
            return "refused", "the student node is still starting (planner warm-up)"
        if seq in self.seqs:
            return "refused", f"goal seq {seq!r} was already used in this session"
        old = self.pending_goal()
        if old is not None:
            self.pending = [p for p in self.pending if p[0] is not old]
            self.finish(old, "refused", f"superseded by act {rec['id']}")
        self.seqs.add(seq)
        self.goal_t = None  # nothing counts until the node has read this one back
        if not self.node_cmd(cmd="goal", seq=seq, **cmd):
            self.event("goal NOT DELIVERED: the node did not answer")
            return "refused", "the node did not answer"

        def check():
            ns = self.node_status
            if ns.get("goal_seq") != seq:
                return None
            self.goal_t, self.goal_seq, self.goal_xy = time.monotonic() - self.t0, seq, ns.get("goal")
            self.goal_rel = tuple(ctx["rel"]) if ctx.get("rel") else None
            self.goal_ctx = dict(ctx, seq=seq, t=round(self.goal_t, 2))
            g = ns.get("goal") or [float("nan")] * 2
            self.event(f"goal {seq} read back by the node: odom ({g[0]:.2f}, {g[1]:.2f})")
            return "done", f"the node read back goal {seq}"
        self.pending.append((rec, time.monotonic() + GOAL_READBACK_S, check, "goal not confirmed by the node"))
        return None

    def act_goal(self, rec):
        """`goal {ahead, left}`: relative to the robot now (the panel's map tap, the terminal's n)."""
        a, l_ = float(rec["args"]["ahead"]), float(rec["args"]["left"])
        if not (math.isfinite(a) and math.isfinite(l_)):
            return "refused", "non-finite goal"
        self.event(f"{rec['source']}: new goal {a:.1f} m ahead {l_:+.1f} m left")
        return self.send_goal(rec, self.next_seq(), {"kind": "rel", "rel": [a, l_], "source": rec["source"],
                                                     "operator": rec.get("operator")}, ahead=a, left=l_)

    def act_goal_odom(self, rec):
        """`goal_odom {x, y, seq, check_id}`: a goal already converted to this node's odometry frame (the hub's
        send_goal after its start check). Optional context stored with the next attempt: start_check (the check in
        force), rehearsal (the rehearsal set: names + verdict), goal_scene, scan."""
        args = rec["args"]
        x, y = float(args["x"]), float(args["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            return "refused", "non-finite goal"
        if args.get("seq") in (None, "") or args.get("check_id") in (None, ""):
            return "refused", "goal_odom needs seq and check_id"
        ctx = {"kind": "odom", "odom": [x, y], "check_id": args["check_id"], "source": rec["source"],
               "operator": rec.get("operator"), "start_check": args.get("start_check") or args.get("check"),
               "rehearsal": args.get("rehearsal") or args.get("rehearsals"), "goal_scene": args.get("goal_scene"),
               "scan": args.get("scan"), "passage": args.get("passage")}
        self.event(f"{rec['source']}: goal odom ({x:.2f}, {y:.2f}), check {args['check_id']}")
        return self.send_goal(rec, args["seq"], ctx, frame="odom", x=x, y=y)

    def goal_lock(self):
        """Why the goal in force does not count for a GO now (None: it does). A goal counts once the node read it
        back, after the start of this cycle (cycle), while the node still has it; a robot's launch goal never."""
        p = self.pending_goal()
        if p is not None:
            return "the node has not read back the goal just sent yet"
        t, why = self.cycle()
        ns = self.node_status
        if self.goal_t is not None:
            if t is not None and self.goal_t < t:
                return f"no goal sent since {why}: send the goal again (one goal per attempt, after Clear map)"
            if ns.get("goal_seq") != self.goal_seq or (
                    self.goal_xy is not None and ns.get("goal") is not None and math.dist(ns["goal"], self.goal_xy) > 0.01):
                return "the node's goal changed outside the console: send the goal again"
            return None
        if self.start_goal_counts and not self.seqs and not self.attempts.rows and self.cleared_t is None:
            return None  # a sim console's launch goal, first cycle only
        if self.args.mode == "robot" and not self.seqs:
            return ("no goal sent: send it (Studio's Send goal after the start check, or the panel's Set goal); "
                    "the launch goal never counts on the robot")
        return f"no goal confirmed by the node since {why}: send a goal" if why else "no goal confirmed: send a goal"

    def cycle(self):
        """(session time this GO cycle began, what began it): max(CONTROL, the last attempt's end, the last Clear
        map). A goal counts only when the node read it back after that."""
        cands = [(self.control_t, "Start SONIC"), (self.cleared_t, "Clear map")]
        ended = [r for r in self.attempts.rows if r.get("t_end") is not None]
        if ended:
            last = max(ended, key=lambda r: r["t_end"])
            cands.append((last["t_end"], f"attempt {last['n']} ended"))
        cands = [c for c in cands if c[0] is not None]
        return max(cands, key=lambda c: c[0]) if cands else (None, None)

    def scan_left(self):
        if self.cleared_t is None:
            return 0.0
        return max(0.0, CLEAR_MAP_SCAN_S - (time.monotonic() - self.t0 - self.cleared_t))

    # -- the GO gate --------------------------------------------------------------------------------------------
    def go_locks(self, source=None):
        """Why GO would be refused now, in the order to fix them (empty: it would be accepted). With
        source "panel" the Studio lease counts too."""
        st, a = self.runner_state, self.args
        out = []
        if "runner" not in self.procs:
            out.append("the runner is not started (Arm first)")
        elif st["estop"]:
            out.append("the runner is in E-STOP: end the session")
        elif not st["control"]:
            out.append("Start SONIC first (the runner is not in CONTROL)")
        elif self.go_settle_left() > 0:
            # a GO right after Start SONIC walks off a stand SONIC has not settled (sim, 2026-10-27: GO 0.2 s after
            # CONTROL, the virtual gantry let go mid-step and the robot fell)
            out.append(f"SONIC is settling the stand ({self.go_settle_left():.1f} s left)")
        if a.mode == "robot" and rehearsal_running():
            # a rehearsal beside the live robot shares the GPU and CPU: the robot walks only when it has ended
            out.append("a rehearsal is running on this PC (about a minute); GO when it has ended")
        if not a.replay:
            ns = self.node_status
            if not self.node_ready():
                out.append("the student node is still starting (planner warm-up)")
            elif ns.get("phase") in ("PRESCAN", "RUN"):
                out.append("the student is already walking")
            if (ns.get("odom") or {}).get("lost"):
                out.append(f"odometry lost ({ns['odom']['lost']}): no walk on legs alone; end the session")
            why = self.goal_lock()
            if why:
                out.append(why)
            left = self.scan_left()
            if left > 0:
                out.append(f"Clear map: the node is still scanning ({left:.0f} s left)")
        if source == "panel":
            out += self.lease_locks()
        return out

    def lease_active(self):
        ls = self.lease
        return ls if ls and ls.get("holder") and time.time() < ls["until"] else None

    def lease_locks(self):
        ls = self.lease_active()
        return [f"Studio holds the operator lease ({ls['holder']}): GO from Studio, or release the lease there"] if ls else []

    def act_lease(self, rec):
        """`lease {holder, until | ttl_s}` from Studio's gateway (through the hub): while held, GO from the phone panel
        is refused. until: unix seconds (a value below 1e9 is read as this PC's monotonic clock); holder null: released."""
        if rec["source"] not in ("studio", "hub"):
            return "refused", "the operator lease comes from Studio's gateway (through the hub)"
        args = rec["args"]
        if not args.get("holder"):
            self.lease = None
            return "done", "lease released"
        until = float(args["until"]) if args.get("until") is not None else time.time() + float(args["ttl_s"])
        if until < 1e9:
            until = time.time() + (until - time.monotonic())
        self.lease = {"holder": str(args["holder"]), "until": until, "lease_id": args.get("lease_id")}
        return "done", f"lease held by {args['holder']} for {until - time.time():.1f} s"

    # -- the rest -----------------------------------------------------------------------------------------------
    def act_arm(self, rec):
        if "runner" in self.procs:
            return "done", "already armed"
        self.event(f"{rec['source']}: ARM, starting the runner")
        self.start_runner()
        return "done", "runner starting (stand ramp about 25 s)"

    def act_quit(self, rec):
        self.event(f"{rec['source']}: end session")
        self.quit_requested = True
        return "done", "ending the session"

    def act_tag(self, rec):
        args = rec["args"]
        ok = self.attempts.tag(int(args.get("n", 0)), args.get("tag"), args.get("note"))
        return ("done", None) if ok else ("refused", "no such attempt, or an unknown tag")

    def runner_label(self):
        st = self.runner_state
        return ("not started" if "runner" not in self.procs else "E-STOP" if st["estop"] else "CONTROL"
                if st["control"] else "INIT done" if st["init"] else "booting")

    def estop_confirmed(self, wait_s=0.0):
        """The runner's log shows its EMERGENCY STOP (looked for up to wait_s)."""
        r = self.procs.get("runner")
        end = time.monotonic() + wait_s
        while r is not None:
            if self.estop_logged:
                return True
            with r.lines_lock:
                tail = r.lines[-80:]
            if any(any(s_ in line for s_ in ESTOP_LINES) for _, line in tail):
                return True
            if time.monotonic() >= end:
                break
            time.sleep(0.01)
        return False

    def estop(self, source, operator=None, act_id=None, kind=None):
        """Runner damping now: its global `o` works in every input interface (keyboard, gamepad, ZMQ stream). Safe from
        any thread (the panel's HTTP thread, the remote watcher's pump); the node is told to quit by the main loop.
        Returns the act reply {ok, id, result, reason, key_sent, runner_alive, runner, confirmed, t_mono}: ok = the
        console took it; key_sent = a live runner got its `o`; confirmed = the runner's log shows EMERGENCY STOP
        (looked for ESTOP_CONFIRM_S)."""
        kind = kind or (source.split()[0] if source.split() and source.split()[0] in ACT_SOURCES else "key")
        rec = self.make_act("estop", kind, operator, act_id, detail=source)
        r = self.procs.get("runner")
        alive = r is not None and r.alive()
        if not alive:
            self.event(f"E-STOP ({source}): no runner running, nothing holds the motors")
            reason = "no runner running: nothing this console started holds the motors"
        else:
            r.key("o")
            first = not self.runner_state["estop"]
            self.runner_state["estop"] = True
            self.event(f"E-STOP ({source}): runner damping")
            if first and kind != "key":
                self.inbox.put(("node_quit", None, source))
            reason = "runner damping"
        confirmed = alive and self.estop_confirmed(ESTOP_CONFIRM_S)
        reply = {"ok": True, "id": rec["id"], "act": "estop", "result": "done", "reason": reason, "key_sent": alive,
                 "runner_alive": r is not None and r.alive(), "runner": self.runner_label(), "confirmed": confirmed,
                 "t_mono": time.monotonic()}
        rec["result"], rec["state"], rec["reason"] = "done", "done", reason + ("; runner confirmed" if confirmed else "")
        self.log_act(rec, "done", rec["reason"])
        rec["done"].set()
        reply["session"] = self.dir.name
        return reply

    def on_remote(self, line):
        """remote_watch.py output (pump thread): Select stops at once; D-up queues a takeover while streaming.
        The runner reads every D-pad press itself and leaves the student stream for that motion set (RUNNER_SETS):
        D-down / left / right also stand the student and raise an alarm (2026-10-27: an L2 + D-left meant as a stop put
        the robot in the boxing set, then L2 = a left hook). Every press is logged."""
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            return
        if m.get("remote") == "status":
            if bool(m.get("connected")) != self.remote["connected"]:
                self.event("remote: " + ("connected" if m.get("connected") else "NOT SEEN (no remote bytes in lowstate)"))
            self.remote.update(connected=bool(m.get("connected")), samples=m.get("samples", 0))
            if m.get("temps"):
                self.temps = m["temps"]
        elif m.get("remote") == "press":
            b = m.get("button")
            self.remote["last"] = b
            st = self.runner_state
            streaming = st["iface"] == "ZMQ"
            if b in RUNNER_SETS:
                st["set"] = b
            if b == "Select":
                self.estop("remote Select")
            elif b == "Up" and streaming:  # in gamepad mode the runner reads D-up itself
                self.inbox.put(("key", "t", "remote D-up"))
            elif b in RUNNER_SETS and b != "Up" and streaming:
                self.event(f"REMOTE D-{b.lower()}: the runner left the student stream for its {RUNNER_SETS[b]} motion set; "
                           "D-up = standing set, Select = E-STOP")
                self.inbox.put(("key", "h", f"remote D-{b.lower()}"))
            elif st["iface"] == "GAMEPAD" and st.get("set") == "Left" and b in BOXING_KEYS:
                self.event(f"REMOTE {b} in the boxing set: {BOXING_KEYS[b]}; D-up = standing set")
            else:
                self.event(f"remote: {b}")

    def driving(self):
        """(what drives the robot, its source key, the runner's motion set): the panel's text and Studio's HUD."""
        st = self.runner_state
        if "runner" not in self.procs:
            return "Unitree controller", "unitree", None
        if st["iface"] == "ZMQ" and st["stream"]:
            return ("recorded reference" if self.args.replay else "student stream"), (
                "replay" if self.args.replay else "student"), None
        if st["iface"] == "GAMEPAD" and st["control"]:
            mset = RUNNER_SETS.get(st.get("set")) if st.get("set") else None
            return "SONIC planner" + (f" ({mset} set)" if mset else ""), "sonic_planner", mset
        return "SONIC reference", "sonic_reference", None

    def panel_state(self):
        """The panel's /state (strict JSON through workspace.dumps), also what the hub and Studio read: the runner, the
        node, the workspace, plus the act and GO-gate fields of CONTRACT §4.1 (go_locks is empty iff GO from Studio or
        the hub would be accepted now; go_locks_panel adds the Studio lease for the phone panel's GO)."""
        st, ns = self.runner_state, self.node_status
        now = time.monotonic()
        src, src_key, mset = self.driving()
        locks = self.go_locks()
        with self.log_lock:
            acts = list(self.acts_recent)
        lease = self.lease_active()
        ctx = self.goal_ctx
        return {"t": now - self.t0, "console_t_mono": now, "session": self.dir.name,
                "mode": "ROBOT " + str(self.args.iface) if self.args.mode == "robot" else "SIMULATION",
                "runner": self.runner_label(), "estop_confirmed": bool(self.estop_logged), "driving": src,
                "driving_source": src_key, "motion_set": mset, "phase": ns.get("phase"),
                "goal_dist_m": ns.get("goal_dist_m"), "latency_s": ns.get("latency_s"),
                "pelvis_z_m": ns.get("pelvis_z_m"), "tilt_rad": ns.get("tilt_rad"), "cloud_age_s": ns.get("cloud_age_s"),
                "xy": ns.get("xy"), "odom": ns.get("odom"),
                "standing_s": None if self.control_t is None or st["estop"] else round(now - self.t0 - self.control_t, 1),
                "go_locks": locks, "go_locks_panel": locks + self.lease_locks(),
                "lease": None if lease is None else {"holder": lease["holder"], "until": lease["until"],
                                                     "left_s": round(lease["until"] - time.time(), 1)},
                "goal": {"seq": self.goal_seq, "ok": self.goal_lock() is None, "kind": ctx.get("kind", "launch"),
                         "rel": list(self.goal_rel) if self.goal_rel else None, "odom": ctx.get("odom"),
                         "node": ns.get("goal"), "check_id": ctx.get("check_id"), "source": ctx.get("source"),
                         "confirmed_t": None if self.goal_t is None else round(self.goal_t, 2)},
                "goal_seq": self.goal_seq, "clear_map_scan_left_s": round(self.scan_left(), 1), "acts": acts,
                "operator": getattr(self.args, "operator", None),
                "record": self.record_line().strip(), "events": [f"{t:6.1f}  {e}" for t, e in self.events[-12:]][::-1],
                "write_errors": {k: {"count": n, "last": e} for k, (n, e) in self.write_errors.items()},
                "dir": str(self.dir), "remote": dict(self.remote), "node_ready": self.node_ready(),
                "procs": {k: p.alive() for k, p in self.procs.items()}, **self.workspace(locks)}

    def workspace_ctx(self):
        t_map, m = self.map_cache
        prof = getattr(self.args, "profile", None)
        return {"mode": self.args.mode, "t": time.monotonic() - self.t0, "runner_state": dict(self.runner_state),
                "node_status": self.node_status, "node_ready": self.node_ready(), "rec_stats": self.rec_stats,
                "procs": {k: p.alive() for k, p in self.procs.items()}, "remote": dict(self.remote),
                "temps": self.temps, "machine_busy": (getattr(self.args, "machine", None) or {}).get("busy") or [],
                "near_body": workspace.near_body(m) if m is not None and time.monotonic() - t_map < 5.0 else None,
                "attempts": self.attempts.public(), "cleared_t": self.cleared_t, "control_t": self.control_t,
                "goal_rel": self.goal_rel, "goal_t": self.goal_t, "goal_ok": self.goal_lock() is None, "profile": prof}

    def workspace(self, go_locks=None):
        """The panel's robot-day view: checks, ladder, attempts and what this session runs (g1deploy/workspace.py)."""
        ctx = self.workspace_ctx()
        ctx["checks"] = workspace.checks(ctx)
        ctx["go_locks"] = self.go_locks() if go_locks is None else go_locks
        a, prof = self.args, ctx["profile"]
        ns = self.node_status
        return {"checks": ctx["checks"], "ladder": workspace.ladder(ctx), "attempts": ctx["attempts"],
                "near_body": ctx["near_body"], "temps": self.temps, "goal_rel": self.goal_rel,
                "setup": {"profile": prof["name"] if prof else None, "identity": self.identity,
                          "bundle": Path(getattr(a, "bundle", None) or ROOT / "bundle").name,
                          "rules": Path(a.rules).name if getattr(a, "rules", None) else None, "guards": a.guards,
                          "lookahead_m": a.goal_lookahead, "rear_blind_deg": a.rear_blind_deg,
                          "rear_blind_range_m": a.rear_blind_range_m, "robot": getattr(a, "robot", None),
                          "odom": getattr(a, "odom", None),
                          "task": Path(self.task).parent.name if getattr(self, "task", None) else None},
                "node": {k: ns.get(k) for k in ("plans", "rule", "no_safe", "max_lag_s", "overruns", "tilt_estop",
                                                 "guards", "last_guard", "calm_hold", "stream_ahead", "standing")},
                "tags": list(TAGS), "go_settle_left_s": round(self.go_settle_left(), 1),
                "rehearsal_running": bool(self.args.mode == "robot" and rehearsal_running())}

    def node_map(self):
        """The node's map for the panel (StudentNode.map_snapshot), at most every 0.4 s whoever asks."""
        with self.map_lock:
            t, m = self.map_cache
            if time.monotonic() - t < 0.4 and m is not None:
                return m
            if not self.node_ready():
                return {"ok": False, "error": "the student node is not up yet (it starts on Arm)"}
            try:
                s = self.zmq.Context.instance().socket(self.zmq.REQ)
                s.setsockopt(self.zmq.LINGER, 0)
                s.connect(self.ctl)
                s.send_string(json.dumps({"cmd": "map"}))
                m = json.loads(s.recv()) if s.poll(800) else {"ok": False, "error": "the node did not answer"}
                s.close(0)
            except Exception as exc:
                m = {"ok": False, "error": str(exc)}
            if m.get("ok"):
                self.map_cache = (time.monotonic(), m)
            return m

    # -- screen ---------------------------------------------------------------------------------------------
    def record_line(self):
        s = self.rec_stats.get("streams", {})
        if not s:
            return " record : " + ("off" if "rec" not in self.procs else "starting ...")

        def hz(k):
            v = s.get(k, {})
            age = v.get("age_s")
            return "-" if not v.get("count") else f"{v.get('rate_hz') or 0:.0f}" if age is not None and age < 1.0 else "STALE"
        mb = sum((v.get("bytes") or 0) for v in s.values() if isinstance(v, dict)) / 1e6
        err = " ERR" if self.rec_stats.get("errors") else ""
        return (f" record : lowstate {hz('lowstate')} Hz  lowcmd {hz('lowcmd')}  lidar {hz('lidar')}  "
                f"pose {hz('zmq_pose')}  status {s.get('node_status', {}).get('count', 0)}  {mb:.0f} MB{err}")

    def screen(self, width=78):
        st, ns = self.runner_state, self.node_status
        t = time.monotonic() - self.t0
        mode = "SIMULATION (MuJoCo)" if self.args.mode == "sim" else f"ROBOT on {self.args.iface}"
        if self.args.replay:
            mode += "  [H4 replay]"
        runner = ("E-STOP" if st["estop"] else "CONTROL" if st["control"] else "INIT done" if st["init"] else "booting")
        src = "student stream" if (st["iface"] == "ZMQ" and st["stream"]) else (
            "SONIC planner" if st["iface"] == "GAMEPAD" and st["control"] else "SONIC reference")
        if src == "SONIC planner" and st.get("set"):
            src += f" ({RUNNER_SETS[st['set']]})"
        lat = ns.get("latency_s")
        rows = [
            f" G1 student deploy  |  {mode}  |  t = {t:6.1f} s",
            "-" * width,
            f" runner : {runner:<10s}  driving: {src:<16s}  procs: " + " ".join(
                f"{k}{'+' if p.alive() else '-'}" for k, p in self.procs.items()),
            f" student: {ns.get('phase', '-'):<9s} goal {ns.get('goal_dist_m', float('nan')):5.2f} m   plans {ns.get('plans', 0):3d}"
            f"   latency {lat if lat is None else round(lat, 2)} s   lidar age {ns.get('cloud_age_s', float('nan')):4.2f} s",
            f" robot  : pelvis z {ns.get('pelvis_z_m', float('nan')):4.2f} m   tilt {ns.get('tilt_rad', float('nan')):4.2f} rad"
            f"   xy {[round(v, 2) for v in ns.get('xy', [])]}",
            self.record_line()[:width],
            "-" * width,
            " [s]start SONIC [g]GO student [h]stand [t]takeover [n]goal [o]E-STOP [q]quit",
            "-" * width,
        ]
        rows += [f" {tt:6.1f}  {txt}"[:width] for tt, txt in self.events[-10:]]
        return "\n".join(rows)

    def record_frame(self, text):
        try:
            self.frames.write(json.dumps({"wall": time.monotonic(), "text": text}) + "\n")
        except (OSError, ValueError) as exc:
            self.write_failed("console_frames.jsonl", exc)

    def shutdown(self):
        self.attempts.close(time.monotonic() - self.t0, "session ended", self.node_status)
        if "node" in self.procs and not self.args.replay:
            self.node_cmd(cmd="quit")
            time.sleep(0.5)
        for name in ("node", "runner", "sim", "lio", "lidar", "livox", "remote", "rec"):  # the recorder last: it sees the end
            if name in self.procs:
                self.procs[name].stop(signal.SIGTERM if name in ("sim", "lio") else signal.SIGINT)
        self.frames.close()
        with self.log_lock:
            self.events_log.close()


# ------------------------------------------------------------------------------------------------- main
def run_console(args):
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))  # clean shutdown on TERM
    signal.signal(signal.SIGINT, signal.default_int_handler)  # background jobs inherit SIGINT ignored
    # a busy PC skews every real-time number (node lag, plan latency, falls): record it, and by policy stop
    isolated = args.mode == "sim" and (getattr(args, "port_base", None) or PORT_BASE) != PORT_BASE
    snap, busy, ok = machine.check(args.busy, args.busy_wait_s, log=lambda m: print(m, flush=True), beside_robot=isolated)
    args.machine = None if snap is None else dict(snap, busy=busy, policy=args.busy)
    if not ok:
        print(f"PC busy, not starting (--busy {args.busy}):\n  " + "\n  ".join(busy), file=sys.stderr)
        return 3
    sess = Session(args)
    if isolated:  # the robot session beside it refuses GO while this runs
        REHEARSAL_LOCK.write_text(f"{os.getpid()} {sess.dir}\n")
        if snap and snap.get("beside"):
            sess.event("rehearsal beside a live robot session: own ports " + str(sess.ports) + "; the robot's GO waits")
    for reason in busy:
        sess.event(f"PC BUSY at start ({reason}): real-time numbers of this session may be off")
    if args.profile:
        p = args.profile
        sess.event(f"profile {p['name']}" + (f" (command line overrides {', '.join(p['overridden'])})"
                                              if p["overridden"] else ""))
    interactive = sys.stdin.isatty() and not args.script
    if args.mode == "robot":
        print("ROBOT MODE. The robot must hang in its gantry with the feet on the floor, the remote in hand,\n"
              "the physical E-stop within reach and nobody inside the robot's reach. Session: " + str(sess.dir))
        if interactive and input("Type 'yes' to start the runner on the robot: ").strip() != "yes":
            return 1
    sess.start()
    if args.panel:
        from g1deploy.panel import start_panel

        try:  # the deploy PC's one panel token (G1_PANEL_TOKEN, else ~/.config/g1-deploy/panel_token): a restarted
            token = service_token.panel_token()  # hub and Studio's gateway reach this console with it
        except OSError:
            token = None  # a random one; panel_url.txt has it
        _, token = start_panel(sess, args.panel, token=token)
        sess.panel_url = f"http://127.0.0.1:{args.panel}/?t={token}"
        (sess.dir / "panel_url.txt").write_text(sess.panel_url + "\n")
        sess.event(f"panel: port {args.panel} (tailnet / this PC only), token in panel_url.txt")
    steps = [s.strip() for s in args.script.split(",") if s.strip()] if args.script else []
    old = termios.tcgetattr(sys.stdin) if interactive else None
    if interactive:
        tty.setcbreak(sys.stdin.fileno())
    step_t, last_draw, result, script_goal = time.monotonic(), 0.0, 0, None
    try:
        while True:
            sess.poll()
            now = time.monotonic()
            if now - last_draw > 0.2:
                text = sess.screen()
                sess.record_frame(text)
                if interactive:
                    sys.stdout.write("\033[H\033[2J" + text + "\n")
                    sys.stdout.flush()
                last_draw = now
            if ("runner" in sess.procs and not sess.procs["runner"].alive() and not steps
                    and not getattr(sess, "_runner_gone", False)):
                sess.event("runner exited")
                sess._runner_gone = True
            while True:
                try:
                    kind, val, src = sess.inbox.get_nowait()
                except queue.Empty:
                    break
                if kind == "act":  # POST /act (panel page, hub, Studio): its HTTP thread waits for the result
                    sess.run_act(val)
                elif kind == "key":  # the remote watcher (D-up = takeover, D-down/left/right = stand)
                    sess.event(f"{src}: [{val}]")
                    sess.act(val, source="remote" if str(src).startswith("remote") else "key", detail=src)
                elif kind == "node_quit" and not args.replay and "node" in sess.procs:
                    sess.node_cmd(cmd="quit")
            if sess.quit_requested:
                break
            if steps:
                kind, _, val = steps[0].partition(":")
                done = False
                if kind == "sleep":
                    done = now - step_t >= float(val)
                elif kind == "key":
                    sess.act(val, source="script")
                    done = True
                elif kind == "goal":  # done once the node has read it back (or it was refused)
                    if script_goal is None:
                        a_, l_ = map(float, val.split(":"))
                        script_goal = sess.act("goal", source="script", ahead=a_, left=l_)
                    done = script_goal["done"].is_set()
                    if done:
                        script_goal = None
                elif kind == "wait":
                    ns, st = sess.node_status, sess.runner_state
                    done = {"init": st["init"], "control": st["control"], "stream": st["stream"],
                            "run": ns.get("phase") == "RUN", "arrived": ns.get("phase") == "ARRIVED",
                            "standing": bool(ns.get("standing")), "ready": ns.get("phase") == "WAIT"}.get(val, False)
                    if now - step_t > args.step_timeout or (sess.runner_state["estop"] and val != "init"):
                        sess.event(f"script: stopped waiting for {val}"
                                   + (" (e-stop)" if sess.runner_state["estop"] else " (timeout)"))
                        result = 2
                        steps = ["sleep:3"]  # keep recording a few seconds, then end
                        step_t = now
                        continue
                if done:
                    steps.pop(0)
                    step_t = now
                    if not steps:
                        sess.event("script: done")
                        break
            if interactive and select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                if ch == "q":
                    if args.mode == "robot" and sess.runner_state["control"]:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
                        ok = input("\nQuitting stops the runner. Robot in the gantry / sitting? type 'yes': ").strip() == "yes"
                        tty.setcbreak(sys.stdin.fileno())
                        if not ok:
                            continue
                    break
                if ch == "n":
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
                    try:
                        a_, l_ = map(float, input("\nnew goal 'ahead left' (m): ").split())
                        sess.act("goal", ahead=a_, left=l_)
                    except ValueError:
                        sess.event("goal: expected two numbers")
                    tty.setcbreak(sys.stdin.fileno())
                elif ch == "\x1b":
                    sess.act("o")
                elif ch in "sghtcor?":
                    sess.act(ch)
            elif not interactive:
                time.sleep(0.05)
    except KeyboardInterrupt:
        sess.event("console interrupted")
    finally:
        if interactive:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        sess.shutdown()
        if isolated and rehearsal_running() == os.getpid():
            REHEARSAL_LOCK.unlink(missing_ok=True)
        summary = {"dir": str(sess.dir), "events": [f"{t:.1f} {e}" for t, e in sess.events], "status": sess.node_status}
        (sess.dir / "console_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
        print(f"\nsession saved: {sess.dir}")
        if args.report:
            subprocess.call([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/session_report.py"), str(sess.dir)],
                            env=sess.env, cwd=str(ROOT))
    return result


def apply_robot_profile(args):
    """robots/<name>.json supplies this robot's defaults (iface, lidar, body, mount calibration, odometry);
    anything given on the command line wins."""
    if not args.robot:
        return args
    path = ROBOTS / f"{args.robot}.json"
    if not path.exists():
        raise SystemExit(f"no robot profile {path} (see robots/README.md)")
    prof = json.loads(path.read_text())
    for key in ("iface", "lidar", "body", "mount_calib", "odom", "rear_blind_deg", "rear_blind_range_m"):
        if getattr(args, key, None) is None and prof.get(key) is not None:
            val = prof[key]
            if key == "mount_calib" and not Path(val).is_absolute():
                val = str(ROOT / val)
            setattr(args, key, val)
    return args


TOOLS = {"check": "tools/check_robot.py", "calibrate": "tools/calibrate_mount.py", "bench": "tools/bench_planner.py",
         "replay": "tools/stream_replay.py", "rehearse": "sim/realtime/rehearse.py", "report": "tools/session_report.py",
         "discover": "tools/dds_discover.py", "remote": "tools/remote_check.py", "site": "tools/site_check.py"}
KIT = ("doctor", "sessions", "note", "courses", "lidar", "lio")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in TOOLS:
        return subprocess.call([str(ROOT / ".venv/bin/python"), str(ROOT / TOOLS[argv[0]]), *argv[1:]], env=_env(),
                               cwd=str(ROOT))
    if argv and argv[0] in KIT:
        return subprocess.call([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/deploy_kit.py"), *argv], env=_env(),
                               cwd=str(ROOT))
    if argv and argv[0] == "lab":  # the robot day's home page (g1deploy/hub.py)
        from g1deploy import hub

        return hub.main(argv[1:])
    if argv and argv[0] == "test":
        return subprocess.call([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", *argv[1:]], env=_env(), cwd=str(ROOT))
    return run_console(parse_args(argv))


def parse_args(argv):
    """`g1 sim|robot` options, with the deploy profile, the robot profile and the defaults applied (in that order of
    precedence after the command line)."""
    ap = argparse.ArgumentParser(prog="g1", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("sim", "robot"):
        p = sub.add_parser(mode)
        p.add_argument("--goal", type=float, nargs=2, default=None, metavar=("AHEAD", "LEFT"),
                       help="launch goal from the start pose (sim: default the course's goal; robot: required)")
        p.add_argument("--goal-lookahead", type=float, default=None, help="1.0 = slower walking (G4: no cost)")
        p.add_argument("--candidates", type=int, default=None, help="planner K (default 16)")
        p.add_argument("--ode-steps", type=int, default=None, help="planner ODE steps (default 10)")
        p.add_argument("--profile", default=None, help="deploy profile: a name under profiles/ (or private/profiles/) "
                                                       "or a .json path; its options fill what the command line leaves "
                                                       "unset (profiles/README.md)")
        p.add_argument("--busy", choices=("warn", "wait", "refuse", "off"), default="warn",
                       help="when the PC is busy (load, GPU, heavy jobs) before the run: warn and go on (default), "
                            "wait up to --busy-wait-s, refuse, or skip the check. Another deploy chain running "
                            "always refuses (g1deploy/machine.py)")
        p.add_argument("--busy-wait-s", type=float, default=300.0)
        p.add_argument("--rules-overrides", type=json.loads, default=None, metavar="JSON",
                       help="settings for the bundle's deploy rules plugin, as JSON (e.g. '{\"route\": {...}}') "
                            "(the plugin decides what it honours; recorded in the session and its identity)")
        p.add_argument("--port-base", type=int, default=None,
                       help=f"the chain's local ports from this base (default {PORT_BASE}: stream, +1 runner debug, +4 "
                            "cloud, +5 LIO odometry, +6 node control). A sim with another base is an isolated rehearsal "
                            "that may run beside a robot session; that robot's GO waits until it ends")
        p.add_argument("--planner-impl", choices=("fused", "graph", "eager"), default=None,
                       help="planner implementation (fused = fast, same plans; eager = as evaluated)")
        p.add_argument("--planner-precision", choices=("fp32", "tf32", "bf16"), default=None,
                       help="denoiser precision (fast planner): fp32 = the evaluated arithmetic; tf32 / bf16 = tensor cores (docs/PLANNER_SPEED.md)")
        p.add_argument("--body", choices=("dex3", "rubber"), default=None, help="hands (self filter; default dex3)")
        p.add_argument("--script", default="")
        p.add_argument("--step-timeout", type=float, default=90.0)
        p.add_argument("--record", default=None, help=f"session directory (default {SESSIONS}/<mode>-<time>)")
        p.add_argument("--replay", default=None, help="H4: stream this recorded episode (closed_loop.npz dir) on [g] "
                                                       "instead of running the planner")
        p.add_argument("--replay-slow", type=float, default=2.0, help="H4 replay time scale (2 = half speed)")
        p.add_argument("--no-record", dest="record_data", action="store_false", help="do not run the data recorder")
        p.add_argument("--no-record-lidar", dest="record_lidar", action="store_false",
                       help="record everything but the raw lidar frames (~1.5 MB/s)")
        p.add_argument("--no-report", dest="report", action="store_false")
        p.add_argument("--rear-blind-deg", type=float, default=None,
                       help="ignore this rear sector of the lidar (180 = back half: spotter, rope, gantry); robot "
                            "profile key rear_blind_deg")
        p.add_argument("--rear-blind-range-m", type=float, default=None,
                       help="with --rear-blind-deg: blank only rear returns closer than this (about 1 m: rope, spotter); "
                            "robot profile key rear_blind_range_m")
        p.add_argument("--arm", action="store_true", help="start the runner only when the panel's Arm is pressed "
                                                            "(recorder, lidar, remote watcher and panel come up first)")
        p.add_argument("--panel", type=int, default=8770 if mode == "robot" else 0, metavar="PORT",
                       help="operator web panel on 127.0.0.1:PORT (E-STOP, start, GO, stand, takeover, goal; "
                            "robot default 8770, 0 = off): g1deploy/panel.py")
        p.add_argument("--debug-plans", action="store_true", default=None,
                       help="node writes every plan's exact inputs and result to <session>/plans (tools/debug_plan.py)")
        p.add_argument("--guards", choices=("off", "deploy", "walls"), default=None,
                       help="deploy-side guards (docs/ROBUSTNESS.md): deploy = self mask, split guidance boxes, walk "
                            "envelope, calm fallback, anchor check; walls = deploy + low overheads as walls (H5-H6)")
        p.add_argument("--self-margin", type=float, default=None, help="self-filter margin override (m)")
        p.add_argument("--memory-clear", action="store_true", default=None,
                       help="free-space clearing in the lidar memory: rays through old obstacle evidence clear it "
                            "(repairs the smear of an odometry error; off by default, the student trained without it)")
        p.add_argument("--bundle", default=None, help="planner bundle directory (default bundle/)")
        p.add_argument("--rules", default=None, help="the bundle's deploy rules plugin (a .py file; needed by bundles "
                                                     "whose student config is not the deployed format)")
        p.add_argument("--operator", default=os.environ.get("G1_OPERATOR"), help="who runs the session (metadata)")
        p.add_argument("--confirm", type=json.loads, default=None, metavar="JSON",
                       help="the operator's start confirmations, stored in session.json (the hub's checklist, e.g. "
                            "'{\"gantry\": true, \"estop\": true, \"remote\": true, \"rope\": true}')")
        p.add_argument("--note", default=None, help="free text stored with the session (course, conditions, ...)")
        if mode == "sim":
            p.add_argument("--layout", default="open-3m", help=f"a course under {COURSES} or a layout under {PANELS}")
            p.add_argument("--task", default=None)
            p.add_argument("--mission", default=None, help="'A,L;A,L': after the launch goal, walk to each "
                                                           "(relative to the robot then); builds --script. "
                                                           "'course' = the course's own mission")
            p.add_argument("--odom", choices=("dds", "leg", "lio", "fused"), default="dds",
                           help="dds: simulator truth; leg: leg odometry (robot day 1); lio: LIO stand-in on :5561; "
                                "fused: the stand-in + leg odometry in one filter (node --odom fused)")
            p.add_argument("--lio-noise", default=None, help="LIO stand-in drift, e.g. '{\"xy_rw_m\": 0.02}', or an odometry error '{\"scale\": 0.9, \"rotate_deg\": -4}' (tools/sim_lio_bridge.py)")
            p.add_argument("--remote-start", action="store_true", help="press Start on the virtual remote for [s]")
            p.add_argument("--require-sent-goal", action="store_true",
                           help="GO needs a goal sent and read back in this cycle, as on the robot (practice sessions); "
                                "by default a sim console counts its launch goal for the first GO")
            p.add_argument("--viewer", action="store_true", help="open the MuJoCo 3D window (needs a display)")
            p.add_argument("--runner-csv", dest="runner_csv", action="store_true", default=False,
                           help="also write the runner's per-tick CSVs (on by default on the robot)")
        else:
            p.add_argument("--robot", default=None, help="robots/<NAME>.json: this robot's iface, lidar, body, "
                                                         "mount calibration and odometry defaults")
            p.add_argument("--iface", default=None)
            p.add_argument("--odom", choices=("leg", "zmq", "fused"), default=None,
                           help="leg (default) | zmq (LIO bridge) | fused (LIO bridge + leg odometry, g1deploy/odom_fusion.py)")
            p.add_argument("--lidar", choices=("dds", "livox"), default=None,
                           help="dds: Unitree's rt/utlidar service; livox: livox_ros_driver2 on this PC + adapter")
            p.add_argument("--mount-calib", default=None)
            p.add_argument("--no-runner-csv", dest="runner_csv", action="store_false", default=True)
    args = ap.parse_args(argv)
    try:  # command line > deploy profile > robot profile > the console's defaults
        args = profiles.apply(args, args.profile)
    except profiles.ProfileError as e:
        raise SystemExit(str(e))
    if args.mode == "robot":
        args = apply_robot_profile(args)
    args = profiles.fill_defaults(args)
    if args.mode == "robot":
        args.remote_start = False
        args.odom = args.odom or "leg"
        args.lidar = args.lidar or "dds"
        if not args.iface:
            raise SystemExit("--iface is required (or --robot NAME with an iface in its profile)")
        if {"scale", "rotate_deg"} & set(json.loads(os.environ.get("G1_LEG_ODOM") or "{}")):
            raise SystemExit("G1_LEG_ODOM scale / rotate_deg inject an odometry error: rehearsals only, never on the robot")
        if args.goal is None:
            raise SystemExit("--goal AHEAD LEFT is required on the robot (metres from the start pose; leg odometry "
                             "can overshoot ~25 %: keep that much clear floor beyond the goal)")
    else:
        args.robot, args.lidar = None, None
        args.task = args.task or resolve_task(args.layout)
        if args.goal is None:  # the course's goal, in the start frame (courses start at the origin facing +x)
            args.goal = json.loads(Path(args.task).read_text()).get("goal_world_xy", [3.5, 0.0])
        if args.mission == "course":
            args.mission = json.loads(Path(args.task).read_text()).get("mission")
            if not args.mission:
                raise SystemExit(f"{args.task} has no 'mission'")
        if args.mission:
            if args.script:
                raise SystemExit("--mission builds the script itself; give one or the other")
            args.script = mission_script(args.mission)
    args.body = args.body or "dex3"
    return args


if __name__ == "__main__":
    sys.exit(main())
