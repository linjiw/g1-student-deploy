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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import machine, profiles, workspace  # noqa: E402
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
        self.lines = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for raw in iter(self.p.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip("\n")
            t = time.monotonic()
            self.f.write(f"{t:.3f} {line}\n")
            self.lines.append((t, line))
            if self.on_line is not None:
                try:
                    self.on_line(line)
                except Exception as exc:  # a bad line must not stop the pump (it also feeds the log)
                    self.f.write(f"{t:.3f} [console] on_line error: {exc!r}\n")
            if len(self.lines) > 4000:
                del self.lines[:2000]

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
        self.node_status, self.node_events = {}, []
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
        self.attempts = Attempts(self.dir / "attempts.jsonl", prof["name"] if prof else None)
        self.temps, self.map_lock, self.map_cache = None, threading.Lock(), (0.0, None)
        self.goal_rel, self.goal_t = (tuple(args.goal) if getattr(args, "goal", None) else None), 0.0
        self.cleared_t = self.control_t = None
        try:
            self.identity = profiles.identity(args).get("short")
        except Exception:
            self.identity = None

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
                "--zmq", f"odom=odom=tcp://127.0.0.1:{P['odom']}", "--status", f"tcp://127.0.0.1:{P['control']}"]
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
                    rear_blind_range_m=a.rear_blind_range_m)
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
                elif "EMERGENCY STOP" in line or "Emergency Stop" in line:
                    st["estop"] = True
        n = self.procs.get("node")
        if n and not self.args.replay:
            seen = len(self.node_events)
            evs = [json.loads(l) for _, l in n.lines if l.startswith("{\"event\"")]
            for e in evs[seen:]:
                if e["event"] not in ("stall_risk", "dq_clipped"):
                    self.event("node: " + e["event"] + ("" if "reason" not in e else f" ({e['reason']})"))
                if e["event"] == "estop" and not self.runner_state["estop"]:
                    # the node's supervisor asked for a stop (tilt / stale state): damping now, no operator needed
                    self.procs["runner"].key("o")
                    self.runner_state["estop"] = True
                    self.event("AUTO E-STOP: runner damping (node supervisor)")
            self.node_events = evs
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

    # -- operator actions ---------------------------------------------------------------------------------
    def act(self, key):
        if "runner" not in self.procs:
            self.event(f"[{key}] ignored: the runner is not started (ARM first)")
            return
        r, st = self.procs["runner"], self.runner_state
        if key == "s":
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
        elif key == "g":
            if not st["control"] or st["estop"]:
                # the stream must never be on before SONIC controls the robot: SONIC would take control tracking
                # plans anchored on the hanging robot (seen in simulation 2026-10-24: the robot ended crouched)
                self.event("GO refused: Start SONIC first (runner not in CONTROL)" if not st["estop"]
                           else "GO refused: runner is in E-STOP")
                return
            settle = self.go_settle_left()
            if settle > 0:
                # a GO right after Start SONIC walks off a stand SONIC has not settled (sim, 2026-10-27: GO 0.2 s
                # after CONTROL, the virtual gantry let go mid-step and the robot fell)
                self.event(f"GO refused: SONIC is settling the stand ({settle:.1f} s left)")
                return
            if self.args.mode == "robot" and rehearsal_running():
                # a rehearsal beside the live robot shares the GPU and CPU: the robot walks only when it has ended
                self.event("GO refused: a rehearsal is running on this PC (about a minute); GO when it has ended")
                return
            if st["iface"] != "ZMQ":
                r.key("#")
                time.sleep(0.2)
            if not st["stream"]:
                r.key("\n")  # ZMQ streaming on
            if not self.args.replay and not self.node_ready():
                self.event("GO refused: the student node is still starting (planner warm-up); wait for 'node: node_up'")
                return
            if self.args.replay:
                if "node" not in self.procs:
                    py = str(ROOT / ".venv/bin/python")
                    self.procs["node"] = Proc("node", [py, "-u", "tools/stream_replay.py", self.args.replay,
                                                       "--slow", str(self.args.replay_slow),
                                                       "--bind", f"tcp://127.0.0.1:{self.ports['stream']}"],
                                              self.dir / "node.log",
                                              self.env)
                    self.event("operator: GO (stream on, recorded reference plays)")
                return
            if self.node_cmd(cmd="go"):
                self.event("operator: GO (stream on, student plans)")
                self.attempts.start(time.monotonic() - self.t0, self.node_status, goal_rel=self.goal_rel)
            else:
                self.event("GO NOT DELIVERED: the node did not answer (stream is on, the student is not planning)")
        elif key == "h":
            if not self.args.replay:
                self.node_cmd(cmd="stand")
                self.event("operator: student stand")
                self.attempts.close(time.monotonic() - self.t0, "operator stand", self.node_status)
        elif key == "t":
            r.key("@")  # SONIC's own planner (safety reset; streaming off)
            if self.args.replay and "node" in self.procs:
                self.procs["node"].stop()
            elif not self.args.replay:
                self.node_cmd(cmd="stand")
            self.event("operator: TAKEOVER -> SONIC planner")
            self.attempts.close(time.monotonic() - self.t0, "takeover", self.node_status)
        elif key == "o":
            self.estop("operator key")
            if not self.args.replay:
                self.node_cmd(cmd="quit")
        elif key == "c":
            if not self.args.replay:
                if self.node_cmd(cmd="clear_map"):
                    self.cleared_t = time.monotonic() - self.t0
                self.event("operator: CLEAR MAP (the node maps again from here; walks wait 10 s)")
        elif key == "r" and self.args.mode == "sim":
            (self.dir / "release.flag").touch()
            self.event("sim: gantry released")
        elif key == "?":
            for line in HELP.splitlines():
                self.event(line.strip())

    def estop(self, source):
        """Runner damping now: its global `o` works in every input interface (keyboard, gamepad, ZMQ stream).
        Safe from any thread; the node is told to quit by the main loop."""
        r = self.procs.get("runner")
        if r is None or not r.alive():
            self.event(f"E-STOP ({source}): no runner running, nothing holds the motors")
            return
        r.key("o")
        first = not self.runner_state["estop"]
        self.runner_state["estop"] = True
        self.event(f"E-STOP ({source}): runner damping")
        if first and source != "operator key":
            self.inbox.put(("node_quit", None, source))

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

    def panel_state(self):
        st, ns = self.runner_state, self.node_status
        runner = ("not started" if "runner" not in self.procs else "E-STOP" if st["estop"] else "CONTROL"
                  if st["control"] else "INIT done" if st["init"] else "booting")
        src = "Unitree controller" if "runner" not in self.procs else "student stream" if (
            st["iface"] == "ZMQ" and st["stream"]) else (
            "SONIC planner" if st["iface"] == "GAMEPAD" and st["control"] else "SONIC reference")
        if src == "SONIC planner" and st.get("set"):
            src += f" ({RUNNER_SETS[st['set']]} set)"
        return {"t": time.monotonic() - self.t0, "mode": "ROBOT " + str(self.args.iface) if self.args.mode == "robot"
                else "SIMULATION", "runner": runner, "driving": src, "phase": ns.get("phase"),
                "goal_dist_m": ns.get("goal_dist_m"), "latency_s": ns.get("latency_s"),
                "pelvis_z_m": ns.get("pelvis_z_m"), "tilt_rad": ns.get("tilt_rad"), "cloud_age_s": ns.get("cloud_age_s"),
                "xy": ns.get("xy"),
                "record": self.record_line().strip(), "events": [f"{t:6.1f}  {e}" for t, e in self.events[-12:]][::-1],
                "dir": str(self.dir), "remote": dict(self.remote), "node_ready": self.node_ready(),
                "procs": {k: p.alive() for k, p in self.procs.items()}, **self.workspace()}

    def workspace_ctx(self):
        t_map, m = self.map_cache
        prof = getattr(self.args, "profile", None)
        return {"mode": self.args.mode, "t": time.monotonic() - self.t0, "runner_state": dict(self.runner_state),
                "node_status": self.node_status, "node_ready": self.node_ready(), "rec_stats": self.rec_stats,
                "procs": {k: p.alive() for k, p in self.procs.items()}, "remote": dict(self.remote),
                "temps": self.temps, "machine_busy": (getattr(self.args, "machine", None) or {}).get("busy") or [],
                "near_body": workspace.near_body(m) if m is not None and time.monotonic() - t_map < 5.0 else None,
                "attempts": self.attempts.public(), "cleared_t": self.cleared_t, "control_t": self.control_t,
                "goal_rel": self.goal_rel, "goal_t": self.goal_t, "profile": prof}

    def workspace(self):
        """The panel's robot-day view: checks, ladder, attempts and what this session runs (g1deploy/workspace.py)."""
        ctx = self.workspace_ctx()
        ctx["checks"] = workspace.checks(ctx)
        a, prof = self.args, ctx["profile"]
        ns = self.node_status
        return {"checks": ctx["checks"], "ladder": workspace.ladder(ctx), "attempts": ctx["attempts"],
                "near_body": ctx["near_body"], "temps": self.temps, "goal_rel": self.goal_rel,
                "setup": {"profile": prof["name"] if prof else None, "identity": self.identity,
                          "bundle": Path(getattr(a, "bundle", None) or ROOT / "bundle").name,
                          "rules": Path(a.rules).name if getattr(a, "rules", None) else None, "guards": a.guards,
                          "lookahead_m": a.goal_lookahead, "rear_blind_deg": a.rear_blind_deg,
                          "rear_blind_range_m": a.rear_blind_range_m, "robot": getattr(a, "robot", None),
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

    def new_goal(self, ahead, left):
        if self.args.replay:
            self.event("goal: no planner in replay mode")
            return
        if not self.node_ready():
            self.event("goal refused: the student node is still starting (planner warm-up)")
            return
        if self.node_cmd(cmd="goal", ahead=ahead, left=left):
            self.goal_rel, self.goal_t = (ahead, left), time.monotonic() - self.t0
            self.event(f"operator: new goal {ahead:.1f} m ahead {left:+.1f} m left")
        else:
            self.event("goal NOT DELIVERED: the node did not answer")

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
        self.frames.write(json.dumps({"wall": time.monotonic(), "text": text}) + "\n")

    def shutdown(self):
        self.attempts.close(time.monotonic() - self.t0, "session ended", self.node_status)
        if "node" in self.procs and not self.args.replay:
            self.node_cmd(cmd="quit")
            time.sleep(0.5)
        for name in ("node", "runner", "sim", "lio", "lidar", "livox", "remote", "rec"):  # the recorder last: it sees the end
            if name in self.procs:
                self.procs[name].stop(signal.SIGTERM if name in ("sim", "lio") else signal.SIGINT)
        self.frames.close()


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

        _, token = start_panel(sess, args.panel, token=os.environ.get("G1_PANEL_TOKEN") or None)
        sess.panel_url = f"http://127.0.0.1:{args.panel}/?t={token}"
        (sess.dir / "panel_url.txt").write_text(sess.panel_url + "\n")
        sess.event(f"panel: port {args.panel} (tailnet / this PC only), token in panel_url.txt")
    steps = [s.strip() for s in args.script.split(",") if s.strip()] if args.script else []
    old = termios.tcgetattr(sys.stdin) if interactive else None
    if interactive:
        tty.setcbreak(sys.stdin.fileno())
    step_t, last_draw, result = time.monotonic(), 0.0, 0
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
            panel_quit = False
            while True:
                try:
                    kind, val, src = sess.inbox.get_nowait()
                except queue.Empty:
                    break
                if kind == "key":
                    sess.event(f"{src}: [{val}]")
                    sess.act(val)
                elif kind == "goal":
                    sess.new_goal(*val)
                elif kind == "node_quit" and not args.replay and "node" in sess.procs:
                    sess.node_cmd(cmd="quit")
                elif kind == "quit":
                    sess.event(f"{src}: end session")
                    panel_quit = True
                elif kind == "arm":
                    sess.event(f"{src}: ARM, starting the runner")
                    sess.start_runner()
            if panel_quit:
                break
            if steps:
                kind, _, val = steps[0].partition(":")
                done = False
                if kind == "sleep":
                    done = now - step_t >= float(val)
                elif kind == "key":
                    sess.act(val)
                    done = True
                elif kind == "goal":
                    a_, l_ = map(float, val.split(":"))
                    sess.new_goal(a_, l_)
                    done = True
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
                        sess.new_goal(a_, l_)
                    except ValueError:
                        sess.event("goal: expected two numbers")
                    tty.setcbreak(sys.stdin.fileno())
                elif ch == "\x1b":
                    sess.act("o")
                else:
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
