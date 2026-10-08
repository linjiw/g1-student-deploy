#!/usr/bin/env python3
"""g1 — operator console for the student planner on the G1 (MuJoCo rehearsal or the real robot).

Runs the real deployment chain: SONIC C++ runner (lucid-g1-deploy, `gamepad_manager` input: keyboard + wireless
remote + ZMQ stream) and the student node (g1deploy/node.py), plus, in `sim` mode, the DDS MuJoCo G1 with the task
obstacles and a realistic MID-360 on the robot's own lidar topic. One screen shows the runner, node and robot
state; single keys drive everything:

    s  start SONIC (runner takes control; the robot stands on SONIC's reference)
    g  GO: hand control to the student (runner streams the node's plans; node pre-scans, plans, walks)
    h  student stand (the stream settles to a stand; deferred near ceilings)
    t  takeover: SONIC's own planner, standing (= remote D-up; NOT near overhead obstacles)
    n  new goal ("ahead left" in metres, relative to the robot now), then g
    o  E-STOP: runner damping (the robot drops: gantry/spotter only)   [also the remote's Select]
    r  (sim) release the virtual gantry          q  quit          ?  help

usage:
  g1 sim   [--layout beam-100 | --task task.json] [--goal AHEAD LEFT] [--script STEPS] [--record DIR]
  g1 robot --iface eth0 --goal 3 0 [--odom leg|zmq] [--mount-calib mount.json] [--body dex3|rubber]
  g1 check|calibrate|bench|replay|test ...     (wrappers around tools/, see docs/OPERATOR_GUIDE.md)

--script runs without a keyboard (tests, videos): comma-separated steps
  wait:init | wait:control | wait:stream | wait:run | wait:arrived | wait:standing | sleep:S | key:K | goal:A:L
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LUCID = Path(os.environ.get("LUCID_DEPLOY_ROOT", Path.home() / "lucid-sonic/lucid-g1-deploy"))
RUNNER = LUCID / "runner/target/release/g1_deploy_onnx_ref"
OBS_CONFIG = LUCID / "config/observation_config_lucid_g1_1570.yaml"
PANELS = Path(os.environ.get("G1_TASK_PANELS", "/home/linjiw/flow/artifacts/panels/test-v1"))
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


class Proc:
    """A child process in its own session, stdout stamped with time.monotonic() into a log file."""

    def __init__(self, name, cmd, log, env, stdin=False):
        self.name, self.log_path = name, Path(log)
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
            if len(self.lines) > 4000:
                del self.lines[:2000]

    def key(self, s):
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
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(args.record) if args.record else ROOT / "runs/console" / f"{args.mode}-{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.procs, self.events = {}, []
        self.runner_state = {"init": False, "control": False, "iface": "GAMEPAD", "stream": False, "estop": False}
        self.node_status, self.node_events = {}, []
        self.frames = open(self.dir / "console_frames.jsonl", "w", buffering=1)
        self.req = None
        self.t0 = time.monotonic()

    # -- processes ----------------------------------------------------------------------------------
    def start(self):
        a, py = self.args, str(ROOT / ".venv/bin/python")
        planner = ROOT / "bundle/sonic/planner/V2/planner_sonic.onnx"
        onnx = ROOT / "bundle/sonic/sonic_release_g1.onnx"
        motions = ROOT / "sim/realtime/motions"
        iface = "lo" if a.mode == "sim" else a.iface
        if a.mode == "sim":
            self.task = a.task or str(PANELS / a.layout / "task.json")
            (self.dir / "remote").mkdir(exist_ok=True)
            self.procs["sim"] = Proc("sim", [py, "-u", "sim/realtime/robot_sim.py", "--task", self.task,
                                             "--log", str(self.dir / "sim.npz"), "--lidar-io", "dds",
                                             "--release-file", str(self.dir / "release.flag"),
                                             "--remote-dir", str(self.dir / "remote")], self.dir / "sim.log", self.env)
            self.event("sim: MuJoCo G1 on DDS (lo), " + Path(self.task).parent.name)
        (self.dir / "session.json").write_text(json.dumps({"mode": a.mode, "task": getattr(self, "task", None),
                                                           "goal": list(a.goal), "t0": self.t0}, indent=1) + "\n")
        runner_cmd = [str(RUNNER), iface, str(onnx), f"{motions}/", "--obs-config", str(OBS_CONFIG),
                      "--input-type", "gamepad_manager", "--planner-file", str(planner),
                      "--zmq-host", "localhost", "--zmq-port", "5556", "--zmq-topic", "pose"]
        if a.mode == "sim":
            runner_cmd.append("--disable-crc-check")  # the simulator computes no CRC; never on the robot
        self.procs["runner"] = Proc("runner", runner_cmd, self.dir / "runner.log", self.env, stdin=True)
        self.event(f"runner: SONIC C++ deploy, gamepad_manager on {iface}")
        node_cmd = [py, "-u", "g1deploy/node.py", "--iface", iface, "--cloud", "dds", "--wait-go",
                    "--goal-frame", "start", "--goal", str(a.goal[0]), str(a.goal[1]),
                    "--log", str(self.dir / "node_plans.json"), "--dump-memory", str(self.dir / "node_memory.npz")]
        if a.mode == "robot":
            node_cmd += ["--odom", a.odom] + (["--mount-calib", a.mount_calib] if a.mount_calib else [])
        if a.goal_lookahead:
            node_cmd += ["--goal-lookahead", str(a.goal_lookahead)]
        if a.candidates:
            node_cmd += ["--candidates", str(a.candidates)]
        if a.ode_steps:
            node_cmd += ["--ode-steps", str(a.ode_steps)]
        node_cmd += ["--planner-impl", a.planner_impl]
        env = dict(self.env, G1_BODY_MODEL=a.body)
        self.procs["node"] = Proc("node", node_cmd, self.dir / "node.log", env)
        self.event(f"node: student planner, goal {a.goal[0]:.1f} m ahead {a.goal[1]:+.1f} m left")
        import zmq

        self.zmq = zmq
        self.req = zmq.Context.instance().socket(zmq.REQ)
        self.req.setsockopt(zmq.LINGER, 0)
        self.req.connect("tcp://localhost:5562")

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
                    self.event("runner: SONIC in CONTROL")
            for _, line in r.lines[-60:]:
                st = self.runner_state
                if "Switched to: ZMQ" in line:
                    st["iface"] = "ZMQ"
                elif "Switched to: GAMEPAD" in line:
                    st["iface"], st["stream"] = "GAMEPAD", False
                elif "ZMQ STREAMING MODE: ENABLED" in line:
                    st["stream"] = True
                elif "STREAMING MODE: DISABLED" in line or "FORCE DISABLED" in line:
                    st["stream"] = False
                elif "EMERGENCY STOP" in line or "Emergency Stop" in line:
                    st["estop"] = True
        n = self.procs.get("node")
        if n:
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
                    self.req.connect("tcp://localhost:5562")
            except self.zmq.ZMQError:
                pass

    def node_cmd(self, **msg):
        try:
            s = self.zmq.Context.instance().socket(self.zmq.REQ)
            s.setsockopt(self.zmq.LINGER, 0)
            s.connect("tcp://localhost:5562")
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
            if st["iface"] != "ZMQ":
                r.key("#")
                time.sleep(0.2)
            if not st["stream"]:
                r.key("\n")  # ZMQ streaming on
            self.node_cmd(cmd="go")
            self.event("operator: GO (stream on, student plans)")
        elif key == "h":
            self.node_cmd(cmd="stand")
            self.event("operator: student stand")
        elif key == "t":
            r.key("@")  # SONIC's own planner (safety reset; streaming off)
            self.node_cmd(cmd="stand")
            self.event("operator: TAKEOVER -> SONIC planner")
        elif key == "o":
            r.key("o")
            self.node_cmd(cmd="quit")
            self.event("operator: E-STOP (damping)")
        elif key == "r" and self.args.mode == "sim":
            (self.dir / "release.flag").touch()
            self.event("sim: gantry released")
        elif key == "?":
            for line in HELP.splitlines():
                self.event(line.strip())

    def new_goal(self, ahead, left):
        self.node_cmd(cmd="goal", ahead=ahead, left=left)
        self.event(f"operator: new goal {ahead:.1f} m ahead {left:+.1f} m left")

    # -- screen ---------------------------------------------------------------------------------------------
    def screen(self, width=78):
        st, ns = self.runner_state, self.node_status
        t = time.monotonic() - self.t0
        mode = "SIMULATION (MuJoCo)" if self.args.mode == "sim" else f"ROBOT on {self.args.iface}"
        runner = ("E-STOP" if st["estop"] else "CONTROL" if st["control"] else "INIT done" if st["init"] else "booting")
        src = "student stream" if (st["iface"] == "ZMQ" and st["stream"]) else (
            "SONIC planner" if st["iface"] == "GAMEPAD" and st["control"] else "SONIC reference")
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
            "-" * width,
            " [s]start SONIC [g]GO student [h]stand [t]takeover [n]goal [o]E-STOP [q]quit",
            "-" * width,
        ]
        rows += [f" {tt:6.1f}  {txt}"[:width] for tt, txt in self.events[-10:]]
        return "\n".join(rows)

    def record_frame(self, text):
        self.frames.write(json.dumps({"wall": time.monotonic(), "text": text}) + "\n")

    def shutdown(self):
        if "node" in self.procs:
            self.node_cmd(cmd="quit")
            time.sleep(0.5)
        for name in ("node", "runner", "sim"):
            if name in self.procs:
                self.procs[name].stop(signal.SIGTERM if name == "sim" else signal.SIGINT)
        self.frames.close()


# ------------------------------------------------------------------------------------------------- main
def run_console(args):
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))  # clean shutdown on TERM
    signal.signal(signal.SIGINT, signal.default_int_handler)  # background jobs inherit SIGINT ignored
    sess = Session(args)
    interactive = sys.stdin.isatty() and not args.script
    if args.mode == "robot":
        print("ROBOT MODE. The robot must hang in its gantry with the feet on the floor, the remote in hand.")
        if interactive and input("Type 'yes' to start the runner on the robot: ").strip() != "yes":
            return 1
    sess.start()
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
            if not sess.procs["runner"].alive() and not steps:
                sess.event("runner exited")
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
        summary = {"dir": str(sess.dir), "events": [f"{t:.1f} {e}" for t, e in sess.events], "status": sess.node_status}
        (sess.dir / "console_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
        print(f"\nsession saved: {sess.dir}")
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tools = {"check": "tools/check_robot.py", "calibrate": "tools/calibrate_mount.py", "bench": "tools/bench_planner.py",
             "replay": "tools/stream_replay.py", "rehearse": "sim/realtime/rehearse.py"}
    if argv and argv[0] in tools:
        return subprocess.call([str(ROOT / ".venv/bin/python"), str(ROOT / tools[argv[0]]), *argv[1:]], env=_env(),
                               cwd=str(ROOT))
    if argv and argv[0] == "test":
        return subprocess.call([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", *argv[1:]], env=_env(), cwd=str(ROOT))
    ap = argparse.ArgumentParser(prog="g1", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("sim", "robot"):
        p = sub.add_parser(mode)
        p.add_argument("--goal", type=float, nargs=2, default=(3.5, 0.0), metavar=("AHEAD", "LEFT"))
        p.add_argument("--goal-lookahead", type=float, default=None, help="1.0 = slower walking (G4: no cost)")
        p.add_argument("--candidates", type=int, default=None, help="planner K (default 16)")
        p.add_argument("--ode-steps", type=int, default=None, help="planner ODE steps (default 10)")
        p.add_argument("--planner-impl", choices=("fused", "graph", "eager"), default="fused",
                       help="planner implementation (fused = fast, same plans; eager = as evaluated)")
        p.add_argument("--body", choices=("dex3", "rubber"), default="dex3", help="hands (self filter)")
        p.add_argument("--script", default="")
        p.add_argument("--step-timeout", type=float, default=90.0)
        p.add_argument("--record", default=None, help="session directory (default runs/console/<mode>-<time>)")
        if mode == "sim":
            p.add_argument("--layout", default="beams2-101", help=f"a layout under {PANELS}")
            p.add_argument("--task", default=None)
            p.add_argument("--remote-start", action="store_true", help="press Start on the virtual remote for [s]")
        else:
            p.add_argument("--iface", required=True)
            p.add_argument("--odom", choices=("leg", "zmq"), default="leg")
            p.add_argument("--mount-calib", default=None)
    args = ap.parse_args(argv)
    if args.mode == "robot":
        args.remote_start = False
    return run_console(args)


if __name__ == "__main__":
    sys.exit(main())
