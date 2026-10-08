#!/usr/bin/env python3
"""Path B: one real-time deployment rehearsal of the student against the SONIC C++ runner.

Processes (all on this machine, DDS over loopback, as on the robot apart from the sensor drivers):
  1. sim/realtime/robot_sim.py   MuJoCo G1 on DDS + task boxes + MID-360 cloud (ZMQ :5560)
  2. SONIC C++ runner             lucid-g1-deploy g1_deploy_onnx_ref, fused release ONNX (1570 -> 29),
                                  `--input-type zmq` (SUB localhost:5556 topic pose), TensorRT
  3. g1deploy/node.py             the student planner node (DDS state + odostate + cloud -> stream)

Operator sequence (scripted, same keys as on the robot):
  runner "Init Done" -> `]` (SONIC takes the stand) -> release the gantry -> Enter (ZMQ streaming)
  -> node pre-scans the lidar, plans from standing, streams; estop event -> `O`.
Scored afterwards from the simulator's physics log with the Isaac TaskScorer rules, from the moment
the first plan streamed; deadline = the task's deadline_s.

usage: rehearse.py --task task.json --out runs/rt/<name> [--route sensor|privileged] [--inflight 2]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from g1deploy import bundle  # noqa: E402
from g1deploy.scoring import TaskScorer  # noqa: E402

LUCID = Path(os.environ.get("LUCID_DEPLOY_ROOT", "/home/linjiw/lucid-sonic/lucid-g1-deploy"))
RUNNER = LUCID / "runner/target/release/g1_deploy_onnx_ref"
OBS_CONFIG = LUCID / "config/observation_config_lucid_g1_1570.yaml"
MOTIONS = ROOT / "sim/realtime/motions"


class Proc:
    def __init__(self, name, cmd, log, env, stdin=False, stamp=False):
        self.name, self.log_path = name, Path(log)
        self.f = open(self.log_path, "w", buffering=1)
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE if stamp else self.f, stderr=subprocess.STDOUT, env=env,
                                  cwd=str(ROOT), stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                  start_new_session=True)
        if stamp:  # prefix every line with time.monotonic() so runner events align with node/sim logs
            def pump():
                for raw in iter(self.p.stdout.readline, b""):
                    self.f.write(f"{time.monotonic():.3f} {raw.decode(errors='replace')}")
            threading.Thread(target=pump, daemon=True).start()

    def text(self):
        try:
            return self.log_path.read_text(errors="replace")
        except FileNotFoundError:
            return ""

    def wait_for(self, pattern, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if pattern in self.text():
                return True
            if self.p.poll() is not None:
                return False
            time.sleep(0.1)
        return False

    def key(self, s):
        self.p.stdin.write(s.encode())
        self.p.stdin.flush()

    def stop(self, sig=signal.SIGINT, timeout=5.0):
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, sig)
                self.p.wait(timeout)
            except Exception:
                os.killpg(self.p.pid, signal.SIGKILL)
        self.f.close()


def score(task, sim_log, t_run_wall, t_end_wall):
    z = np.load(sim_log)
    wall, pos, quat, contact = z["wall"], z["root_pos"], z["root_quat"], z["contact_n"]
    sel = np.flatnonzero((wall >= t_run_wall) & (wall <= t_end_wall))
    scorer = TaskScorer(task)
    outcome = None
    for k in range(0, len(sel) - 1, 2):  # log at 100 Hz -> 50 Hz control ticks
        i = sel[k:k + 2]
        v = (pos[i[-1]] - pos[max(i[0] - 1, 0)]) / max(wall[i[-1]] - wall[max(i[0] - 1, 0)], 1e-3)
        outcome = scorer.update(pos[i[-1]], v, physics_root_positions=pos[i], physics_root_quaternions=quat[i],
                                physics_contact_force_n=contact[i])
        if outcome is not None:
            break
    rep = scorer.report()
    if outcome is None:
        rep["outcome"] = "timeout" if scorer.steps * 0.02 >= task["deadline_s"] - 0.05 else "incomplete"
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--route", choices=("sensor", "privileged"), default="sensor")
    ap.add_argument("--inflight", type=int, default=1,
                    help="2+ is harmful: a second in-flight plan is predicted from the landed plan, not the one "
                         "running in between (G4: 32/98 vs 79/98)")
    ap.add_argument("--prescan-s", type=float, default=0.5)
    ap.add_argument("--boot-timeout", type=float, default=600.0, help="first run builds the TensorRT engine")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--lidar-io", choices=("zmq", "dds"), default="dds",
                    help="dds: the G1's own lidar topic and format (default); zmq: deskewed world clouds")
    ap.add_argument("--lidar-cfg", default=None, help="simulator MID-360 options (JSON), e.g. a mount error")
    ap.add_argument("--mount-calib", default=None)
    ap.add_argument("--odom", choices=("dds", "leg"), default="dds",
                    help="dds: simulator ground truth (stands in for LIO); leg: leg odometry from rt/lowstate only")
    ap.add_argument("--replay", default=None,
                    help="H4 check: stream this recorded episode (closed_loop.npz) open-loop instead of running the node")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    task = json.loads(Path(args.task).read_text())
    env = dict(os.environ)
    dds = Path(env.get("CYCLONEDDS_HOME", Path.home() / "opt/sonic-deploy-toolchain/cyclonedds"))
    env["LD_LIBRARY_PATH"] = f"{dds / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    lucid_env = subprocess.run(["bash", "-c", f"source {LUCID}/env.sh >/dev/null && env -0"], capture_output=True).stdout
    for kv in lucid_env.split(b"\0"):
        if kv.startswith((b"onnxruntime_ROOT=", b"LD_LIBRARY_PATH=", b"CUDA_HOME=")):
            k, v = kv.decode().split("=", 1)
            env[k] = v if k != "LD_LIBRARY_PATH" else f"{dds / 'lib'}:{v}"
    env["PYTHONPATH"] = f"{ROOT}:{LUCID / 'sdk'}"
    env.setdefault("G1_BODY_MODEL", "dex3")  # the rehearsal robot (lucid gear_sonic_sim) has Dex3 hands
    py = str(ROOT / ".venv/bin/python")
    release, start = out / "release.flag", out / "start.flag"
    boxes_file = out / "boxes.json"
    boxes_file.write_text(json.dumps(task["scene_boxes_world"]))
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    procs = []
    result = {"task_id": task.get("task_id"), "family": task.get("family")}
    t_run = t_stop = arrived_at = None
    try:
        sim = Proc("sim", [py, "-u", "sim/realtime/robot_sim.py", "--task", args.task, "--log", str(out / "sim.npz"),
                           "--release-file", str(release), "--lidar-io", args.lidar_io]
                   + (["--lidar-cfg", args.lidar_cfg] if args.lidar_cfg else []) + (["--viewer"] if args.viewer else []),
                   out / "sim.log", env)
        procs.append(sim)
        if not sim.wait_for("[sim] task", 60):
            raise RuntimeError("simulator did not start:\n" + sim.text()[-2000:])
        runner = Proc("runner", [str(RUNNER), "lo", str(bundle.sonic_onnx()), f"{MOTIONS}/", "--obs-config", str(OBS_CONFIG),
                                 "--disable-crc-check", "--input-type", "zmq", "--zmq-host", "localhost", "--zmq-port",
                                 "5556", "--zmq-topic", "pose"], out / "runner.log", env, stdin=True, stamp=True)
        procs.append(runner)
        if not runner.wait_for("Init Done", args.boot_timeout):
            raise RuntimeError("runner did not reach Init Done:\n" + runner.text()[-3000:])
        time.sleep(1.0)
        runner.key("]")
        if not runner.wait_for("transitioning to CONTROL", 15):
            raise RuntimeError("runner did not enter CONTROL:\n" + runner.text()[-2000:])
        time.sleep(1.0)
        release.touch()  # gantry released: SONIC balances the stand
        time.sleep(1.5)
        if args.replay:
            runner.key("\n")  # Enter: ZMQ streaming
            time.sleep(0.3)
            node = Proc("node", [py, "-u", "tools/stream_replay.py", args.replay], out / "node.log", env)
            procs.append(node)
            if not node.wait_for("frames", 60):
                raise RuntimeError("replay did not start:\n" + node.text()[-2000:])
            t_run = time.monotonic() + 1.0  # stream_replay prefixes a 1 s settle from the stand
            dur = float(node.text().split(" frames (")[1].split(" s)")[0])  # "<n> frames (<s> s)"
            t_end, arrived_at = min(t_run + task["deadline_s"], t_run - 1.0 + dur + 1.0), None
            while time.monotonic() < t_end:
                time.sleep(0.05)
        else:
            goal_args = ["--goal", *map(str, task["goal_world_xy"])]
            if args.odom == "leg":  # the robot only knows where it started: goal as (ahead, left) of the start pose
                z0 = np.load(task["initial_state_record"])
                q0 = z0["root_quaternion_wxyz"][int(task.get("initial_frame", 0))]
                yaw0 = float(np.arctan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2)))
                start_xy = np.asarray(task.get("initial_root_xy", z0["root_position_m"][0][:2]), dtype=float)
                d = np.asarray(task["goal_world_xy"]) - start_xy
                rel = [np.cos(yaw0) * d[0] + np.sin(yaw0) * d[1], -np.sin(yaw0) * d[0] + np.cos(yaw0) * d[1]]
                goal_args = ["--goal", *map(str, rel), "--goal-frame", "start", "--odom", "leg"]
            node = Proc("node", [py, "-u", "g1deploy/node.py", *goal_args, "--route", args.route,
                                 "--cloud", "dds" if args.lidar_io == "dds" else "tcp://localhost:5560",
                                 *(["--mount-calib", args.mount_calib] if args.mount_calib else []),
                                 "--boxes", str(boxes_file), "--inflight", str(args.inflight), "--prescan-s",
                                 str(args.prescan_s), "--start-file", str(start), "--log", str(out / "node_plans.json")],
                        out / "node.log", env)
            procs.append(node)
            if not node.wait_for('"node_up"', 120):
                raise RuntimeError("node did not start:\n" + node.text()[-3000:])
            runner.key("\n")  # Enter: switch the runner to ZMQ streaming
            time.sleep(0.3)
            start.touch()
            if not node.wait_for('"run"', 60):
                raise RuntimeError("node never started streaming:\n" + node.text()[-3000:])
            events = [json.loads(l) for l in node.text().splitlines() if l.startswith("{")]
            t_run = next(e["t"] for e in events if e["event"] == "run")
            t_end, arrived_at = t_run + task["deadline_s"], None
            while time.monotonic() < t_end:
                txt = node.text()
                if '"estop"' in txt:
                    runner.key("o")
                    result["estop"] = True
                    time.sleep(1.0)
                    break
                if '"arrived"' in txt and arrived_at is None:
                    arrived_at = time.monotonic()
                if arrived_at is not None and time.monotonic() - arrived_at > 0.5:
                    break
                if any(p.p.poll() is not None for p in procs):
                    break
                time.sleep(0.05)
        t_stop = time.monotonic()
    finally:
        for p in reversed(procs):
            p.stop(signal.SIGTERM if p.name == "sim" else signal.SIGINT)
    rep = score(task, out / "sim.npz", t_run, t_stop)
    plans = json.loads((out / "node_plans.json").read_text())["plans"] if (out / "node_plans.json").exists() else []
    lat = [p["latency_s"] for p in plans]
    late = [p["late_ticks"] for p in plans]
    runner_txt = runner.text()
    waits = [float(l.split(" ", 1)[0]) for l in runner_txt.splitlines() if "waiting following motion" in l]
    stalls_in_run = sum(1 for w in waits if t_run <= w <= (arrived_at or t_stop))
    result.update(rep, runner_stall_lines_during_run=stalls_in_run, plan_latency_s={"p50": float(np.median(lat)), "p95": float(np.quantile(lat, .95))} if lat else None,
                  late_ticks_p95=float(np.quantile(late, .95)) if late else None, plans=len(plans),
                  runner_catchup_resets=runner_txt.count("catch-up") + runner_txt.count("Catch-up"),
                  runner_waiting=runner_txt.count("waiting following motion"))
    (out / "score.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: result.get(k) for k in ("task_id", "outcome", "elapsed_s", "plans", "plan_latency_s",
                                                 "late_ticks_p95", "runner_catchup_resets",
                                                 "runner_stall_lines_during_run")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
