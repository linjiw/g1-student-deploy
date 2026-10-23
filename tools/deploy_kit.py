#!/usr/bin/env python3
"""Deployment kit helpers behind `g1 doctor | sessions | note | courses | lidar` (docs/DEPLOY_KIT.md).

  g1 doctor [--robot NAME | --iface NIC]   pre-flight: desk (GPU, bundle, runner, engines, disk, stale processes)
                                           and, with a robot, network, DDS rates, controller mode, lidar path,
                                           calibration. Read-only: it sends the robot one mode *query*, nothing else.
  g1 sessions [-n 15]                      recorded sessions, newest first: mode, walks, flags, size
  g1 note [SESSION|latest] "text"          append a timestamped note (e.g. "tape: stopped 2.93 m ahead, 0.08 left")
  g1 courses                               the simulation course library (sim/courses)
  g1 lidar start|stop|status [--robot NAME | --iface NIC]
                                           robots without Unitree's lidar service: livox_ros_driver2 (if not running)
                                           + g1deploy/adapters/livox_to_utlidar.py in the background, for `g1 check
                                           --perception` / `g1 calibrate` before a console session
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy.console import LIVOX_WS, LUCID, ROBOTS, ROS_SETUP, RUNNER, SESSIONS, cyclone_xml  # noqa: E402

OK, WARN, FAIL, INFO = "ok", "WARN", "FAIL", "·"
PORTS = {5556: "node pose stream", 5557: "runner g1_debug", 5560: "sim cloud", 5561: "LIO bridge", 5562: "node control"}
LIDAR_DIR = ROOT / "runs/lidar"


def profile(args):
    if getattr(args, "robot", None):
        path = ROBOTS / f"{args.robot}.json"
        if not path.exists():
            raise SystemExit(f"no robot profile {path}")
        prof = json.loads(path.read_text())
        if getattr(args, "iface", None):
            prof["iface"] = args.iface
        return prof
    return {"iface": args.iface} if getattr(args, "iface", None) else {}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, section, name, status, detail="", fix=""):
        self.rows.append((section, name, status, detail, fix))
        mark = {OK: "✓", WARN: "!", FAIL: "✗", INFO: "·"}[status]
        print(f"  {mark} {name:<26} {detail}" + (f"\n      → {fix}" if fix and status in (WARN, FAIL) else ""),
              flush=True)


def sh(cmd, timeout=10):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


# --------------------------------------------------------------------------------------------- doctor
def doctor_desk(r: Report):
    print("\nDesk (this PC)")
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            r.add("desk", "GPU / CUDA", OK, f"{torch.cuda.get_device_name(0)}, {free / 1e9:.1f} of {total / 1e9:.1f} GB free")
        else:
            r.add("desk", "GPU / CUDA", FAIL, "torch sees no CUDA device", "check nvidia-smi and the .venv torch build")
    except Exception as exc:
        r.add("desk", "GPU / CUDA", FAIL, str(exc))
    try:
        from g1deploy import bundle

        bad = bundle.verify(quick=True)
        r.add("desk", "Policy bundle", OK if not bad else FAIL, "manifest sizes match" if not bad else str(bad)[:120],
              "re-extract the deploy-2026-10-23 release into bundle/ (README, docs/DEPLOY_KIT.md)")
    except Exception as exc:
        r.add("desk", "Policy bundle", FAIL, f"{type(exc).__name__}: {exc}"[:160])
    if RUNNER.exists():
        commit = sh(f"git -C {LUCID} rev-parse --short HEAD")
        r.add("desk", "SONIC runner", OK, f"{RUNNER.name} @ {commit}")
    else:
        r.add("desk", "SONIC runner", FAIL, f"missing {RUNNER}", f"cd {LUCID} && source env.sh && bash build.sh")
    engines = sorted(p.name for p in (ROOT / "bundle/sonic").rglob("*.trt"))
    r.add("desk", "TensorRT engines", OK if len(engines) >= 2 else WARN, ", ".join(engines) or "none",
          "the first runner start builds them (policy ~30 s, planner ~7 min): do it in sim before robot day")
    r.add("desk", "Planner kernel cache", OK if (ROOT / ".cache/inductor").is_dir() else WARN,
          str(ROOT / ".cache/inductor"), "run `./g1 sim` once: the first node start compiles kernels (~2 min)")
    SESSIONS.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(SESSIONS).free / 1e9
    r.add("desk", "Disk for sessions", OK if free > 50 else WARN if free > 15 else FAIL,
          f"{free:.0f} GB free at {SESSIONS} (a robot session records ~3 MB/s)", "free disk or set G1_SESSIONS")
    desc = sh(f"git -C {ROOT} describe --tags --always")
    dirty = sh(f"git -C {ROOT} status --porcelain --untracked-files=no")
    r.add("desk", "Deploy code", OK if not dirty else INFO, f"{desc}" + (" (modified: " + " ".join(
        l.split(maxsplit=1)[-1] for l in dirty.splitlines()[:4]) + ")" if dirty else ""))
    from g1deploy.dds import rmem_max

    rmax = rmem_max()
    r.add("desk", "UDP receive buffer cap", OK if rmax and rmax >= 8 << 20 else FAIL,
          f"net.core.rmem_max = {rmax} (lidar frames are 0.3-0.9 MB bursts)",
          "printf 'net.core.rmem_max=33554432\\nnet.core.rmem_default=8388608\\n' | sudo tee "
          "/etc/sysctl.d/60-g1-dds.conf && sudo sysctl --system   (g1deploy/dds.py)")
    procs = sh("pgrep -af 'g1_deploy_onnx_ref|g1deploy/node.py|robot_sim.py|stream_replay.py|g1deploy/recorder.py' "
               "| grep -v pgrep")
    r.add("desk", "Stale deploy processes", OK if not procs else FAIL, "none" if not procs else procs.splitlines()[0][:100],
          "a previous session is still running: quit it (or kill those PIDs) before starting another")
    listening = sh("ss -ltn")
    busy = [f"{p} ({n})" for p, n in PORTS.items() if f":{p} " in listening]
    r.add("desk", "ZMQ ports free", OK if not busy else FAIL, "5556 5557 5560 5561 5562" if not busy else ", ".join(busy),
          "another process holds the deploy ports: find it with `ss -ltnp`")


def dds_sample(iface, seconds=3.0):
    """Count lowstate / lowcmd / lidar samples for a few seconds (subscribe only) and query the controller mode."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    from g1deploy.perception.unitree_lidar import TOPIC_CLOUD

    ChannelFactoryInitialize(0, iface)
    out = {"lowstate": 0, "lowcmd": 0, "lidar": 0, "kp_max": 0.0, "ticks": set(), "tilt": 0.0, "temp_max": 0}
    lock = threading.Lock()

    def on_low(m):
        with lock:
            out["lowstate"] += 1
            out["ticks"].add(int(m.tick))
            w, x, y, z = m.imu_state.quaternion
            out["tilt"] = max(out["tilt"], float(__import__("math").acos(max(-1, min(1, 1 - 2 * (x * x + y * y))))))
            out["temp_max"] = max(out["temp_max"], max(int(s.temperature[0]) for s in list(m.motor_state)[:29]))

    def on_cmd(m):
        with lock:
            out["lowcmd"] += 1
            out["kp_max"] = max(out["kp_max"], max(float(c.kp) for c in m.motor_cmd))

    def on_cloud(m):
        with lock:
            out["lidar"] += 1

    subs = []
    for topic, typ, cb in (("rt/lowstate", LowState_, on_low), ("rt/lowcmd", LowCmd_, on_cmd),
                           (TOPIC_CLOUD, PointCloud2_, on_cloud)):
        s = ChannelSubscriber(topic, typ)
        s.Init(cb, 100)
        subs.append(s)
    time.sleep(seconds)
    with lock:
        res = {k: v for k, v in out.items() if k != "ticks"}
        res["unique_ticks"] = len(out["ticks"])
    mode = None
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        c = MotionSwitcherClient()
        c.SetTimeout(3.0)
        c.Init()
        code, data = c.CheckMode()  # a query: never SelectMode / ReleaseMode here
        mode = data.get("name") if code == 0 and data else f"query failed (code {code})"
    except Exception as exc:
        mode = f"query failed ({exc})"
    res["mode"] = mode
    return res


def doctor_robot(r: Report, prof):
    iface = prof.get("iface")
    print(f"\nRobot ({prof.get('name', iface)} on {iface})")
    if not Path(f"/sys/class/net/{iface}").exists():
        r.add("robot", "Network interface", FAIL, f"no interface {iface}", "ip -br link  (pick the wired port)")
        return
    carrier = sh(f"cat /sys/class/net/{iface}/carrier") == "1"
    addr = sh(f"ip -4 -br addr show {iface}")
    ok_ip = "192.168.123." in addr
    r.add("robot", "Link + address", OK if carrier and ok_ip else FAIL, addr or f"{iface} down",
          "cable in? then `nmcli con up g1-robot` (static 192.168.123.222/24, never-default)")
    if not (carrier and ok_ip):
        return
    for host, name in (("161", "PC1 (motion, DDS)"), ("164", "Orin (PC2)"), ("120", "MID-360 lidar")):
        out = sh(f"ping -c 2 -W 1 -I {iface} 192.168.123.{host} | tail -1")
        r.add("robot", f"ping .{host} {name}", OK if "rtt" in out else (WARN if host == "164" else FAIL),
              out.split("=")[-1].strip() if "rtt" in out else "no reply", "power / cable / robot IP plan")
    s = dds_sample(iface)
    hz = s["unique_ticks"] / 3.0
    r.add("robot", "rt/lowstate", OK if hz > 400 else FAIL,
          f"{hz:.0f} Hz unique ticks ({s['lowstate'] / 3.0:.0f} Hz received), tilt {s['tilt']:.2f} rad, "
          f"motors max {s['temp_max']} °C", "no robot state: DDS domain 0 on this NIC? robot booted?")
    if s["lowcmd"] and s["kp_max"] > 0:
        r.add("robot", "rt/lowcmd writer", WARN, f"{s['lowcmd'] / 3.0:.0f} Hz with kp up to {s['kp_max']:.0f}: something "
              "holds the motors (Unitree controller standing, or another runner)",
              "only one motor writer: the SONIC runner releases Unitree's controller itself; stop any other runner")
    else:
        r.add("robot", "rt/lowcmd writer", OK, f"{s['lowcmd'] / 3.0:.0f} Hz zero-gain (Unitree controller idle)"
              if s["lowcmd"] else "none")
    r.add("robot", "Controller mode", INFO, f"{s['mode']!r} (the runner releases it at start)")
    lidar_hz = s["lidar"] / 3.0
    hint = ("`g1 lidar start --robot NAME` (livox adapter)" if prof.get("lidar") == "livox"
            else "Unitree lidar service on? (this firmware may lack it: use lidar 'livox')")
    r.add("robot", "rt/utlidar cloud", OK if lidar_hz >= 8 else WARN, f"{lidar_hz:.1f} Hz", hint)
    if prof.get("lidar") == "livox":
        drv = sh("pgrep -f '^[^ ]*livox_ros_driver2_node'")
        r.add("robot", "livox_ros_driver2", OK if drv else INFO, "running" if drv else "not running (the console starts it)")
    mc = prof.get("mount_calib")
    if mc:
        path = Path(mc) if Path(mc).is_absolute() else ROOT / mc
        r.add("robot", "Mount calibration", OK if path.exists() else FAIL, str(path),
              "g1 calibrate --iface IFACE --out " + str(path) + " (robot standing still on open floor)")
    else:
        r.add("robot", "Mount calibration", WARN, "none in the profile",
              "`g1 calibrate` is mandatory before walking (3° error -> 15/98 in sim)")


def cmd_doctor(args):
    r = Report()
    doctor_desk(r)
    prof = profile(args)
    if prof.get("iface"):
        doctor_robot(r, prof)
    fails = [x for x in r.rows if x[2] == FAIL]
    warns = [x for x in r.rows if x[2] == WARN]
    print(f"\n{'READY' if not fails else 'NOT READY'}: {len(fails)} failing, {len(warns)} warnings")
    return 1 if fails else 0


# ------------------------------------------------------------------------------------------- sessions
def session_dirs():
    return sorted((p for p in SESSIONS.glob("*") if (p / "session.json").exists()), key=lambda p: p.stat().st_mtime,
                  reverse=True)


def resolve_session(name):
    if name in (None, "latest"):
        dirs = session_dirs()
        if not dirs:
            raise SystemExit(f"no sessions under {SESSIONS}")
        return dirs[0]
    p = Path(name)
    return p if p.is_dir() else SESSIONS / name


def cmd_sessions(args):
    print(f"{'session':<40} {'mode':<6} {'walks':<24} {'flags':>5} {'MB':>7}")
    for d in session_dirs()[:args.n]:
        rep = {}
        try:
            rep = json.loads((d / "report.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
        legs = rep.get("node", {}).get("legs", [])
        walks = " ".join((l.get("end") or "?").split(" ")[0][:7] for l in legs)[:24] or "-"
        mb = rep.get("size_mb")
        if mb is None:
            mb = round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 1)
        mode = json.loads((d / "session.json").read_text()).get("mode", "?")
        print(f"{d.name:<40} {mode:<6} {walks:<24} {len(rep.get('flags', [])) if rep else '?':>5} {mb:>7}")
    return 0


def cmd_note(args):
    d = resolve_session(args.session)
    with open(d / "notes.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "text": args.text}) + "\n")
    print(f"noted in {d / 'notes.jsonl'}")
    return 0


def cmd_courses(args):
    for t in sorted((ROOT / "sim/courses").glob("*/task.json")):
        doc = json.loads(t.read_text())
        print(f"  {t.parent.name:<12} goal {doc.get('goal_world_xy')}  boxes {len(doc.get('scene_boxes_world', []))}"
              + (f"  mission {doc['mission']}" if doc.get("mission") else "") + f"\n      {doc.get('description', '')}")
    print("\n  ./g1 sim --layout NAME [--mission course] [--odom leg]")
    return 0


# ---------------------------------------------------------------------------------------------- lidar
def cmd_lidar(args):
    LIDAR_DIR.mkdir(parents=True, exist_ok=True)
    pidfile = LIDAR_DIR / "pids.json"
    pids = json.loads(pidfile.read_text()) if pidfile.exists() else {}

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except (OSError, TypeError):
            return False

    if args.action == "stop":
        for name, pid in pids.items():
            if alive(pid):
                os.killpg(pid, signal.SIGINT)
                print(f"stopped {name} ({pid})")
        pidfile.unlink(missing_ok=True)
        return 0
    prof = profile(args)
    iface = prof.get("iface")
    if args.action == "status":
        for name, pid in pids.items():
            print(f"  {name:<8} pid {pid} {'running' if alive(pid) else 'gone'}")
        print("  livox_ros_driver2:", "running" if sh("pgrep -f '^[^ ]*livox_ros_driver2_node'") else "not running")
        print("  adapters running:", sh("pgrep -f '^/usr/bin/python3 .*livox_to_utlidar' | wc -l"))
        if iface:
            s = dds_sample(iface, 2.0)
            print(f"  rt/utlidar/cloud_livox_mid360 on {iface}: {s['lidar'] / 2.0:.1f} Hz")
        return 0
    if not iface:
        raise SystemExit("give --robot NAME or --iface NIC")
    if any(alive(p) for p in pids.values()) or sh("pgrep -f '^/usr/bin/python3 .*livox_to_utlidar'"):
        raise SystemExit("an adapter is already running (two would publish every frame twice): `g1 lidar status`")
    xml = cyclone_xml(iface, LIDAR_DIR / f"cyclonedds_{iface}.xml")
    ros = (f"source {ROS_SETUP} >/dev/null && source {LIVOX_WS} >/dev/null && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
           f"CYCLONEDDS_URI=file://{xml} && ")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
    started = {}
    if not sh("pgrep -f '^[^ ]*livox_ros_driver2_node'"):
        p = subprocess.Popen(["bash", "-c", ros + "exec ros2 launch livox_ros_driver2 msg_MID360_launch.py"],
                             stdout=open(LIDAR_DIR / "livox.log", "w"), stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        started["livox"] = p.pid
    p = subprocess.Popen(["bash", "-c", ros + f"exec /usr/bin/python3 -u {ROOT}/g1deploy/adapters/livox_to_utlidar.py --stats-s 10 --custom-out /livox/lidar_10hz"],
                         stdout=open(LIDAR_DIR / "adapter.log", "w"), stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    started["adapter"] = p.pid
    pidfile.write_text(json.dumps(started))
    time.sleep(4)
    s = dds_sample(iface, 2.0)
    print(f"started {started}; rt/utlidar/cloud_livox_mid360 on {iface}: {s['lidar'] / 2.0:.1f} Hz "
          f"(logs in {LIDAR_DIR}; `g1 lidar stop` ends them)")
    return 0 if s["lidar"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--robot")
    p.add_argument("--iface")
    p = sub.add_parser("sessions")
    p.add_argument("-n", type=int, default=15)
    p = sub.add_parser("note")
    p.add_argument("session", nargs="?", default="latest")
    p.add_argument("text")
    sub.add_parser("courses")
    p = sub.add_parser("lidar")
    p.add_argument("action", choices=("start", "stop", "status"))
    p.add_argument("--robot")
    p.add_argument("--iface")
    args = ap.parse_args(argv)
    return {"doctor": cmd_doctor, "sessions": cmd_sessions, "note": cmd_note, "courses": cmd_courses,
            "lidar": cmd_lidar}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
