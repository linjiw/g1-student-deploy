#!/usr/bin/env python3
"""g1 lab — the robot day's deploy service: scans and their twins, practice in simulation, rehearsal sets, the start
check and goal conversion, the robot session's start and stop, and every session's evidence, as a JSON API (v1) for
Deploy Studio, plus the original hub page at /legacy.

`./g1 lab [--headless] [--port 8769] [--robot lab-g1] [--day runs/lab-day-YYYYMMDD] [--no-studio-link]` serves on
127.0.0.1:PORT; every request needs `?t=TOKEN` (g1deploy/service_token.py: G1_PANEL_TOKEN, else
~/.config/g1-deploy/panel_token, the token the consoles use too). The hub never drives the robot itself. It starts the
console (`g1 sim|robot ... --profile P --arm --panel PORT`) as its own process group, so the console keeps running if
the hub stops, and relays operator acts to the console's panel (/act), where every motion gate stays. A robot session
ends from its panel; the hub stops a session only before Arm, after an E-STOP, or standing after an attempt (never
while it walks). A restarted hub re-attaches to the consoles still running, from any lab-day-* directory beside its
own.

API v1 (strict JSON: non-finite numbers are null; state computed at 1 Hz in the background, served from a cache):
  GET  /api/v1/state | /api/v1/readiness?scan=&slot= | /api/v1/sessions?offset=&limit=
  GET  /api/v1/scan/live | /api/v1/scan/<name>/scene | /api/v1/rehearsal/<run>/track
  GET  /api/v1/live/<slot>/view?cloud_after=&mem_after= | /api/v1/panel/<slot>/state | /api/v1/session/<name>/manifest
  POST /api/v1/scan | /api/v1/practice/scan | /api/v1/start | /api/v1/stop | /api/v1/start_check | /api/v1/send_goal
  POST /api/v1/rehearse | /api/v1/act | /api/v1/doctor
Frames (g1deploy/frames.py): goals cross the API as goal_scene; the hub converts them to each console's odometry.

Sessions land in the day directory, which is also linked into Deploy Studio's robot-sessions directory so its Replay
lists each one (skipped with --no-studio-link). Every robot session that ends gets a twin of its final map too.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import attempts as A, contract, evidence, frames, machine, profiles, registration, wire, workspace  # noqa: E402
from g1deploy.service_token import panel_token  # noqa: E402

STUDIO_DATA = Path(os.environ.get("G1_STUDIO_DATA", Path.home() / "research-data/g1-deploy-studio"))
STUDIO_PORT = int(os.environ.get("G1_STUDIO_PORT", "8790"))
PANEL_PORTS = {"robot": 8770, "sim": 8771}
REHEARSAL_PORT_BASE = 5656  # hub sims run isolated (g1 sim --port-base): they may run beside a live robot session
PORT_BASES = {"robot": 5556, "sim": REHEARSAL_PORT_BASE}  # node control = base + 6
# GPU memory a rehearsal chain needs beside a live robot session: its planner (a larger bundle grows to ~7-8 GB on
# no-safe resampling, tools/replay_plans.py measurements) and a second SONIC runner (~1 GB)
REHEARSAL_GPU_MIB = int(os.environ.get("G1_REHEARSAL_GPU_MIB", "8500"))
LIDAR_LOG = Path(os.environ.get("G1_LIDAR_DIR", ROOT / "runs/lidar")) / "adapter.log"  # `g1 lidar start` --stats-s 10
LIO_LOG = Path(os.environ.get("G1_LIO_DIR", ROOT / "runs/lio")) / "bridge.log"  # `g1 lio start` stats every 5 s
TAILNET_HOST = os.environ.get("G1_TAILNET_HOST", "g1-deploy-pc")


def gpu_free_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        used, total = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return total - used
    except Exception:
        return None
# the automatic rehearsal in a scanned twin: the robot-day ladder without a hand on it (the panel shows it live)
REHEARSE_SCRIPT = "wait:init,key:s,wait:control,wait:ready,sleep:3,key:g,wait:arrived,sleep:3"
# The second rehearsal of every pair: true odometry with the step-tied error lab-g1's leg odometry showed
# (g1deploy/odom_error.py preset lab-g1-leg, through the sim LIO stand-in: the node registers raw lidar frames with
# that pose, as on the robot). A goal the exact rehearsal reaches but the drifted one does not, or reaches another way,
# depends on odometry the robot may not have. (Not the sim's own leg odometry: it already under-counts ~17 %.)
DRIFT = {"motion": "lab-g1-leg"}
OTHER_WAY_M = 0.5  # the drifted walk left the exact walk's path by more than this: it went another way
SCAN_FRESH_S = 15 * 60  # a robot session reuses the scan its goal was picked on when it is younger than this
REHEARSAL_COURSES = ("lab-door-real", "rp-door-rope", "rp-rope-behind", "lab-site-2m", "open-2m", "open-3m", "slalom",
                     "pillar", "box-low")
# API v1
CONFIRM_ROBOT = ("gantry", "estop", "remote", "rope")  # the robot start checklist (stored in the session)
CONFIRM_LEGACY = ("gantry", "estop", "remote", "clear")  # the legacy page's
RUN_KINDS = ("exact", "drift", "offset")  # a rehearsal set, in order
ENVELOPE_SLACK_M, ENVELOPE_SLACK_DEG = 0.05, 5.0  # the start check is inside a set's offset envelope with this slack
TRACK_HZ, CONTACT_N = 25.0, 5.0
GOAL_CLEAR_M = 0.25  # the coarse live goal check: no obstacle cell this close to the goal
OBSTACLE_M = 0.3  # workspace.MAP_CODES: support >= 0.3 m is an obstacle ('2', '3')
VIEW_HZ, VIEW_TIMEOUT_S, VIEW_RETRY_S = 5.0, 0.5, 30.0
SESSIONS_PAGE = 40
CAPABILITIES = ["state", "readiness", "sessions", "scan", "scan_live", "scan_scene", "practice", "start", "stop",
                "start_check", "send_goal", "rehearse", "track", "live_view", "act", "panel_state", "manifest", "doctor"]
STATUS_OF_LEVEL = {"ok": "pass", "warn": "warn", "bad": "block", "na": "unknown"}
NOTICE = """<!doctype html><meta charset="utf-8"><title>G1 deploy service</title>
<body style="font:16px system-ui;margin:2em;max-width:44em">
<h1>G1 deploy service</h1>
<p>The deploy experience is in <b>Deploy Studio</b> (its Deploy workspace). This hub is its headless service: API v1
under <code>/api/v1/</code>.</p>
<p>The original hub page is still here: <a href="/legacy?t=__TOKEN__">/legacy</a>.</p></body>"""


def studio_id(day: Path, session: Path) -> str:
    """Deploy Studio's Replay id of a console session linked one level down (studio/server.py discover_console)."""
    return "robot-" + re.sub(r"[^a-zA-Z0-9_-]", "-", f"{day.name}-{session.name}")[:60]


def list_profiles():
    out = []
    for d in profiles.SEARCH:
        for p in sorted(d.glob("*.json")):
            try:
                prof, path = profiles.load(p)
            except profiles.ProfileError as exc:
                out.append({"name": p.stem, "error": str(exc)})
                continue
            raw = json.loads(path.read_text())  # name / description / student are informational (profiles.INFO)
            out.append({"name": raw.get("name") or p.stem, "file": str(path.relative_to(ROOT)),
                        "private": "private" in path.parts, "description": raw.get("description", ""),
                        "student": raw.get("student"), "rules_overrides": raw.get("rules_overrides"),
                        "bundle": Path(prof["bundle"]).name if prof.get("bundle") else "bundle",
                        "rules": Path(prof["rules"]).name if prof.get("rules") else None,
                        "guards": prof.get("guards"), "lookahead": prof.get("goal_lookahead"),
                        "blind": [prof.get("rear_blind_deg"), prof.get("rear_blind_range_m")],
                        "bundle_ok": (Path(prof.get("bundle") or ROOT / "bundle") / "MANIFEST.json").exists()})
    return out


def session_summary(d: Path, day: Path):
    info = {}
    try:
        info = json.loads((d / "session.json").read_text())
    except (OSError, ValueError):
        pass
    rows = A.load(d / "attempts.jsonl")
    rep = {}
    try:
        rep = json.loads((d / "report.json").read_text())
    except (OSError, ValueError):
        pass
    prof = info.get("profile") or {}
    return {"name": d.name, "dir": str(d), "mode": info.get("mode"), "started": info.get("started"),
            "profile": prof.get("name") if isinstance(prof, dict) else None, "identity": (info.get("deploy_identity") or {}).get("short"),
            "task": Path(info["task"]).parent.name if info.get("task") else None, "note": info.get("note"),
            "attempts": [{k: r.get(k) for k in ("n", "result", "tag", "note", "duration_s", "walked_m", "max_tilt",
                                                "goal_rel", "profile")} for r in rows],
            "report": (d / "report.md").exists(), "flags": rep.get("flags") or [],
            "studio": studio_id(day, d), "ended": (d / "console_summary.json").exists(),
            "scene_scan": (d / "scene_scan.txt").read_text().strip() if (d / "scene_scan.txt").exists() else None,
            "rehearsal": json.loads((d / "rehearsal.json").read_text()) if (d / "rehearsal.json").exists() else None}


def _git(*args, cwd=ROOT):
    try:
        r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def kit_version():
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {"kit_commit": _git("rev-parse", "HEAD"), "kit_dirty": bool(dirty) if dirty is not None else None,
            "private_commit": _git("rev-parse", "HEAD", cwd=ROOT / "private") if (ROOT / "private").exists() else None}


def console_supports(flag: str) -> bool:
    """Does this kit's console take `flag` (the hub passes newer options only to a console that has them)?"""
    try:
        return f'"{flag}"' in (ROOT / "g1deploy/console.py").read_text()
    except OSError:
        return False


def _tail_line(path: Path, *markers, max_bytes=16384):
    """The last line of a log holding every marker, and the log's mtime: (line, mtime) or (None, None)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            text = f.read().decode(errors="replace")
        mtime = path.stat().st_mtime
    except OSError:
        return None, None
    for line in reversed(text.splitlines()):
        if all(m in line for m in markers):
            return line, mtime
    return None, mtime


def lidar_stats(path=None):
    """The livox adapter's last stats line (`g1 lidar start` runs it with --stats-s 10): frames/s and its time."""
    line, mtime = _tail_line(path or LIDAR_LOG, "[livox_to_utlidar]", "frames/s")
    m = re.search(r"([\d.]+) frames/s", line or "")
    return {"hz": float(m.group(1)) if m else None, "at": mtime if m else None, "line": line}


def lio_stats(path=None):
    """The LIO bridge's last stats line (`g1 lio start`, every 5 s): its mean LIO pose age."""
    line, mtime = _tail_line(path or LIO_LOG, "[dds_lio]", "LIO odom in")
    m = re.search(r"LIO pose age (\d+) ms", line or "")
    return {"pose_age_s": int(m.group(1)) / 1000.0 if m else None, "at": mtime if line else None, "line": line}


_MODEL_JOINTS = {}


def model_joint_names(n_qpos):
    """Hinge joint names of the sim model whose qpos has `n_qpos` entries (free joint 7 + one per hinge): the 43-DoF G1
    with Dex3 hands (50) or the 29-DoF one (36), from the kit's MuJoCo models in sim/assets."""
    if n_qpos not in _MODEL_JOINTS:
        for xml in ("g1_29dof_dex3.xml", "g1_29dof.xml"):
            try:
                text = (ROOT / "sim/assets" / xml).read_text()
            except OSError:
                continue
            names = [n for n in re.findall(r'<joint\s[^>]*?name="([^"]+)"', text) if n != "floating_base_joint"]
            _MODEL_JOINTS.setdefault(7 + len(names), names)
    names = _MODEL_JOINTS.get(n_qpos)
    if names is None:
        raise ValueError(f"no sim model with {n_qpos} qpos entries")
    return names


def qpos36(qpos, n_qpos=None):
    """Sim qpos rows (free joint + every hinge, the model's order) -> 36-D rows: pelvis xyz, quat wxyz and the 29
    body joints in contract.MJ_JOINTS order, picked by name (the Dex3 hand joints sit between the arms)."""
    import numpy as np

    q = np.asarray(qpos, dtype=float)
    q = q.reshape(-1, q.shape[-1])
    names = model_joint_names(n_qpos or q.shape[1])
    idx = [7 + names.index(j) for j in contract.MJ_JOINTS]
    return np.concatenate([q[:, :7], q[:, idx]], axis=1)


class Adopted:
    """A console the hub did not start in this run (it restarted while the session went on): poll() by pid."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        if self.returncode is None and not Path(f"/proc/{self.pid}").exists():
            self.returncode = 0
        return self.returncode


class ViewPoller:
    """One 5 Hz poller of a live node's `view` (its control socket, REQ), whatever the number of clients: keeps the
    last reply and the cloud and memory parts by seq; serves deltas by each client's cursors. A node without `view`
    (it answers {"queued": "view"}) is asked again only every VIEW_RETRY_S."""

    def __init__(self, port, hz=VIEW_HZ, max_points=4000):
        self.port, self.period, self.max_points = int(port), 1.0 / hz, max_points
        self.base = None  # the last reply without its parts
        self.cloud = self.memory = None
        self.t_recv = None  # hub monotonic of the last good reply
        self.supported = None
        self.error = None
        self.polls = 0
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()

    def _socket(self, ctx):
        import zmq

        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://127.0.0.1:{self.port}")
        return s

    def loop(self):
        import zmq

        ctx = zmq.Context.instance()
        sock = self._socket(ctx)
        try:
            while not self.stop_flag.is_set():
                t0 = time.monotonic()
                req = {"cmd": "view", "cloud_after": self.cloud["seq"] if self.cloud else -1,
                       "mem_after": self.memory["seq"] if self.memory else -1, "max_points": self.max_points}
                try:
                    sock.send_string(json.dumps(req))
                    if not sock.poll(int(VIEW_TIMEOUT_S * 1000)):
                        raise TimeoutError("no reply in 0.5 s")
                    reply = json.loads(sock.recv())
                except Exception as exc:  # a REQ socket that missed its reply is stuck: start over
                    sock.close(0)
                    sock = self._socket(ctx)
                    with self.lock:
                        self.error = f"{type(exc).__name__}: {exc}"
                    self.stop_flag.wait(max(0.0, self.period - (time.monotonic() - t0)))
                    continue
                self.polls += 1
                if "queued" in reply or ("ok" in reply and not reply.get("ok") and "t_mono" not in reply):
                    with self.lock:  # an older node: it queued (and ignored) the unknown command
                        self.supported, self.error = False, reply.get("error") or "this node has no view command"
                    self.stop_flag.wait(VIEW_RETRY_S)
                    continue
                with self.lock:
                    self.supported, self.error = True, None
                    cl, mem = reply.pop("cloud", None), reply.pop("memory", None)
                    if cl is not None and (self.cloud is None or cl.get("seq", -1) != self.cloud.get("seq")):
                        self.cloud = cl
                    if mem is not None and (self.memory is None or mem.get("seq", -1) != self.memory.get("seq")):
                        self.memory = mem
                    self.base, self.t_recv = reply, time.monotonic()
                self.stop_flag.wait(max(0.0, self.period - (time.monotonic() - t0)))
        finally:
            sock.close(0)

    def view(self, cloud_after=-1, mem_after=-1):
        """The cached view with the parts newer than the client's cursors (None when it has them)."""
        with self.lock:
            if self.base is None:
                return None
            out = dict(self.base)
            out["cloud"] = self.cloud if self.cloud and self.cloud.get("seq", -1) > cloud_after else None
            out["memory"] = self.memory if self.memory and self.memory.get("seq", -1) > mem_after else None
            t_mono = out.get("t_mono")
            out["served_age_s"] = (time.monotonic() - t_mono) if isinstance(t_mono, (int, float)) else None
            out["received_age_s"] = time.monotonic() - self.t_recv
            return out

    def status(self):
        with self.lock:
            return {"port": self.port, "supported": self.supported, "error": self.error, "polls": self.polls,
                    "age_s": None if self.t_recv is None else round(time.monotonic() - self.t_recv, 2)}


class Hub:
    def __init__(self, args):
        self.args = args
        self.day = Path(args.day).resolve()
        self.day.mkdir(parents=True, exist_ok=True)
        self.token = panel_token()
        # one robot session and one sim session (a rehearsal or a practice session) at a time, side by side (the sim
        # on its own ports)
        self.slots = {"robot": {"proc": None, "info": None}, "sim": {"proc": None, "info": None}}
        self.lock = threading.Lock()
        self.doctor = {"running": False, "output": None, "t": None, "exit": None}
        try:
            self.doctor.update(json.loads((self.day / "doctor.json").read_text()), running=False)
        except (OSError, ValueError):
            pass
        self.scan_job = {"running": False, "name": None, "error": None, "t": None, "then": None}
        self.scans_dir = self.day / "scans"
        self.twinned = set()  # robot sessions whose final map already has a twin
        self.pair_cache = {}  # (exact, drifted) rehearsal names -> rehearsal_pairs row
        self.panel_tokens = {}  # session dir -> the token in its panel_url.txt (adopted consoles)
        self.console_state = {}  # slot -> (hub monotonic, the console's full /state)
        self.console_seen = {}  # slot -> console monotonic of its last /state (or the hub's at receive)
        self.checks = {}  # slot -> the last start check (with its session)
        self.views = {}  # slot -> (session dir, ViewPoller)
        self.set_job = None  # the rehearsal set running (or the last one)
        self.practice_proc = None
        self.summaries = {}  # session dir -> (key, summary)
        self.scene_cache = {}  # scan -> scene dict
        self.cache = None  # the v1 state, computed in the background
        self.cache_t = 0.0
        self.goal_seq = int(time.time() * 10) % 1_000_000_000  # goal_odom seq: increasing across hub restarts
        self._version = (0.0, None)
        self._profiles = (0.0, None)
        self._gpu = (0.0, None)
        self._link = (0.0, None)
        self.adopt()
        self._pc = (0.0, None)
        self.studio_link = None if getattr(args, "no_studio_link", False) else self.link_studio()
        self.bg_stop = threading.Event()
        self.bg_wake = threading.Event()  # set after an action: the state is computed again now

    def adopt(self):
        """Re-attach to the consoles still running (the hub restarted under them): any session of a lab-day-*
        directory beside this hub's day (a hub restarted after midnight keeps yesterday's live session), with the
        token from its panel_url.txt."""
        found = {}
        for pid, cmd in machine._procs({os.getpid()}):
            m = re.search(r"g1deploy/console\.py (sim|robot)\b.*--record (\S+)", cmd)
            if not m:
                continue
            d = Path(m.group(2)).resolve()
            if not (d.parent == self.day or (d.parent.parent == self.day.parent and d.parent.name.startswith("lab-day-"))):
                continue
            try:
                started = d.stat().st_mtime
            except OSError:
                continue
            if m.group(1) in found and found[m.group(1)][0] >= started:
                continue
            found[m.group(1)] = (started, pid, cmd, d)
        for slot, (started, pid, cmd, d) in found.items():
            port = re.search(r"--panel (\d+)", cmd)
            prof = re.search(r"--profile (\S+)", cmd)
            task = re.search(r"--task (\S+)", cmd)
            base = re.search(r"--port-base (\d+)", cmd)
            odom = re.search(r"--odom (\S+)", cmd)
            course = ("scan:" + Path(task.group(1)).parents[1].name) if task and "/scans/" in task.group(1) else None
            try:
                tok = parse_qs(urlparse((d / "panel_url.txt").read_text().strip()).query).get("t", [None])[0]
            except OSError:
                tok = None
            if tok:
                self.panel_tokens[str(d)] = tok
            self.slots[slot] = {"proc": Adopted(pid), "info": {
                "pid": pid, "mode": slot, "profile": prof.group(1) if prof else None, "dir": str(d), "cmd": cmd,
                "started": started, "panel_port": int(port.group(1)) if port else PANEL_PORTS[slot],
                "port_base": int(base.group(1)) if base else 5556, "course": course, "adopted": True,
                "odom": odom.group(1) if odom else None, "rehearsal": "--script" in cmd,
                "practice": d.name.startswith("practice-")}}
            print(f"g1 lab: re-attached to the running {slot} session {d.name} (pid {pid})", flush=True)

    def live(self, slot):
        """The running session of a slot (robot | sim), or None."""
        s = self.slots[slot]
        return s["info"] if s["proc"] is not None and s["proc"].poll() is None else None

    def walking(self, slot="robot"):
        """The live robot (or the slot's robot) is walking (or about to): no rehearsal, scan or stop beside it then."""
        cur = self.live(slot)
        ps = self.panel_state(cur) if cur else None
        return bool(ps and ps.get("phase") in ("PRESCAN", "RUN"))

    def _token_for(self, cur):
        return self.panel_tokens.get(cur.get("dir")) or self.token

    def panel_proxy(self, method, path, query="", body=None, slot="robot", timeout=3.0):
        """A running session's panel, through the hub (same origin as the page that embeds it): /panel/ is the robot
        session's, /sim-panel/ the rehearsal's."""
        cur = self.live(slot)
        if cur is None:
            return 404, b'{"error": "no session running"}', "application/json"
        prefix = "/panel" if slot == "robot" else "/sim-panel"
        if path == "/":  # this kit's current page (a console started earlier serves the page of its own start)
            page = (Path(__file__).with_name("panel.html")).read_text()
            page = page.replace("__TOKEN__", self._token_for(cur)).replace("__HUB_PORT__", "")
            for p_ in ("/act?t=", "/state?t=", "/map?t="):  # its requests go back through the hub
                page = page.replace(f"'{p_}", f"'{prefix}{p_}")
            return 200, page.encode(), "text/html; charset=utf-8"
        url = f"http://127.0.0.1:{cur['panel_port']}{path}?t={self._token_for(cur)}"
        req = urllib.request.Request(url, data=body, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data, ctype = r.read(), r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), "application/json"
        except Exception as exc:
            return 502, json.dumps({"error": f"panel unreachable: {exc}"}).encode(), "application/json"
        return 200, data, ctype

    # -- Studio ----------------------------------------------------------------------------------------
    def link_studio(self):
        """The day directory under Studio's robot-sessions (Studio lists sessions one level down)."""
        base = STUDIO_DATA / "robot-sessions"
        if not base.is_dir():
            return None
        link = base / self.day.name
        try:
            if link.is_symlink() and link.resolve() != self.day:
                link.unlink()
            if not link.exists():
                link.symlink_to(self.day, target_is_directory=True)
            return str(link)
        except OSError:
            return None

    # -- status ----------------------------------------------------------------------------------------
    def pc(self):
        t, snap = self._pc
        if time.monotonic() - t > 5.0 or snap is None:
            snap = machine.snapshot(os.getpid())
            own = {i["pid"] for i in (self.live("robot"), self.live("sim")) if i}
            if own:  # the hub's own sessions (their consoles and everything under them) are not "other" deploy chains
                snap["chains"] = [c for c in snap["chains"] if not own & machine._ancestors(c["pid"])]
            snap["busy"] = machine.busy_reasons(snap)
            self._pc = (time.monotonic(), snap)
        return snap

    def robot_link(self):
        iface = self.args.iface
        try:
            carrier = (Path(f"/sys/class/net/{iface}/carrier").read_text().strip() == "1")
        except OSError:
            carrier = False
        livox = subprocess.run(["pgrep", "-f", "^[^ ]*livox_ros_driver2_node"], capture_output=True).returncode == 0
        lio = subprocess.run(["pgrep", "-f", "^[^ ]*python[^ ]* -u [^ ]*adapters/dds_lio_bridge.py"],
                             capture_output=True).returncode == 0  # `g1 lio start`: FAST-LIO + its bridge on :5561
        return {"iface": iface, "carrier": carrier, "livox_driver": livox, "lio_bridge": lio, "robot": self.args.robot}

    def robot_link_v1(self):
        """robot_link plus the lidar rate (the adapter's stats log, stale after 30 s) and the LIO pose age."""
        t, cached = self._link
        if cached is not None and time.monotonic() - t < 2.0:
            return cached
        link = self.robot_link()
        now = time.time()
        ls = lidar_stats()
        fresh = ls["at"] is not None and now - ls["at"] < 30.0 and link["livox_driver"]
        lo = lio_stats()
        lio_fresh = lo["at"] is not None and now - lo["at"] < 15.0 and link["lio_bridge"]
        out = dict(link, lidar_hz=ls["hz"] if fresh else None, lidar_at=ls["at"] if fresh else None,
                   lio_pose_age_s=lo["pose_age_s"] if lio_fresh else None, lio_at=lo["at"] if lio_fresh else None)
        self._link = (time.monotonic(), out)
        return out

    def _console_full_state(self, cur, slot=None, timeout=1.0):
        """The console's /state (None when it does not answer); remembered for readiness and its heartbeat."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{cur['panel_port']}/state?t={self._token_for(cur)}",
                                        timeout=timeout) as r:
                s = json.loads(r.read())
        except Exception:
            return None
        if slot is not None:
            self.console_state[slot] = (time.monotonic(), s)
            cm = s.get("console_t_mono")
            self.console_seen[slot] = float(cm) if isinstance(cm, (int, float)) else time.monotonic()
        return s

    def running(self):
        """Both slots: the session's info, running or the last one ended, with its panel's state."""
        out = {}
        for slot in ("robot", "sim"):
            twin = ended = None
            with self.lock:
                sl = self.slots[slot]
                if sl["proc"] is not None and sl["proc"].poll() is not None:
                    sl["info"].update(exit=getattr(sl["proc"], "returncode", None), ended=time.time())
                    sl["proc"] = None
                    d = sl["info"]["dir"]
                    ended = d
                    if slot == "robot" and d not in self.twinned and (Path(d) / "node_memory.npz").exists():
                        self.twinned.add(d)  # what the robot saw on its walks, as a twin too (no map before Arm)
                        twin = d
                info = dict(sl["info"]) if sl["info"] else None
                live = sl["proc"] is not None
            if ended:
                self.session_ended(slot, ended)
            if twin:
                self.scan(source_session=twin)
            if info:
                info["running"] = live
                if live:
                    info["panel_ready"] = (Path(info["dir"]) / "panel_url.txt").exists()
                    info["panel"] = self.panel_state(info, slot=slot)
            out[slot] = info
        return out

    def session_ended(self, slot, d):
        """A session of this hub ended: its evidence manifest, its live view stops."""
        try:
            evidence.write(d)
        except Exception as exc:
            print(f"g1 lab: manifest of {d} failed: {exc}", flush=True)
        v = self.views.pop(slot, None)
        if v is not None:
            v[1].stop()

    def panel_state(self, cur, slot=None):
        s = self._console_full_state(cur, slot=slot)
        if s is None:
            return None
        try:
            now = next((x for x in s.get("ladder", []) if x["state"] == "now"), None)
            temps = s.get("temps") or {}
            return {"runner": s.get("runner"), "phase": s.get("phase"), "next": now and now["label"],
                    "attempts": len(s.get("attempts", [])), "interlock": "rehearsal_running" in s,
                    "rehearsal_running": s.get("rehearsal_running"), "goal_rel": s.get("goal_rel"),
                    "xy": (s.get("node") or {}).get("xy") or s.get("xy"), "driving": s.get("driving"),
                    "go_locks": s.get("go_locks"), "remote": {"connected": (s.get("remote") or {}).get("connected")},
                    "temps": {"max_c": temps.get("max_c"), "max_joint": temps.get("max_joint")},
                    "standing_s": s.get("standing_s"), "node_ready": s.get("node_ready"), "odom": s.get("odom")}
        except Exception:
            return None

    def state(self):
        sessions = sorted((d for d in self.day.iterdir() if d.is_dir() and (d / "session.json").exists()),
                          key=lambda d: d.stat().st_mtime, reverse=True)
        pc = self.pc()
        du = shutil.disk_usage(self.day)
        live = self.running()
        live_pub = {k: ({kk: vv for kk, vv in v.items()} if v else None) for k, v in live.items()}
        return {"day": str(self.day), "token_hint": self.token[:3] + "...", "pc": {
                    "load": pc.get("load"), "cpus": pc.get("cpus"), "gpu": pc.get("gpu"), "busy": pc.get("busy"),
                    "chains": pc.get("chains"), "disk_free_gb": round(du.free / 1e9, 1)},
                "robot_link": self.robot_link(), "profiles": list_profiles(), "courses": [c for c in REHEARSAL_COURSES
                                                                                         if (ROOT / "sim/courses" / c).is_dir()],
                "live": live_pub, "current": self.current_view(live_pub),
                "sessions": (summ := [session_summary(d, self.day) for d in sessions[:40]]), "pairs": self.rehearsal_pairs(summ),
                "studio": {"port": STUDIO_PORT, "linked": self.studio_link}, "doctor": self.doctor,
                "panel_ports": PANEL_PORTS, "default_robot": self.args.robot, "plan": self.plan(),
                "scans": self.list_scans(), "scan_job": dict(self.scan_job, then=bool(self.scan_job.get("then"))),
                "default_profile": self.default_profile(), "picker": self.picker(),
                "gpu": {"free_mib": gpu_free_mib() if self.live("robot") else None, "rehearsal_mib": REHEARSAL_GPU_MIB}}

    @staticmethod
    def current_view(live):
        """The page's main session: the live robot session, else the live rehearsal, else the last ended one."""
        for slot in ("robot", "sim"):
            if live.get(slot) and live[slot]["running"]:
                return live[slot]
        ended = [v for v in live.values() if v]
        return max(ended, key=lambda v: v.get("ended") or 0) if ended else None

    def picker(self):
        """The policy picker's first-class profiles, their display names and what each changes: private/lab_hub.json
        {"featured": [names], "labels": {name: text}, "notes": {name: text}}, else the baseline and the default."""
        try:
            cfg = json.loads((ROOT / "private/lab_hub.json").read_text())
            return {"featured": [str(n) for n in cfg.get("featured") or []], "labels": dict(cfg.get("labels") or {}),
                    "notes": dict(cfg.get("notes") or {})}
        except (OSError, ValueError, TypeError):
            return {"featured": [], "labels": {}, "notes": {}}

    def default_profile(self):
        """The profile the page selects first: private/lab_hub.json {"default_profile": NAME}, else `deploy`."""
        try:
            return str(json.loads((ROOT / "private/lab_hub.json").read_text())["default_profile"])
        except (OSError, ValueError, KeyError, TypeError):
            return "deploy"

    def robot_profile(self):
        try:
            return json.loads((ROOT / "robots" / f"{self.args.robot}.json").read_text())
        except (OSError, ValueError):
            return {}

    # -- the real scene ----------------------------------------------------------------------------------
    def list_scans(self, limit=20):
        out = []
        for d in sorted(self.scans_dir.glob("*/scan.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                x = json.loads(d.read_text())
            except (OSError, ValueError):
                continue
            x["walked_since"] = self.walked_since(d.stat().st_mtime)
            # the site check judged the goal typed at scan time too; the page judges the goal picked now, so the scan
            # itself is only about where the robot stands (and the planner's merged boxes, for the record)
            reasons = x.get("reasons") or []
            x["start_blocked"] = [r for r in reasons if r.startswith("start inside")]
            x["notes"] = [r for r in reasons if not (r.startswith("start inside") or "goal" in r or "route" in r)]
            x["mtime"] = d.stat().st_mtime
            out.append(x)
        return out

    def walked_since(self, t):
        """Robot sessions started after time `t` whose robot walked: a scan from before them no longer starts where
        the robot stands (goals are relative to the robot's pose at the session start)."""
        out = []
        for d in self.day.glob("robot-*"):
            if (d / "session.json").exists() and (d / "session.json").stat().st_mtime > t:
                walked = sum((a.get("walked_m") or 0.0) for a in A.load(d / "attempts.jsonl"))
                if walked > 0.3:
                    out.append({"session": d.name, "walked_m": round(walked, 2)})
        return out

    def scan(self, goal=None, seconds=8.0, then=None, source_session=None):
        """tools/scan_scene.py in the background; `then`: a robot start request to launch when the scan is done
        (whatever the scan found: the panel's own checks still gate GO)."""
        if self.scan_job["running"]:
            return {"ok": False, "error": "a scan is running"}
        if source_session is None:
            if self.live("sim"):
                return {"ok": False, "error": "a rehearsal is running: scan when it ends (about a minute)"}
            if self.walking():
                return {"ok": False, "error": "the robot is walking: scan when it stands"}
            if not self.robot_link()["carrier"]:
                return {"ok": False, "error": f"no link on {self.args.iface}: robot off or cable out"}
        name = f"session-{Path(source_session).name}" if source_session else f"scan-{time.strftime('%H%M%S')}"
        cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/scan_scene.py"), "--out", str(self.scans_dir / name)]
        if source_session:
            cmd += ["--from-session", str(source_session)]
        else:
            cmd += ["--robot", self.args.robot, "--goal", *(str(float(v)) for v in (goal or [2.0, 0.0])),
                    "--seconds", str(float(seconds)), "--live-out", str(self.scans_dir / name / "live.json")]
        self.scan_job = {"running": True, "name": name, "error": None, "t": time.time(), "then": then}

        def go():
            (self.scans_dir / name).mkdir(parents=True, exist_ok=True)
            try:
                r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=240,
                                   stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONPATH=str(ROOT)))
                err = None if r.returncode == 0 else (r.stderr or r.stdout)[-500:]
            except Exception as exc:
                err = f"scan failed: {exc}"
            then_req = self.scan_job.get("then")
            self.scan_job = {"running": False, "name": name, "error": err, "t": time.time(), "then": None}
            if then_req is not None:
                then_req["scan"] = False
                then_req["scene_scan"] = name
                res = self.start(then_req)
                if not res.get("ok"):
                    self.scan_job["error"] = "scan done, session not started: " + res.get("error", "")
        threading.Thread(target=go, daemon=True).start()
        return {"ok": True, "name": name}

    def scan_grid(self, name):
        """The scan's lidar memory around where the robot stood (workspace.code_grid, 10 cm cells, 5 m each way)."""
        f = self.scan_file(name, "memory.npz")
        if f is None:
            return None
        import numpy as np

        with np.load(f, allow_pickle=False) as z:
            start = [float(v) for v in z["start_xy"]] if "start_xy" in z.files else [0.0, 0.0]
            yaw = float(z["start_yaw"]) if "start_yaw" in z.files else 0.0
            g = workspace.code_grid(z["support"], z["ceiling"], z["lo"], float(z["res"]), start, 5.0, 0.1)
        return dict(g, start=[start[0], start[1], yaw])

    def start_check(self, scan, goal=None):
        """The live robot now vs where it stood for `scan` (g1deploy/registration.py): its move since the scan and the
        scan's goal from the robot now. On 2026-10-27 every walk started 16-31 deg right of its scan's heading and
        0.16-0.26 m off (the robot hangs in the gantry until the runner takes over)."""
        if not self.live("robot"):
            return {"ok": False, "error": "no robot session is live"}
        grid = self.scan_grid(scan) if scan else None
        if grid is None:
            return {"ok": False, "error": "pick the scan the goal was placed on"}
        code, data, _ = self.panel_proxy("GET", "/map")
        try:
            live = json.loads(data) if code == 200 else None
        except ValueError:
            live = None
        out = registration.start_check(grid, live, goal)
        return dict(out, scan=scan, goal_scan=goal, at=time.time())  # `t` stays the transform (A1)

    def record_check(self, check, slot="robot"):
        cur = self.live(slot)
        if cur is not None:
            with open(Path(cur["dir"]) / "start_check.jsonl", "a") as f:
                f.write(wire.dumps(check) + "\n")

    def trajectory(self, name):
        """A sim session in a twin, in the start frame (ahead = x, left = y): the walked pelvis path, every plan's
        pelvis path, contacts and the attempt result."""
        d = (self.day / name).resolve()
        if d.parent != self.day:
            return None
        import numpy as np

        out = {"name": name, "walked": None, "plans": [], "contact_max_n": None,
               "attempts": A.load(d / "attempts.jsonl"), "running": not (d / "console_summary.json").exists()}
        try:
            with np.load(d / "sim.npz", allow_pickle=False) as z:
                rp = z["root_pos"]
                step = max(1, len(rp) // 400)
                out["walked"] = np.round(rp[::step, :2], 3).tolist()
                out["contact_max_n"] = round(float(np.max(z["contact_n"])), 1) if len(z["contact_n"]) else 0.0
        except (OSError, KeyError, ValueError):
            pass
        for f in sorted((d / "plans").glob("plan_*_out.npz"))[:200]:
            try:
                with np.load(f, allow_pickle=False) as z:
                    out["plans"].append(np.round(z["qpos"][::4, :2], 3).tolist())
            except (OSError, KeyError, ValueError):
                continue
        try:  # a rehearsal from where the robot stood: back into the scan's frame, where the page draws
            reh = json.loads((d / "rehearsal.json").read_text())
        except (OSError, ValueError):
            reh = {}
        fr = reh.get("from_robot")
        if fr:
            back = lambda xy: np.round(registration.rotate(np.asarray(xy, float).reshape(-1, 2) - fr["t"], -fr["theta_rad"]), 3).tolist()  # noqa: E731
            out["walked"] = back(out["walked"]) if out["walked"] else out["walked"]
            out["plans"] = [back(pl) for pl in out["plans"]]
        if reh.get("scene_from_course"):  # a start-offset run of a rehearsal set: its course back into the scene
            s_from_c = reh["scene_from_course"]
            back = lambda xy: np.round(frames.apply(s_from_c, np.asarray(xy, float).reshape(-1, 2)), 3).tolist()  # noqa: E731
            out["walked"] = back(out["walked"]) if out["walked"] else out["walked"]
            out["plans"] = [back(pl) for pl in out["plans"]]
        if reh.get("drift"):
            out["plans"] = []  # planned in the drifted odometry's frame, not the scene's
        return out

    def rehearse(self, req):
        """The automatic rehearsal of a goal in a scanned twin: `g1 sim --task TWIN --goal A L --script ...`, then the
        same with odometry drift (DRIFT) when it ends (`pair`, default on). (The legacy page's; API v1 runs a set of
        three: rehearse_set.)
        `from_robot`: from where the live robot stands now (start_check), not from the scan's pose: the twin moved
        into the robot's frame and the goal sent as the same spot (needs the GPU room beside the live session)."""
        scan, goal = req.get("scan"), req.get("goal")
        if not scan or not goal:
            return {"ok": False, "error": "pick a scan and a goal first"}
        goal = [float(goal[0]), float(goal[1])]
        r = dict(req, mode="sim", course="scan:" + scan, script=REHEARSE_SCRIPT, label="rehearse" + ("-drift" if req.get("drift") else ""))
        meta = {"scan": scan, "goal": goal, "profile": req.get("profile"), "drift": DRIFT if req.get("drift") else None}
        if req.get("from_robot"):
            check = req.get("check") or self.start_check(scan, goal)
            if not check.get("ok"):
                return {"ok": False, "error": check.get("error") or "no start check"}
            src = self.scan_file(scan, "course/task.json")
            task = registration.repose_task(json.loads(src.read_text()), check["theta_rad"], check["t"])
            f = self.day / "reposed" / f"{scan}-{time.strftime('%H%M%S')}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(task, indent=1) + "\n")
            r.update(task_file=str(f), goal=check["goal_now"], label=r["label"] + "-from-robot")
            meta.update(from_robot={k: check[k] for k in ("theta_rad", "t", "moved_m", "turned_deg", "goal_now")})
        res = self.start(r)
        if res.get("ok"):
            (Path(res["dir"]) / "rehearsal.json").write_text(json.dumps(meta) + "\n")
            if req.get("pair", True) and not req.get("drift"):
                then = dict(req, drift=True, pair=False, check=req.get("check") or (meta.get("from_robot") and check))
                threading.Thread(target=self._then_drift, args=(res["dir"], then), daemon=True).start()
        return res

    def _then_drift(self, first_dir, req):
        """The drifted rehearsal once the exact one has ended (whatever its result)."""
        while self.live("sim") and self.live("sim").get("dir") == first_dir:
            time.sleep(1.0)
        res = self.rehearse(req)
        if not res.get("ok"):
            f = Path(first_dir) / "rehearsal.json"
            meta = json.loads(f.read_text())
            f.write_text(json.dumps(dict(meta, drift_error=res.get("error"))) + "\n")

    def rehearsal_pairs(self, sessions):
        """Exact / drifted rehearsals of the same scan, goal, profile and start: did the drifted one arrive the same
        way? (sim.npz walked paths; OTHER_WAY_M)."""
        def key(x):
            r = x["rehearsal"]
            return (r["scan"], tuple(r["goal"]), r.get("profile"), json.dumps(r.get("from_robot"), sort_keys=True))
        exact = {key(x): x for x in sessions if x.get("rehearsal") and not x["rehearsal"].get("drift")
                 and not x["rehearsal"].get("set") and x["ended"]}
        out = {}
        for x in sessions:
            r = x.get("rehearsal")
            if not r or not r.get("drift") or r.get("set") or not x["ended"] or key(x) not in exact:
                continue
            e = exact[key(x)]
            if (e["name"], x["name"]) in self.pair_cache:  # ended sessions do not change
                out[e["name"]] = self.pair_cache[(e["name"], x["name"])]
                continue
            out[e["name"]] = self.pair_cache[(e["name"], x["name"])] = self.pair_verdict(e, x)
        return out

    def pair_verdict(self, e, x):
        """The exact run `e` and the drifted run `x` (session summaries): did the drifted one arrive the same way?"""
        import numpy as np

        res = lambda s: (s["attempts"][-1].get("result") if s["attempts"] else None)  # noqa: E731
        way = None
        try:
            with np.load(Path(e["dir"]) / "sim.npz") as a, np.load(Path(x["dir"]) / "sim.npz") as b:
                pa, pb = a["root_pos"][::10, :2], b["root_pos"][::10, :2]
            way = round(float(np.max(np.min(np.linalg.norm(pb[:, None] - pa[None], axis=2), axis=1))), 2)
        except (OSError, KeyError, ValueError):
            pass
        verdict = ("robust" if res(x) == "arrived" and res(e) == "arrived" and (way is None or way <= OTHER_WAY_M)
                   else "other way" if res(x) == "arrived" and res(e) == "arrived" else
                   "fails with drift" if res(e) == "arrived" else "fails")
        return {"drift": x["name"], "exact_result": res(e), "drift_result": res(x), "apart_m": way, "verdict": verdict}

    def scan_file(self, name, rel):
        if rel not in ("site.png", "twin.json", "scan.json", "site.json", "course/task.json", "memory.npz", "live.json"):
            return None
        p = (self.scans_dir / str(name) / rel).resolve()
        if self.scans_dir.resolve() not in p.parents or not p.is_file():
            return None
        return p

    def plan(self):
        """The day's plan from the private overlay (private/lab_plan.json: [[step, detail], ...]), else the page's own."""
        try:
            return [[str(a), str(b)] for a, b in json.loads((ROOT / "private/lab_plan.json").read_text())]
        except (OSError, ValueError, TypeError):
            return None

    # -- actions ---------------------------------------------------------------------------------------
    def command(self, req):
        """The console command line for a start request (also what the page shows before starting)."""
        mode = req.get("mode")
        if mode not in ("sim", "robot"):
            raise ValueError("mode must be sim or robot")
        prof = req.get("profile")
        profiles.find(prof)  # raises ProfileError
        stamp = time.strftime("%H%M%S")
        name = f"{req.get('label') or mode}-{stamp}-{re.sub(r'[^a-zA-Z0-9_-]', '-', prof)}"
        if mode == "sim":
            name += "-" + re.sub(r"[^a-zA-Z0-9_-]", "-", (req.get("course") or "lab-door-real").replace("scan:", "twin-"))
        out = self.day / name
        cmd = [str(ROOT / "g1"), mode, "--profile", prof, "--arm", "--panel", str(PANEL_PORTS[mode]),
               "--record", str(out), "--busy", "warn"]
        if mode == "sim":
            course = req.get("course") or "lab-door-real"
            if course.startswith("scan:"):  # a twin of a scan of today's real scene (tools/scan_scene.py)
                task = (self.scans_dir / course[5:] / "course" / "task.json").resolve()
                if req.get("task_file"):  # the twin moved to where the robot stands (a start offset, from_robot)
                    task = Path(req["task_file"]).resolve()
                    if (self.day / "reposed").resolve() not in task.parents:
                        raise ValueError("a reposed course lives in the day's reposed/")
                if self.scans_dir.resolve() not in task.parents and not req.get("task_file") or not task.is_file():
                    raise ValueError(f"no scanned twin {course[5:]}")
                cmd += ["--task", str(task)]
                if req.get("drift"):
                    cmd += ["--odom", "lio", "--lio-noise", json.dumps(dict(DRIFT, seed=int(time.time()) % 10000))]
            elif not (ROOT / "sim/courses" / course).is_dir():
                raise ValueError(f"no course {course}")
            else:
                cmd += ["--layout", course]
            if req.get("sim_odom") and not req.get("drift"):  # a practice session: the robot's odometry choice
                if req["sim_odom"] not in ("dds", "leg", "lio", "fused"):
                    raise ValueError("sim odometry must be dds, leg, lio or fused")
                cmd += ["--odom", req["sim_odom"]]
            if req.get("goal"):
                cmd += ["--goal", *(str(float(v)) for v in req["goal"])]
        else:
            cmd += ["--robot", req.get("robot") or self.args.robot]
            goal = req.get("goal") or [1.5, 0.0]
            cmd += ["--goal", *(str(float(v)) for v in goal)]
            if req.get("odom"):  # leg (the robot profile's default) | zmq (FAST-LIO) | fused (FAST-LIO + legs, LIO lost -> stand)
                if req["odom"] not in ("leg", "zmq", "fused"):
                    raise ValueError("odometry must be leg, zmq or fused")
                cmd += ["--odom", req["odom"]]
        if mode == "sim":  # isolated ports, low priority: it may run beside a live robot session
            cmd = ["nice", "-n", "10"] + cmd + ["--port-base", str(REHEARSAL_PORT_BASE)]
        if mode == "sim" and req.get("script"):
            cmd = [c for c in cmd if c != "--arm"] + ["--script", req["script"], "--step-timeout", "60"]
        if req.get("note"):
            cmd += ["--note", str(req["note"])[:300]]
        if req.get("operator"):
            cmd += ["--operator", str(req["operator"])[:60]]
        if req.get("confirm_json") and console_supports("--confirm"):  # the console stores it in session.json
            cmd += ["--confirm", json.dumps(req["confirm_json"], sort_keys=True)]
        return cmd, out

    def start(self, req, v1=False):
        slot = req.get("mode")
        if slot not in self.slots:
            return {"ok": False, "error": "mode must be sim or robot"}
        if self.live(slot):
            return {"ok": False, "error": ("a robot session is live: end it from its panel first" if slot == "robot"
                                           else "a sim session is running (rehearsal or practice): wait for it or "
                                                "end it")}
        job = self.set_job
        if slot == "sim" and job and job.get("running") and req.get("_set") != job["name"]:
            return {"ok": False, "error": "a rehearsal set is running: wait for its three runs"}
        if slot == "sim" and self.live("robot"):
            if req.get("practice"):
                return {"ok": False, "error": "a robot session is live: practice and robot sessions never run together"}
            ps = self.panel_state(self.live("robot")) or {}
            if not ps.get("interlock"):
                return {"ok": False, "error": "the live robot session started before rehearsals could run beside it "
                                              "(it cannot hold GO during one): end it, then rehearse, then start it again"}
            if ps.get("phase") in ("PRESCAN", "RUN"):
                return {"ok": False, "error": "the robot is walking: rehearse when it stands"}
            free = gpu_free_mib()
            if free is not None and free < REHEARSAL_GPU_MIB:
                return {"ok": False, "error": f"the GPU has {free / 1024:.1f} GB free beside the live robot session; a "
                                              f"rehearsal needs about {REHEARSAL_GPU_MIB / 1024:.1f} GB. Rehearse before "
                                              "starting the robot session, or end it (robot in the gantry) to rehearse"}
        if slot == "robot" and self.live("sim"):
            if (self.live("sim") or {}).get("practice"):
                return {"ok": False, "error": "a practice session is live: end it first (practice and robot sessions "
                                              "never run together)"}
            return {"ok": False, "error": "a rehearsal is running: start the robot session when it ends (about a minute)"}
        if req.get("mode") == "robot":
            need = CONFIRM_ROBOT if v1 else CONFIRM_LEGACY
            missing = [k for k in need if not (req.get("confirm") or {}).get(k)]
            if missing:
                return {"ok": False, "error": "confirm the robot checklist first: " + ", ".join(missing)}
            link = self.robot_link()
            if not link["carrier"]:
                return {"ok": False, "error": f"no link on {self.args.iface}: robot off or cable out"}
            if req.get("odom") in ("zmq", "fused") and not link["lio_bridge"]:
                return {"ok": False, "error": "no FAST-LIO bridge runs: `g1 lio start` and `g1 lio check` with the robot "
                                              "standing first, or pick leg odometry"}
            fresh = None
            if req.get("scene_scan"):
                f = self.scans_dir / str(req["scene_scan"]) / "scan.json"
                if (f.is_file() and time.time() - f.stat().st_mtime < SCAN_FRESH_S
                        and not self.walked_since(f.stat().st_mtime)):  # the robot has not walked off since
                    fresh = req["scene_scan"]
            if v1 and not fresh:
                return {"ok": False, "error": "scan stale: scan again from where the robot stands (POST /api/v1/scan)",
                        "scan_stale": True}
            if fresh:
                req = dict(req, scan=False)
            if req.get("scan", True):  # the scene first: a scan and its twin, then the session
                r = self.scan(goal=req.get("goal"), seconds=8.0, then=dict(req))
                return dict(r, scanning=True) if r.get("ok") else r
        try:
            cmd, out = self.command(req)
        except (ValueError, profiles.ProfileError) as exc:
            return {"ok": False, "error": str(exc)}
        out.mkdir(parents=True, exist_ok=True)
        if req.get("scene_scan"):
            (out / "scene_scan.txt").write_text(req["scene_scan"] + "\n")
        hub_meta = {k: req.get(k) for k in ("mode", "profile", "odom", "operator", "confirm", "practice", "start_offset",
                                             "scene_scan", "course", "goal")}
        (out / "hub_start.json").write_text(wire.dumps(dict(hub_meta, t=time.time(), cmd=cmd)) + "\n")
        env = dict(os.environ, G1_PANEL_TOKEN=self.token, G1_HUB_PORT=str(self.args.port))
        log = open(out / "console.out", "w")
        # its own process group and session: the console (and the runner it owns) outlive the hub
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        with self.lock:
            self.slots[slot] = {"proc": proc, "info": {
                "pid": proc.pid, "mode": slot, "profile": req["profile"], "dir": str(out), "cmd": " ".join(cmd),
                "started": time.time(), "panel_port": PANEL_PORTS[slot], "course": req.get("course"),
                "rehearsal": bool(req.get("script")), "practice": bool(req.get("practice")),
                "port_base": PORT_BASES[slot], "odom": req.get("sim_odom") or req.get("odom") or (
                    ("lio" if req.get("drift") else "dds") if slot == "sim" else self.robot_profile().get("odom", "leg")),
                "operator": req.get("operator")}}
        return {"ok": True, "dir": str(out), "name": out.name, "cmd": " ".join(cmd)}

    def stop(self, slot=None):
        live = self.running()
        slot = slot or ("robot" if live["robot"] and live["robot"]["running"] else "sim")
        cur = live.get(slot)
        if not cur or not cur.get("running"):
            return {"ok": False, "error": "no session running"}
        if cur["mode"] == "robot":
            ps = cur.get("panel") or {}
            if ps.get("runner") not in ("not started", "E-STOP"):
                return {"ok": False, "error": "a robot session with the runner up ends from its panel (End session), "
                                              "with the robot in the gantry or sitting"}
        try:
            os.killpg(cur["pid"], signal.SIGINT)  # the console shuts down its processes and writes the report
        except ProcessLookupError:
            pass
        return {"ok": True}

    def send_goal(self, goal, scan=None):
        """A new goal for the live robot session. With `scan`, `goal` is where it was placed on that scan (ahead / left
        of the robot at the scan) and is sent as the same spot in the scene, from the robot now (start_check); without,
        it is relative to the robot now (the panel's Set goal). (The legacy page's; API v1: send_goal_v1.)"""
        if not self.live("robot"):
            return {"ok": False, "error": "no robot session is live"}
        if self.walking():
            return {"ok": False, "error": "the robot is walking: send the goal when it stands"}
        a, l = (float(v) for v in goal)
        check = None
        if scan:
            check = self.start_check(scan, [a, l])
            self.record_check(dict(check, action="send_goal"))
            if not check.get("ok"):
                return {"ok": False, "check": check, "error": (check.get("error") or "no start check") +
                        "; or set the goal relative to the robot now in the robot panel"}
            a, l = check["goal_now"]
        code, data, _ = self.panel_proxy("POST", "/act", body=json.dumps({"act": "goal", "ahead": a, "left": l}).encode())
        return {"ok": code == 200, "sent": [a, l], "check": check, "error": None if code == 200 else data.decode()[:200]}

    def run_doctor(self):
        if self.doctor["running"]:
            return {"ok": False, "error": "doctor is running"}
        self.doctor = {"running": True, "output": None, "t": time.time(), "exit": None}

        def go():
            try:
                r = subprocess.run([str(ROOT / "g1"), "doctor", "--robot", self.args.robot], cwd=str(ROOT),
                                   capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)
                out, code = (r.stdout + r.stderr)[-12000:], r.returncode
            except Exception as exc:
                out, code = f"doctor failed: {exc}", None
            self.doctor = {"running": False, "output": out, "t": time.time(), "exit": code}
            try:
                (self.day / "doctor.json").write_text(wire.dumps({k: self.doctor[k] for k in ("output", "t", "exit")}))
            except OSError:
                pass
        threading.Thread(target=go, daemon=True).start()
        return {"ok": True}

    def report(self, name):
        d = (self.day / name).resolve()
        if d.parent != self.day:
            return None
        try:
            return (d / "report.md").read_text()
        except OSError:
            return None

    # ===================================================================================== API v1
    def session_dir(self, name):
        d = (self.day / str(name)).resolve()
        return d if d.parent == self.day and d.is_dir() else None

    def _summary(self, d: Path):
        """session_summary, cached until one of its files changes."""
        key = []
        for f in ("session.json", "attempts.jsonl", "report.json", "console_summary.json", "manifest.json", "rehearsal.json"):
            try:
                key.append((d / f).stat().st_mtime)
            except OSError:
                key.append(None)
        key = tuple(key)
        hit = self.summaries.get(str(d))
        if hit and hit[0] == key:
            return hit[1]
        s = session_summary(d, self.day)
        try:
            hs = json.loads((d / "hub_start.json").read_text())
        except (OSError, ValueError):
            hs = {}
        s["practice"] = bool(hs.get("practice")) or d.name.startswith("practice-")
        s["operator"] = hs.get("operator")
        try:
            s["manifest"] = {"complete": json.loads((d / "manifest.json").read_text()).get("complete")}
        except (OSError, ValueError):
            s["manifest"] = None
        self.summaries[str(d)] = (key, s)
        return s

    def all_sessions(self):
        dirs = sorted((d for d in self.day.iterdir() if d.is_dir() and (d / "session.json").exists()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
        return [self._summary(d) for d in dirs]

    def sessions_v1(self, all_s=None, offset=0, limit=SESSIONS_PAGE):
        all_s = self.all_sessions() if all_s is None else all_s
        robot = [s for s in all_s if s.get("mode") == "robot"]
        other = [s for s in all_s if s.get("mode") != "robot"]
        return {"robot": robot, "other": other[offset:offset + limit], "other_total": len(other), "offset": offset,
                "limit": limit}

    def scan_age(self, name):
        f = self.scans_dir / str(name) / "scan.json"
        try:
            return time.time() - f.stat().st_mtime
        except OSError:
            return None

    def scan_fresh(self, name):
        """(fresh, reason): younger than SCAN_FRESH_S and the robot has not walked since."""
        f = self.scans_dir / str(name) / "scan.json"
        if not name or not f.is_file():
            return False, "no such scan"
        age = time.time() - f.stat().st_mtime
        if age >= SCAN_FRESH_S:
            return False, f"scan stale: {age / 60:.0f} min old (fresh for {SCAN_FRESH_S // 60} min)"
        walked = self.walked_since(f.stat().st_mtime)
        if walked:
            return False, f"scan stale: the robot walked since ({walked[0]['session']})"
        return True, None

    @staticmethod
    def goal_free_reasons(reasons):
        """The site check's reasons that do not depend on the goal typed at scan time."""
        return [r for r in reasons or [] if not ("goal" in r or "route" in r)]

    def scans_v1(self, sets):
        out = []
        for x in self.list_scans(limit=40):
            reasons = self.goal_free_reasons(x.get("reasons"))
            verdict = x.get("verdict")
            if verdict in ("GO", "CHECK", "NO-GO"):
                verdict = "NO-GO" if any(r.startswith("start inside") for r in reasons) else "CHECK" if reasons else "GO"
            out.append({"name": x.get("name"), "time": x.get("time"), "verdict": verdict, "site_verdict": x.get("verdict"),
                        "reasons": reasons, "walked_since": x.get("walked_since"),
                        "age_s": round(time.time() - x["mtime"], 1), "fresh": self.scan_fresh(x.get("name"))[0],
                        "practice": x.get("practice"), "goal_ahead_left": x.get("goal_ahead_left"),
                        "rehearsals": [s["name"] for s in sets if s.get("scan") == x.get("name")]})
        return out

    def rehearsal_sets(self, all_s):
        """Rehearsal sets (exact, drift, offset runs of one scan + goal + profile + odometry), newest first; legacy
        exact / drift pairs as sets without an offset run."""
        sets = {}
        for s in all_s:
            r = s.get("rehearsal") or {}
            if r.get("set"):
                st = sets.setdefault(r["set"], {"name": r["set"], "scan": r.get("scan"), "goal_scene": r.get("goal_scene"),
                                                "profile": r.get("profile"), "odom": r.get("odom"), "runs": {},
                                                "summ": {}, "offset": None})
                st["runs"][r.get("kind")] = s["name"]
                st["summ"][r.get("kind")] = s
                if r.get("kind") == "offset":
                    st["offset"] = r.get("offset")
        job = self.set_job
        if job and job["name"] not in sets:
            sets[job["name"]] = {k: job.get(k) for k in ("name", "scan", "goal_scene", "profile", "odom")}
            sets[job["name"]].update(runs={}, summ={}, offset=job.get("offset"))
        out = []
        for name, st in sets.items():
            summ = st.pop("summ")
            res = {k: (summ[k]["attempts"][-1].get("result") if summ.get(k) and summ[k]["attempts"] else None)
                   for k in RUN_KINDS}
            ended = {k: bool(summ.get(k) and summ[k]["ended"]) for k in RUN_KINDS}
            running = bool(job and job["name"] == name and job.get("running"))
            verdict = apart = None
            if ended["exact"] and ended["drift"]:
                ck = (summ["exact"]["name"], summ["drift"]["name"])
                if ck not in self.pair_cache:
                    self.pair_cache[ck] = self.pair_verdict(summ["exact"], summ["drift"])
                verdict, apart = self.pair_cache[ck]["verdict"], self.pair_cache[ck]["apart_m"]
            offset_ok = (res["offset"] == "arrived") if ended["offset"] else None
            out.append(dict(st, runs={k: st["runs"].get(k) for k in RUN_KINDS}, results=res, running=running,
                            verdict=verdict, apart_m=apart, offset_ok=offset_ok,
                            error=job.get("error") if job and job["name"] == name else None,
                            current=job.get("current") if running else None,
                            envelope=self._envelope_of(st.get("offset")) if offset_ok else None))
        for e, p in self.rehearsal_pairs(all_s).items():  # the legacy page's pairs
            ex = next((s for s in all_s if s["name"] == e), None)
            r = (ex or {}).get("rehearsal") or {}
            out.append({"name": e, "scan": r.get("scan"), "goal_scene": r.get("goal") if not r.get("from_robot") else None,
                        "profile": r.get("profile"), "odom": None, "runs": {"exact": e, "drift": p["drift"], "offset": None},
                        "results": {"exact": p["exact_result"], "drift": p["drift_result"], "offset": None},
                        "running": False, "verdict": p["verdict"], "apart_m": p["apart_m"], "offset_ok": None,
                        "legacy_pair": True, "offset": None, "envelope": None})
        out.sort(key=lambda s: s["name"], reverse=True)
        return out

    @staticmethod
    def _envelope_of(offset):
        if not offset or not offset.get("pose"):
            return None
        m, deg = frames.offset_envelope(offset["pose"])
        return {"moved_m": round(m, 3), "turned_deg": round(deg, 1)}

    def offset_envelope(self, scan):
        """The start offsets the newest rehearsal set of `scan` whose offset run arrived covered: {moved_m, turned_deg,
        set} or None."""
        all_s = self.all_sessions()
        for st in self.rehearsal_sets(all_s):
            if st.get("scan") == scan and st.get("offset_ok") and st.get("envelope"):
                return dict(st["envelope"], set=st["name"])
        return None

    def profiles_v1(self):
        t, cached = self._profiles
        if cached is not None and time.monotonic() - t < 30.0:
            return cached
        pick = self.picker()
        out = []
        for p in list_profiles():
            n = p["name"]
            out.append({"name": n, "label": pick["labels"].get(n) or p.get("student") or n,
                        "featured": n in pick["featured"], "note": pick["notes"].get(n) or p.get("description") or "",
                        "bundle_ok": p.get("bundle_ok"), "error": p.get("error")})
        self._profiles = (time.monotonic(), out)
        return out

    def version(self):
        t, cached = self._version
        if cached is None or time.monotonic() - t > 60.0:
            cached = kit_version()
            self._version = (time.monotonic(), cached)
        return cached

    def gpu_v1(self):
        t, cached = self._gpu
        if cached is None or time.monotonic() - t > 5.0:
            cached = gpu_free_mib()
            self._gpu = (time.monotonic(), cached)
        return {"free_mib": cached, "rehearsal_mib": REHEARSAL_GPU_MIB}

    def live_v1(self, slot, info):
        if not info:
            return None
        d = Path(info["dir"])
        reh = None
        try:
            r = json.loads((d / "rehearsal.json").read_text())
            reh = {"set": r.get("set"), "kind": r.get("kind") or ("drift" if r.get("drift") else "exact")}
        except (OSError, ValueError):
            pass
        seen = self.console_seen.get(slot)
        v = self.views.get(slot)
        return {"name": d.name, "dir": str(d), "mode": info.get("mode"), "practice": bool(info.get("practice")),
                "rehearsal": reh, "profile": info.get("profile"), "odom": info.get("odom"),
                "operator": info.get("operator"), "running": bool(info.get("running")), "pid": info.get("pid"),
                "panel_port": info.get("panel_port"), "panel_token_file": str(d / "panel_url.txt"),
                "panel_ready": info.get("panel_ready"), "started": info.get("started"), "ended": info.get("ended"),
                "console_t_mono_age_s": round(time.monotonic() - seen, 3) if info.get("running") and seen else None,
                "port_base": info.get("port_base") or PORT_BASES[slot],
                "control_port": (info.get("port_base") or PORT_BASES[slot]) + 6,
                "view": v[1].status() if v and v[0] == str(d) else None,
                "panel": info.get("panel") if info.get("running") else None}

    def compute_state(self):
        live = self.running()
        self.ensure_views(live)
        pc = self.pc()
        try:
            du = shutil.disk_usage(self.day)
            disk = round(du.free / 1e9, 1)
        except OSError:
            disk = None
        all_s = self.all_sessions()
        sets = self.rehearsal_sets(all_s)
        robot_live = live.get("robot") and live["robot"].get("running")
        phone = f"http://{TAILNET_HOST}:{live['robot']['panel_port']}/" if robot_live else None
        job = dict(self.scan_job)
        job.pop("then", None)
        return {"api": 1, "capabilities": CAPABILITIES + (["confirm"] if console_supports("--confirm") else []),
                "version": self.version(), "day": str(self.day), "t": time.time(), "t_mono": time.monotonic(),
                "pc": {"load": pc.get("load"), "cpus": pc.get("cpus"), "gpu": pc.get("gpu"), "busy": pc.get("busy"),
                       "chains": pc.get("chains"), "disk_free_gb": disk},
                "robot_link": self.robot_link_v1(), "profiles": self.profiles_v1(),
                "default_profile": self.default_profile(), "default_odom": self.robot_profile().get("odom", "leg"),
                "live": {slot: self.live_v1(slot, live.get(slot)) for slot in ("robot", "sim")},
                "scans": self.scans_v1(sets), "scan_job": job,
                "sessions": self.sessions_v1(all_s), "rehearsals": sets,
                "rehearsal_job": {k: v for k, v in (self.set_job or {}).items() if k != "cancel"} or None,
                "gpu": self.gpu_v1(),
                "doctor": {"t": self.doctor.get("t"), "exit": self.doctor.get("exit"), "text": self.doctor.get("output"),
                           "running": self.doctor.get("running")},
                "phone_panel": phone, "studio_link": self.studio_link}

    def state_v1(self):
        """The cached state (computed now when there is none yet), its live ages brought up to date."""
        if self.cache is None:
            self.refresh()
        st = dict(self.cache)
        st["state_age_s"] = round(time.monotonic() - self.cache_t, 3)
        live = {}
        for slot, x in (st.get("live") or {}).items():
            if x and x.get("running") and self.console_seen.get(slot):
                x = dict(x, console_t_mono_age_s=round(time.monotonic() - self.console_seen[slot], 3))
            live[slot] = x
        st["live"] = live
        job = dict(self.scan_job)  # the jobs as they are now (a POST may have started one since the last refresh)
        job.pop("then", None)
        st["scan_job"] = job
        st["rehearsal_job"] = {k: v for k, v in (self.set_job or {}).items() if k != "cancel"} or None
        return st

    def refresh(self):
        st = self.compute_state()
        self.cache, self.cache_t = st, time.monotonic()
        return st

    def background(self, period=1.0):
        """The 1 Hz state loop (and the live views it keeps)."""
        def loop():
            while not self.bg_stop.is_set():
                t0 = time.monotonic()
                try:
                    self.refresh()
                except Exception as exc:  # a state that fails once must not stop the service
                    print(f"g1 lab: state failed: {type(exc).__name__}: {exc}", flush=True)
                self.bg_wake.wait(max(0.05, period - (time.monotonic() - t0)))
                self.bg_wake.clear()
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    # -- live views ----------------------------------------------------------------------------------
    def ensure_views(self, live):
        """One view poller per live slot (its node control port); stopped when the session ends."""
        for slot in ("robot", "sim"):
            info = live.get(slot)
            want = str(info["dir"]) if info and info.get("running") else None
            have = self.views.get(slot)
            if have and have[0] != want:
                have[1].stop()
                self.views.pop(slot, None)
                have = None
            if want and not have:
                port = (info.get("port_base") or PORT_BASES[slot]) + 6
                self.views[slot] = (want, ViewPoller(port))

    def align_of(self, slot):
        """How to draw a slot's odometry in the scene: the last start check of the session, or, for a rehearsal run,
        its course (exact: identity; offset: the repose; drift: None, its estimate is never drawn in the scene)."""
        info = self.live(slot)
        if not info:
            return None
        d = Path(info["dir"])
        try:
            r = json.loads((d / "rehearsal.json").read_text())
        except (OSError, ValueError):
            r = None
        if r is not None:
            if r.get("drift") or r.get("kind") == "drift" or r.get("from_robot"):
                return None
            s_from_c = r.get("scene_from_course") or [0.0, 0.0, 0.0]
            rfs = frames.inverse(s_from_c)
            return {"scene_from_odom": s_from_c, "P_c": None, "theta": rfs[2], "t": rfs[:2], "at": info.get("started"),
                    "overlap": None, "ok": True, "moved_m": round(math.hypot(*s_from_c[:2]), 3),
                    "turned_deg": round(math.degrees(s_from_c[2]), 1), "source": "course"}
        c = self.checks.get(slot)
        if not c or c.get("session") != d.name or not c.get("ok") or not c.get("scene_from_odom"):
            return None
        return {k: c.get(k) for k in ("scene_from_odom", "P_c", "theta", "t", "at", "overlap", "ok", "moved_m",
                                      "turned_deg")} | {"source": "start_check", "check_id": c.get("check_id")}

    def live_view(self, slot, cloud_after=-1, mem_after=-1):
        if slot not in self.slots:
            return 404, {"ok": False, "error": "slot must be robot or sim"}
        info = self.live(slot)
        if not info:
            return 404, {"ok": False, "error": f"no {slot} session is live"}
        v = self.views.get(slot)
        if not v or v[0] != str(info["dir"]):
            self.ensure_views({slot: dict(info, running=True)})
            v = self.views.get(slot)
        out = v[1].view(cloud_after, mem_after)
        if out is None:
            st = v[1].status()
            return 200, {"ok": False, "error": st["error"] or "no view yet", "supported": st["supported"],
                         "session": Path(info["dir"]).name}
        out["ok"] = bool(out.get("ok", True))
        out["session"] = Path(info["dir"]).name
        out["slot"] = slot
        out["align"] = self.align_of(slot)
        return 200, out

    # -- scans ---------------------------------------------------------------------------------------------
    def scan_v1(self, req):
        try:
            seconds = min(30.0, max(3.0, float(req.get("seconds") or 8.0)))
        except (TypeError, ValueError):
            return {"ok": False, "error": "seconds must be a number"}
        if self.live("sim") and (self.live("sim") or {}).get("practice"):
            return {"ok": False, "error": "a practice session is live: end it before a robot scan"}
        return self.scan(goal=[2.0, 0.0], seconds=seconds)

    def scan_live(self):
        job = self.scan_job
        name = job.get("name")
        out = {"running": bool(job.get("running")), "name": name, "practice": bool(job.get("practice")),
               "phase": job.get("phase"), "error": job.get("error"), "t": None, "n_frames": 0, "qpos": None,
               "joint_names": list(contract.MJ_JOINTS), "cloud": None, "memory": None, "start_S": None}
        f = self.scans_dir / str(name) / "live.json" if name else None
        if f is not None and f.is_file():
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                data = {}
            out.update({k: data.get(k) for k in ("t", "n_frames", "qpos", "joint_names", "cloud", "memory", "start_S",
                                                 "elapsed_s", "verdict", "frame", "memory_frame") if k in data})
            out["file_age_s"] = round(time.time() - f.stat().st_mtime, 2)
        return out

    def scan_scene(self, name):
        """A scan for the scene view: its memory (memory_S), start_S and scene_from_memory, the twin boxes (memory_S)
        and the course boxes (scene)."""
        if name in self.scene_cache:
            return self.scene_cache[name]
        import numpy as np

        f = self.scan_file(name, "memory.npz")
        info = self.scan_file(name, "scan.json")
        if f is None or info is None:
            return None
        x = json.loads(info.read_text())
        with np.load(f, allow_pickle=False) as z:
            start = [float(v) for v in z["start_xy"]] if "start_xy" in z.files else [0.0, 0.0]
            yaw = float(z["start_yaw"]) if "start_yaw" in z.files else 0.0
            near = z["near"] if "near" in z.files else None
            mem = wire.encode_memory(z["support"], z["ceiling"], z["lo"], float(z["res"]), near=near, seq=0, epoch=0)
            goal_xy = [float(v) for v in z["goal_xy"]] if "goal_xy" in z.files else None
        start_s = frames.pose([*start, yaw])
        twin = []
        tf = self.scan_file(name, "twin.json")
        if tf is not None:
            for b in json.loads(tf.read_text()).get("boxes") or []:
                twin.append({k: b.get(k) for k in ("center_xy", "half_xy", "yaw", "z_bottom", "z_top")})
        course = []
        cf = self.scan_file(name, "course/task.json")
        goal_course = None
        if cf is not None:
            task = json.loads(cf.read_text())
            course = [{k: b.get(k) for k in ("center_xy", "half_xy", "yaw", "z_bottom", "z_top")}
                      for b in task.get("scene_boxes_world") or []]
            goal_course = task.get("goal_world_xy")
        s_from_m = frames.scene_from_memory(start_s)
        out = {"name": name, "time": x.get("time"), "verdict": x.get("verdict"),
               "reasons": self.goal_free_reasons(x.get("reasons")), "site_reasons": x.get("reasons"),
               "frames": x.get("frames"), "start_S": start_s, "scene_from_memory": s_from_m, "memory": mem,
               "memory_frame": "memory_S", "twin": twin, "twin_frame": "memory_S", "course_boxes": course,
               "course_frame": "scene", "memory_file": str(f), "goal_ahead_left": x.get("goal_ahead_left"),
               "goal_scene_at_scan": frames.apply(s_from_m, goal_xy).tolist() if goal_xy else goal_course,
               "practice": x.get("practice"), "fresh": self.scan_fresh(name)[0], "age_s": self.scan_age(name)}
        self.scene_cache[name] = out
        return out

    # -- practice ------------------------------------------------------------------------------------------
    def practice_scan(self, req):
        """A practice scan: robot_sim alone (unarmed: the stand pose held), scanned like the robot (`g1 site --iface lo
        --odom dds` with the robot profile's mount and rear blind), then the sim stops. The result is a normal scan
        named practice-HHMMSS."""
        if self.scan_job["running"]:
            return {"ok": False, "error": "a scan is running"}
        if self.live("robot") or self.live("sim"):
            return {"ok": False, "error": "a session is live: a practice scan runs its simulator alone"}
        if self.set_job and self.set_job.get("running"):
            return {"ok": False, "error": "a rehearsal set is running"}
        try:
            seconds = min(30.0, max(3.0, float(req.get("seconds") or 8.0)))
        except (TypeError, ValueError):
            return {"ok": False, "error": "seconds must be a number"}
        if req.get("scan"):
            task = self.scan_file(req["scan"], "course/task.json")
            source = f"scan:{req['scan']}"
        else:
            course = str(req.get("course") or "lab-door-real")
            task = (ROOT / "sim/courses" / course / "task.json").resolve()
            if (ROOT / "sim/courses").resolve() not in task.parents or not task.is_file():
                task = None
            source = course
        if task is None:
            return {"ok": False, "error": "no such course or scan"}
        t = json.loads(Path(task).read_text())
        g = t.get("goal_world_xy") or [2.0, 0.0]
        s0 = t.get("initial_root_xy") or [0.0, 0.0]
        goal = [float(g[0]) - float(s0[0]), float(g[1]) - float(s0[1])]
        name = f"practice-{time.strftime('%H%M%S')}"
        out = self.scans_dir / name
        out.mkdir(parents=True, exist_ok=True)
        self.scan_job = {"running": True, "name": name, "error": None, "t": time.time(), "then": None,
                         "practice": True, "phase": "sim starting", "source": source}
        threading.Thread(target=self._practice_scan_job, args=(name, out, Path(task), goal, seconds, source),
                         daemon=True).start()
        return {"ok": True, "name": name}

    def _practice_scan_job(self, name, out, task, goal, seconds, source):
        py = str(ROOT / ".venv/bin/python")
        err = None
        sim = None
        try:
            from g1deploy.console import _env

            env = _env()
            sim_cmd = [py, "-u", "sim/realtime/robot_sim.py", "--task", str(task), "--log", str(out / "practice_sim.npz"),
                       "--iface", "lo", "--lidar-io", "dds", "--cloud-bind", f"tcp://127.0.0.1:{REHEARSAL_PORT_BASE + 4}"]
            log = open(out / "sim.log", "w")
            sim = subprocess.Popen(sim_cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
            self.practice_proc = sim
            t0 = time.monotonic()
            while time.monotonic() - t0 < 90 and sim.poll() is None:
                if "[sim] task" in (out / "sim.log").read_text(errors="replace"):
                    break
                time.sleep(0.5)
            if sim.poll() is not None:
                raise RuntimeError("the simulator stopped: " + (out / "sim.log").read_text(errors="replace")[-400:])
            time.sleep(3.0)  # DDS discovery and the first lidar frames
            self.scan_job["phase"] = "scanning"
            cmd = [py, str(ROOT / "tools/scan_scene.py"), "--robot", self.args.robot, "--iface", "lo", "--odom", "dds",
                   "--goal", *(f"{v:.3f}" for v in goal), "--seconds", str(seconds), "--out", str(out), "--no-studio",
                   "--live-out", str(out / "live.json"), "--practice", source]
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=360,
                               stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONPATH=str(ROOT)))
            if r.returncode:
                err = (r.stderr or r.stdout)[-500:]
        except Exception as exc:
            err = f"practice scan failed: {type(exc).__name__}: {exc}"
        finally:
            if sim is not None and sim.poll() is None:
                try:
                    os.killpg(sim.pid, signal.SIGINT)
                    sim.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(sim.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            self.practice_proc = None
            self.scan_job = {"running": False, "name": name, "error": err, "t": time.time(), "then": None,
                             "practice": True, "phase": "done", "source": source}

    def reposed_course(self, scan, offset, tag):
        """`scan`'s course moved so the robot starts at `offset` (its pose in the scene): the course origin is the
        robot's start, so scene_from_course = offset. Returns (task file, robot_from_scene)."""
        src = self.scan_file(scan, "course/task.json")
        if src is None:
            raise ValueError(f"no scanned twin {scan}")
        rfs = frames.inverse(offset)
        task = registration.repose_task(json.loads(src.read_text()), rfs[2], rfs[:2])
        task["start_offset"] = {"scene_from_course": frames.pose(offset), "scan": scan}
        f = self.day / "reposed" / f"{scan}-{tag}-{time.strftime('%H%M%S')}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(task, indent=1) + "\n")
        return f, rfs

    @staticmethod
    def parse_offset(v):
        if v is None:
            return frames.offset_pose()
        o = [float(x) for x in v]
        if len(o) != 3 or not all(math.isfinite(x) for x in o):
            raise ValueError("start_offset is [dx, dy, dyaw]")
        return frames.pose(o)

    def start_v1(self, req):
        mode = req.get("mode")
        operator = str(req.get("operator") or "")[:60] or None
        if mode == "robot":
            scan = req.get("scan")
            if not scan or self.scan_file(scan, "scan.json") is None:
                return {"ok": False, "error": "pick the scan the goal was placed on (scan)"}
            fresh, why = self.scan_fresh(scan)
            if not fresh:
                return {"ok": False, "error": why, "scan_stale": True}
            conf = {k: bool((req.get("confirm") or {}).get(k)) for k in CONFIRM_ROBOT}
            r = {"mode": "robot", "profile": req.get("profile") or self.default_profile(), "odom": req.get("odom"),
                 "scene_scan": scan, "scan": False, "operator": operator, "confirm": conf, "note": req.get("note"),
                 "confirm_json": dict(conf, operator=operator, t=time.time())}
            return self.start(r, v1=True)
        if mode != "sim":
            return {"ok": False, "error": "mode must be sim or robot"}
        if not req.get("practice"):
            return {"ok": False, "error": "a v1 sim start is a practice session (practice: true); rehearse runs sets"}
        scan = req.get("scan")
        if not scan or self.scan_file(scan, "course/task.json") is None:
            return {"ok": False, "error": "pick a scan with a course (a practice scan)"}
        if self.live("robot"):
            return {"ok": False, "error": "a robot session is live: practice and robot sessions never run together"}
        try:
            offset = self.parse_offset(req.get("start_offset"))
            task, rfs = self.reposed_course(scan, offset, "practice")
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        odom = req.get("odom") or self.robot_profile().get("odom", "leg")
        sim_odom = {"leg": "leg", "zmq": "lio", "fused": "fused", "lio": "lio", "dds": "dds"}.get(odom)
        if sim_odom is None:
            return {"ok": False, "error": "odometry must be leg, zmq or fused"}
        goal = json.loads(task.read_text()).get("goal_world_xy")
        r = {"mode": "sim", "practice": True, "profile": req.get("profile") or self.default_profile(), "label": "practice",
             "course": "scan:" + scan, "task_file": str(task), "sim_odom": sim_odom, "operator": operator,
             "note": req.get("note"), "goal": goal[:2] if goal else None, "start_offset": offset,
             "confirm": req.get("confirm"), "scene_scan": scan}
        if req.get("confirm"):
            r["confirm_json"] = dict(req["confirm"], operator=operator, practice=True, t=time.time())
        res = self.start(r)
        if res.get("ok"):
            meta = {"scan": scan, "start_offset": offset, "scene_from_course": offset, "odom": odom, "task": str(task)}
            (Path(res["dir"]) / "practice.json").write_text(wire.dumps(meta) + "\n")
        return res

    def stop_v1(self, slot):
        slot = slot or ("robot" if self.live("robot") else "sim")
        if slot not in self.slots:
            return {"ok": False, "error": "slot must be robot or sim"}
        cur = self.live(slot)
        if not cur:
            return {"ok": False, "error": "no session running"}
        reh = bool(cur.get("rehearsal"))
        if reh:  # a rehearsal run: stopping it ends its set too
            if self.set_job and self.set_job.get("running"):
                self.set_job["cancel"] = True
        else:
            s = self._console_full_state(cur, slot=slot)
            if s is None:
                return {"ok": False, "error": "the console does not answer: E-STOP from Studio or the panel, or the "
                                              "physical E-stop"}
            phase, runner = s.get("phase"), s.get("runner")
            if phase in ("PRESCAN", "RUN"):
                return {"ok": False, "error": "the robot is walking: Stand or E-STOP first, end the session when it stands"}
            atts = s.get("attempts") or []
            after_attempt = bool(atts) and atts[-1].get("result") is not None
            if runner not in ("not started", "E-STOP") and not (runner == "CONTROL" and after_attempt):
                return {"ok": False, "error": "a session with the runner up ends before Arm, after an E-STOP, or standing "
                                              "after an attempt (robot in the gantry)"}
        try:
            os.killpg(cur["pid"], signal.SIGINT)  # the console shuts down its processes and writes the report
        except ProcessLookupError:
            pass
        return {"ok": True}

    # -- rehearsal sets ------------------------------------------------------------------------------------
    def rehearse_set(self, req):
        """Three rehearsals of one goal, in sequence in the sim slot: exact (the scan's pose, true odometry), drift (the
        measured leg drift, DRIFT) and start offset (the course reposed so the robot starts 20 deg right / 0.2 m off,
        as on 2026-10-27). Today's guards for each run (GPU beside a robot session, never while the robot walks)."""
        scan = req.get("scan")
        try:
            goal = [float(v) for v in req.get("goal_scene")]
            if len(goal) != 2 or not all(math.isfinite(v) for v in goal):
                raise ValueError
        except (TypeError, ValueError):
            return {"ok": False, "error": "goal_scene must be two numbers"}
        if not scan or self.scan_file(scan, "course/task.json") is None:
            return {"ok": False, "error": "pick a scan with a course"}
        prof = req.get("profile") or self.default_profile()
        try:
            profiles.find(prof)
        except profiles.ProfileError as exc:
            return {"ok": False, "error": str(exc)}
        if self.set_job and self.set_job.get("running"):
            return {"ok": False, "error": "a rehearsal set is running"}
        if self.live("sim"):
            return {"ok": False, "error": "a sim session is running (practice or rehearsal): end it first"}
        if self.walking():
            return {"ok": False, "error": "the robot is walking: rehearse when it stands"}
        try:
            offset = self.parse_offset(req.get("start_offset"))
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        name = f"set-{time.strftime('%H%M%S')}"
        job = {"name": name, "scan": scan, "goal_scene": goal, "profile": prof, "odom": req.get("odom"),
               "offset": {"pose": offset}, "runs": {k: None for k in RUN_KINDS}, "current": None, "running": True,
               "error": None, "t": time.time(), "operator": req.get("operator")}
        self.set_job = job
        threading.Thread(target=self._run_set, args=(job,), daemon=True).start()
        return {"ok": True, "name": name}

    def _run_set(self, job):
        try:
            for kind in RUN_KINDS:
                if job.get("cancel"):
                    job["error"] = job.get("error") or "stopped"
                    break
                job["current"] = kind
                r = {"mode": "sim", "profile": job["profile"], "course": "scan:" + job["scan"], "script": REHEARSE_SCRIPT,
                     "label": "rehearse" if kind == "exact" else f"rehearse-{kind}", "goal": job["goal_scene"],
                     "_set": job["name"], "operator": job.get("operator")}
                meta = {"scan": job["scan"], "goal": job["goal_scene"], "goal_scene": job["goal_scene"],
                        "profile": job["profile"], "odom": job.get("odom"), "set": job["name"], "kind": kind,
                        "drift": None, "scene_from_course": [0.0, 0.0, 0.0]}
                if kind == "drift":
                    r["drift"] = True
                    meta["drift"] = DRIFT
                elif kind == "offset":
                    offset = job["offset"]["pose"]
                    task, rfs = self.reposed_course(job["scan"], offset, "offset")
                    r.update(task_file=str(task), goal=frames.apply(rfs, job["goal_scene"]).tolist())
                    meta.update(goal=r["goal"], scene_from_course=offset, offset={"pose": offset, "task": str(task)})
                res = self.start(r)
                if not res.get("ok"):
                    job["error"] = f"{kind} run not started: {res.get('error')}"
                    break
                job["runs"][kind] = res["name"]
                (Path(res["dir"]) / "rehearsal.json").write_text(wire.dumps(meta) + "\n")
                while True:
                    cur = self.live("sim")
                    if not cur or cur.get("dir") != res["dir"]:
                        break
                    time.sleep(1.0)
                self.running()  # the run's end: its manifest
        except Exception as exc:
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["running"], job["current"] = False, None
            try:
                with open(self.day / "rehearsal_sets.jsonl", "a") as f:
                    f.write(wire.dumps({k: v for k, v in job.items() if k != "cancel"}) + "\n")
            except OSError:
                pass

    def track(self, run):
        """A rehearsal run's sim truth in the scene (after the run): 25 Hz qpos (36-D, mapped by joint name from the
        sim model's), contacts over CONTACT_N at the robot, the result."""
        import numpy as np

        d = self.session_dir(run)
        if d is None:
            return 404, {"ok": False, "error": "no such run"}
        try:
            reh = json.loads((d / "rehearsal.json").read_text())
        except (OSError, ValueError):
            return 404, {"ok": False, "error": "not a rehearsal run"}
        cur = self.live("sim")
        live = bool(cur and cur.get("dir") == str(d))
        rows = A.load(d / "attempts.jsonl")
        if live or not (d / "sim.npz").exists():
            ended = (d / "console_summary.json").exists()
            prog = {"kind": reh.get("kind"), "set": reh.get("set"),
                    "elapsed_s": round(time.time() - cur["started"], 1) if live else None,
                    "phase": ((self.console_state.get("sim") or (0, {}))[1] or {}).get("phase") if live else None,
                    "attempt": rows[-1].get("result") if rows else None}
            if live or not ended:
                return 200, {"name": run, "running": True, "progress": prog}
            return 200, {"name": run, "running": False, "error": "the run ended without sim truth (sim.npz)",
                         "progress": prog}
        with np.load(d / "sim.npz", allow_pickle=False) as z:
            t, q, cn, rp = z["t"], z["qpos"], z["contact_n"], z["root_pos"]
        s_from_c = reh.get("scene_from_course") or [0.0, 0.0, 0.0]
        tq = t[::2][:len(q)]  # qpos is logged every other log row (50 Hz of the 100 Hz log)
        step = max(1, int(round((1.0 / TRACK_HZ) / max(float(np.median(np.diff(tq))) if len(tq) > 1 else 0.02, 1e-3))))
        q36 = frames.qpos_to(s_from_c, qpos36(q[::step]))
        t25 = tq[::step]
        # contacts: the largest force in each 1/25 s window over the threshold, drawn at the robot
        contacts = []
        win = max(1, int(round((1.0 / TRACK_HZ) / max(float(np.median(np.diff(t))) if len(t) > 1 else 0.01, 1e-3))))
        for i in range(0, len(cn), win):
            j = i + int(np.argmax(cn[i:i + win]))
            if cn[j] > CONTACT_N:
                contacts.append({"t": round(float(t[j]), 3), "root_xy": np.round(frames.apply(s_from_c, rp[j, :2]), 3).tolist(),
                                 "n": round(float(cn[j]), 1)})
        goal = reh.get("goal_scene") or reh.get("goal")
        final = frames.apply(s_from_c, rp[-1, :2]) if len(rp) else None
        result = rows[-1].get("result") if rows else None
        return 200, {"name": run, "running": False, "arrived": result == "arrived", "result": result,
                     "kind": reh.get("kind") or ("drift" if reh.get("drift") else "exact"), "set": reh.get("set"),
                     "frame": "scene", "scene_from_course": s_from_c, "goal_scene": goal,
                     "joint_names": list(contract.MJ_JOINTS), "hz": round(1.0 / max(float(np.median(np.diff(t25))), 1e-3), 1)
                     if len(t25) > 1 else None,
                     "t": np.round(t25, 3).tolist(), "qpos": np.round(q36, 4).tolist(), "contacts": contacts,
                     "contact_max_n": round(float(cn.max()), 1) if len(cn) else 0.0,
                     "final_dist_m": round(float(np.hypot(*(final - goal))), 3) if final is not None and goal else None,
                     "duration_s": (rows[-1].get("duration_s") if rows and rows[-1].get("duration_s") is not None
                                    else round(float(t[-1] - t[0]), 2) if len(t) else None)}

    # -- start check and goals -------------------------------------------------------------------------
    def _live_map(self, slot, half_m=None):
        """The node's map (StudentNode.map_snapshot, odometry frame): the console's /map, or `half_m` straight from
        the node's control socket."""
        if half_m is None:
            code, data, _ = self.panel_proxy("GET", "/map", slot=slot)
            try:
                return json.loads(data) if code == 200 else None
            except ValueError:
                return None
        info = self.live(slot)
        if not info:
            return None
        return self.node_request((info.get("port_base") or PORT_BASES[slot]) + 6,
                                 {"cmd": "map", "half_m": half_m, "res_m": 0.1}, timeout=0.5)

    @staticmethod
    def node_request(port, msg, timeout=1.0):
        try:
            import zmq

            s = zmq.Context.instance().socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(f"tcp://127.0.0.1:{port}")
            try:
                s.send_string(json.dumps(msg))
                if not s.poll(int(timeout * 1000)):
                    return None
                return json.loads(s.recv())
            finally:
                s.close(0)
        except Exception:
            return None

    def start_check_v1(self, scan, slot="robot"):
        out, _ = self._start_check(scan, slot)
        return out

    def _start_check(self, scan, slot):
        slot = slot or "robot"
        if slot not in self.slots:
            return {"ok": False, "error": "slot must be robot or sim"}, None
        cur = self.live(slot)
        if not cur:
            return {"ok": False, "error": f"no {slot} session is live"}, None
        grid = self.scan_grid(scan) if scan else None
        if grid is None:
            return {"ok": False, "error": "pick the scan the goal was placed on"}, None
        live = self._live_map(slot)
        out = registration.start_check(grid, live)
        out.update(scan=scan, slot=slot, session=Path(cur["dir"]).name, at=time.time())
        env = self.offset_envelope(scan)
        out["offset_envelope"] = env
        if out.get("ok") and env:
            out["within_offset_envelope"] = bool(out["moved_m"] <= env["moved_m"] + ENVELOPE_SLACK_M
                                                 and abs(out["turned_deg"]) <= env["turned_deg"] + ENVELOPE_SLACK_DEG)
        else:
            out["within_offset_envelope"] = None
        cs = (self.console_state.get(slot) or (None, None))[1] or {}
        out["closed_attempts"] = sum(1 for a in cs.get("attempts") or [] if a.get("result") is not None)
        self.checks[slot] = out
        self.record_check(dict(out, action="start_check"), slot=slot)
        return out, live

    def coarse_goal_check(self, slot, goal_odom, live_map=None):
        """The goal on the live node memory, in odometry: seen, not an obstacle (none within GOAL_CLEAR_M), inside the
        memory square. The view's full memory when it is fresh, else the node's map out to 6 m."""
        import numpy as np

        gx, gy = (float(v) for v in goal_odom)
        v = self.views.get(slot)
        mem = None
        if v is not None:
            with v[1].lock:
                fresh = v[1].t_recv is not None and time.monotonic() - v[1].t_recv < 2.0
                mem = v[1].memory if fresh else None
        if mem is not None:
            sup, _, _ = wire.decode_memory(mem)
            lo, res = np.asarray(mem["lo"], float), float(mem["res"])
            n = sup.shape
            i, j = int(round((gx - lo[0]) / res)), int(round((gy - lo[1]) / res))
            src = "view memory"
            if not (0 <= i < n[0] and 0 <= j < n[1]):
                return {"ok": False, "reason": "the goal is outside the live memory square", "source": src}
            if not np.isfinite(sup[i, j]):
                return {"ok": False, "reason": "the goal cell was never seen on the live map", "source": src}
            r = int(math.ceil(GOAL_CLEAR_M / res))
            ii, jj = np.mgrid[max(0, i - r):min(n[0], i + r + 1), max(0, j - r):min(n[1], j + r + 1)]
            near = np.hypot((ii - i) * res, (jj - j) * res) <= GOAL_CLEAR_M
            h = sup[ii[near], jj[near]]
            if np.any(np.isfinite(h) & (h >= OBSTACLE_M)):
                return {"ok": False, "reason": f"an obstacle within {GOAL_CLEAR_M} m of the goal on the live map",
                        "source": src}
            return {"ok": True, "reason": None, "source": src}
        m = self._live_map(slot, half_m=6.0) or live_map
        if not m or not m.get("ok"):
            return {"ok": False, "reason": "no live map to check the goal on", "source": None}
        src = "node map" if m is not live_map else "start check map"
        res, x0, y0, rows = float(m["res"]), float(m["x0"]), float(m["y0"]), m["rows"]
        i, j = int(math.floor((gx - x0) / res)), int(math.floor((gy - y0) / res))
        if not (0 <= i < len(rows) and 0 <= j < len(rows[0])):
            return {"ok": False, "reason": "the goal is outside the live memory square", "source": src}
        if rows[i][j] == " ":
            return {"ok": False, "reason": "the goal cell was never seen on the live map", "source": src}
        r = int(math.ceil(GOAL_CLEAR_M / res))
        for a in range(max(0, i - r), min(len(rows), i + r + 1)):
            for b in range(max(0, j - r), min(len(rows[0]), j + r + 1)):
                if math.hypot((a - i) * res, (b - j) * res) <= GOAL_CLEAR_M and rows[a][b] in "23s":
                    return {"ok": False, "reason": f"an obstacle within {GOAL_CLEAR_M} m of the goal on the live map",
                            "source": src}
        return {"ok": True, "reason": None, "source": src}

    def next_seq(self):
        with self.lock:
            self.goal_seq += 1
            return self.goal_seq

    def send_goal_v1(self, req):
        slot = req.get("slot") or "robot"
        check_id = req.get("check_id")
        refuse = lambda why, **kw: dict({"ok": False, "reason": why, "error": why, "check": None, "goal_odom": None,  # noqa: E731
                                         "act": None}, **kw)
        if not isinstance(check_id, str) or not check_id.strip():
            return refuse("no goal check: send_goal needs the check_id of a passed goal check")
        if slot not in self.slots or not self.live(slot):
            return refuse(f"no {slot} session is live")
        try:
            goal = [float(v) for v in req.get("goal_scene")]
            if len(goal) != 2 or not all(math.isfinite(v) for v in goal):
                raise ValueError
        except (TypeError, ValueError):
            return refuse("goal_scene must be two numbers")
        if self.walking(slot):
            return refuse("the robot is walking: send the goal when it stands")
        check, live_map = self._start_check(req.get("scan"), slot)
        check["check_id"] = check_id
        if not check.get("ok"):
            return refuse((check.get("error") or "the start check failed") + ": no goal sent", check=check)
        s_from_o = check["scene_from_odom"]
        g_odom = frames.apply(frames.inverse(s_from_o), goal).tolist()
        coarse = self.coarse_goal_check(slot, g_odom, live_map)
        check["coarse"] = coarse
        if not coarse["ok"]:
            return refuse(coarse["reason"] + ": no goal sent", check=check, goal_odom={"x": g_odom[0], "y": g_odom[1]})
        seq = self.next_seq()
        act = {"act": "goal_odom", "x": round(g_odom[0], 4), "y": round(g_odom[1], 4), "seq": seq, "check_id": check_id,
               "id": req.get("id") or str(uuid.uuid4()), "source": req.get("source") or "studio",
               "operator": req.get("operator")}
        if req.get("expect") is not None:
            act["expect"] = req["expect"]
        code, reply = self.act(slot, act)
        ok = code == 200 and bool(reply.get("ok", True)) and reply.get("result") != "refused"
        self.record_check({"action": "send_goal", "check_id": check_id, "goal_scene": goal, "goal_odom": g_odom,
                           "seq": seq, "ok": ok, "reply": reply, "at": time.time()}, slot=slot)
        return {"ok": ok, "reason": None if ok else (reply.get("reason") or reply.get("error") or f"HTTP {code}"),
                "check": check, "goal_odom": {"x": act["x"], "y": act["y"], "seq": seq, "check_id": check_id},
                "coarse": coarse, "act": reply}

    # -- acts ----------------------------------------------------------------------------------------------
    def act(self, slot, msg):
        """An operator act relayed to the slot's console (POST /act), audited to DAY/hub_actions.jsonl."""
        msg = {k: v for k, v in dict(msg).items() if k != "slot"}
        msg.setdefault("source", "studio")
        cur = self.live(slot) if slot in self.slots else None
        if cur is None:
            code, reply = 404, {"ok": False, "error": f"no {slot} session is live"}
        elif slot == "robot" and msg.get("act") == "g" and self.live("sim") and (self.live("sim") or {}).get("rehearsal"):
            code, reply = 409, {"ok": False, "result": "refused", "error": "a rehearsal is running: GO when it has ended"}
        else:
            code, data, _ = self.panel_proxy("POST", "/act", body=json.dumps(msg).encode(), slot=slot,
                                             timeout=2.5 if msg.get("act") != "estop" else 1.5)
            try:
                reply = json.loads(data)
                if not isinstance(reply, dict):
                    reply = {"reply": reply}
            except ValueError:
                reply = {"ok": False, "error": data.decode(errors="replace")[:300]}
        if code != 200:
            reply.setdefault("ok", False)
        rec = {"wall": time.time(), "t_mono": time.monotonic(), "slot": slot,
               "session": Path(cur["dir"]).name if cur else None, "act": msg.get("act"), "id": msg.get("id"),
               "source": msg.get("source"), "operator": msg.get("operator"), "expect": msg.get("expect"),
               "args": {k: v for k, v in msg.items() if k not in ("act", "id", "source", "operator", "expect")},
               "code": code, "reply": reply}
        try:
            with open(self.day / "hub_actions.jsonl", "a") as f:
                f.write(wire.dumps(rec) + "\n")
        except OSError:
            pass
        return code, reply

    def panel_state_v1(self, slot):
        if slot not in self.slots:
            return 404, {"ok": False, "error": "slot must be robot or sim"}
        cur = self.live(slot)
        if not cur:
            return 404, {"ok": False, "error": f"no {slot} session is live"}
        s = self._console_full_state(cur, slot=slot, timeout=1.5)
        if s is None:
            return 502, {"ok": False, "error": "the console does not answer"}
        s["hub_received_t_mono"] = time.monotonic()
        return 200, s

    def manifest_v1(self, name):
        d = self.session_dir(name)
        if d is None:
            return 404, {"ok": False, "error": "no such session"}
        return 200, evidence.write(d)

    # -- readiness -----------------------------------------------------------------------------------------
    def readiness(self, scan=None, slot=None):
        st = self.state_v1()
        live = st.get("live") or {}
        if slot not in ("robot", "sim"):
            slot = "robot" if (live.get("robot") or {}).get("running") else (
                "sim" if (live.get("sim") or {}).get("practice") and (live.get("sim") or {}).get("running") else "robot")
        info = live.get(slot) if (live.get(slot) or {}).get("running") else None
        practice = bool(info and info.get("practice")) or (slot == "sim")
        cs_t, cs = self.console_state.get(slot) or (None, None)
        if info is None:
            cs = None
        cs_at = (time.time() - (time.monotonic() - cs_t)) if cs is not None and cs_t else None
        now = time.time()
        rp = self.robot_profile()
        odom = (info or {}).get("odom") or rp.get("odom", "leg")
        out = []

        def add(id_, group, label, status, evidence_, detail, at, max_age, after, anchor=None, fix=None):
            if status != "unknown" and at is not None and max_age is not None and now - at > max_age:
                status, detail = "unknown", f"{detail} (measured {now - at:.0f} s ago)"
            out.append({"id": id_, "group": group, "label": label, "status": status, "evidence": evidence_,
                        "detail": detail, "at": at, "max_age_s": max_age, "measurable_after": after, "anchor": anchor,
                        "fix": {"hint": fix} if fix else None})

        link = st.get("robot_link") or {}
        if practice:
            add("link.carrier", "link", "Robot link", "pass", "CONFIGURED", "practice: the simulator on lo", now, None,
                "connect")
            add("link.lidar", "link", "Lidar rate", "pass", "CONFIGURED", "practice: the simulated MID-360 at 10 Hz",
                now, None, "connect")
        else:
            add("link.carrier", "link", "Robot link", "pass" if link.get("carrier") else "block", "MEASURED",
                f"{link.get('iface')}: " + ("carrier" if link.get("carrier") else "no carrier (robot off or cable out)"),
                now, 5, "connect", fix=None if link.get("carrier") else "plug the robot's Ethernet in, power it on")
            hz = link.get("lidar_hz")
            if hz is None:
                add("link.lidar", "link", "Lidar rate", "unknown", "MEASURED",
                    "no lidar adapter stats (`g1 lidar start` writes them every 10 s)", None, 30, "connect",
                    fix="g1 lidar start --robot " + str(self.args.robot))
            else:
                add("link.lidar", "link", "Lidar rate", "pass" if hz >= 8.0 else "warn" if hz >= 1.0 else "block",
                    "MEASURED", f"{hz:.1f} frames/s from the adapter", link.get("lidar_at"), 30, "connect",
                    fix=None if hz >= 8.0 else "the driver is alive but silent: check eno1 and the lidar's power")
        if odom in ("zmq", "fused", "lio"):
            age = link.get("lio_pose_age_s")
            if practice:
                add("link.lio", "link", "LIO odometry", "pass", "CONFIGURED", "practice: the sim LIO stand-in", now,
                    None, "connect")
            elif not link.get("lio_bridge"):
                add("link.lio", "link", "LIO odometry", "block", "MEASURED", f"odometry {odom} needs the LIO bridge",
                    now, 5, "connect", fix="g1 lio start, then g1 lio check with the robot standing")
            else:
                add("link.lio", "link", "LIO odometry", "unknown" if age is None else "pass" if age < 0.2 else "warn",
                    "MEASURED", "LIO pose age " + ("unknown" if age is None else f"{age * 1000:.0f} ms"),
                    link.get("lio_at"), 15, "connect")
        else:
            add("link.lio", "link", "LIO odometry", "pass", "CONFIGURED", f"odometry {odom}: LIO not used", now, None,
                "connect")
        pc = st.get("pc") or {}
        add("robot.pc", "robot", "Deploy PC load", "warn" if pc.get("busy") else "pass", "MEASURED",
            "; ".join(pc.get("busy") or []) or "quiet", st.get("t"), 15, "connect")
        doc = st.get("doctor") or {}
        if doc.get("t") is None or doc.get("running"):
            add("robot.doctor", "robot", "g1 doctor", "unknown", "MEASURED",
                "running" if doc.get("running") else "not run today", doc.get("t"), 1800, "connect", fix="run doctor")
        else:
            add("robot.doctor", "robot", "g1 doctor", "pass" if doc.get("exit") == 0 else "warn", "MEASURED",
                f"exit {doc.get('exit')}", doc.get("t"), 1800, "connect", fix=None if doc.get("exit") == 0 else
                "read its text")
        checks = {c.get("key"): c for c in (cs or {}).get("checks") or []}
        for key, label, after in (("lowstate", "Lowstate rate", "session"), ("temps", "Motor temperatures", "session"),
                                  ("remote", "Remote", "session"), ("lag", "Node loop lag", "node"),
                                  ("latency", "Plan latency", "node")):
            c = checks.get(key)
            if key in ("remote", "temps") and practice:
                continue
            group = "perception" if key in ("lag", "latency") else "robot"
            if c is None:
                add(f"{group}.{key}", group, label, "unknown", "MEASURED", f"after {after}", None, 5, after)
            else:
                add(f"{group}.{key}", group, label, STATUS_OF_LEVEL.get(c.get("level"), "unknown"), "MEASURED",
                    c.get("detail") or "", cs_at, 5, after)
        runner = (cs or {}).get("runner")
        add("robot.runner", "robot", "Runner", "unknown" if runner in (None, "not started", "booting") else
            "block" if runner == "E-STOP" else "pass" if runner in ("CONTROL", "INIT done") else "unknown", "MEASURED",
            runner or "after arm", cs_at, 5, "arm")
        blind, brange = rp.get("rear_blind_deg"), rp.get("rear_blind_range_m")
        add("perception.rear_blind", "perception", "Rear blind sector", "pass", "CONFIGURED",
            f"returns behind the torso closer than {brange} m are dropped ({blind} deg)" if blind else "off",
            now, None, "connect", anchor={"kind": "wedge", "frame": "robot", "deg": blind, "range_m": brange}
            if blind else None)
        nb = (cs or {}).get("near_body")
        for key, label in (("near", "Near the body"), ("behind", "Tall behind")):
            c = checks.get(key)
            cells = (nb or {}).get("close_cells") if key == "near" else None
            add(f"perception.{key}", "perception", label, "unknown" if c is None else
                STATUS_OF_LEVEL.get(c.get("level"), "unknown"), "MEASURED", (c or {}).get("detail") or "after arm",
                cs_at if c else None, 5, "arm", anchor={"kind": "cells", "frame": "robot", "cells": cells} if cells else None)
        scans = {x["name"]: x for x in st.get("scans") or []}
        sx = scans.get(scan) if scan else None
        if sx is None:
            add("scene.scan", "scene", "Scan", "unknown", "DERIVED", "pick or take a scan", None, SCAN_FRESH_S, "scan")
        else:
            fresh, why = self.scan_fresh(scan)
            add("scene.scan", "scene", "Scan", "pass" if fresh else "block", "DERIVED",
                f"{sx['age_s'] / 60:.0f} min old" if fresh else why, now - (sx.get("age_s") or 0), SCAN_FRESH_S if fresh
                else None, "scan", anchor={"kind": "pose", "frame": "scene", "pose": [0.0, 0.0, 0.0]},
                fix=None if fresh else "scan again where the robot stands")
            rs = sx.get("reasons") or []
            add("scene.site", "scene", "Site check (start)", "block" if any(r.startswith("start inside") for r in rs)
                else "warn" if rs else "pass", "DERIVED", "; ".join(rs) or "nothing near the start",
                now - (sx.get("age_s") or 0), None, "scan")
        c = self.checks.get(slot)
        sess = (info or {}).get("name")
        if c and sess and c.get("session") == sess and (not scan or c.get("scan") == scan):
            closed = sum(1 for a in (cs or {}).get("attempts") or [] if a.get("result") is not None)
            stale = closed > (c.get("closed_attempts") or 0)  # an attempt ended since: the robot walked
            robot_scene = frames.compose(c["scene_from_odom"], c["P_c"]) if c.get("P_c") else None
            within = c.get("within_offset_envelope")
            add("scene.start_check", "scene", "Start check", "unknown" if stale else ("pass" if within else "warn")
                if c.get("ok") else "block", "MEASURED",
                ("an attempt ended since: check again; " if stale else "") + (
                    f"moved {c.get('moved_m')} m, turned {c.get('turned_deg')} deg, overlap {c.get('overlap')}"
                    + ("" if within else " (outside the rehearsed start offsets)" if within is False else
                       " (no start-offset rehearsal of this scan)") if c.get("ok") else c.get("error") or "no match"),
                c.get("at"), 300, "node",
                anchor={"kind": "pose", "frame": "scene", "pose": robot_scene} if robot_scene else None)
        else:
            add("scene.start_check", "scene", "Start check", "unknown", "MEASURED", "after the node runs (Arm)", None,
                300, "node")
        locks = (cs or {}).get("go_locks")
        if cs is None:
            add("scene.goal", "scene", "Goal delivered", "unknown", "DERIVED", "after goal", None, None, "goal")
            add("operator.go", "operator", "GO locks", "unknown", "DERIVED", "after arm", None, 5, "arm")
        elif locks is None:
            add("scene.goal", "scene", "Goal delivered", "unknown", "DERIVED", "this console reports no GO locks", cs_at,
                5, "goal")
            add("operator.go", "operator", "GO locks", "unknown", "DERIVED", "this console reports no GO locks", cs_at,
                5, "arm")
        else:
            gl = [x for x in locks if "goal" in str(x).lower()]
            add("scene.goal", "scene", "Goal delivered", "block" if gl else "pass", "DERIVED", "; ".join(gl) or
                "a goal is confirmed by the node", cs_at, 5, "goal")
            add("operator.go", "operator", "GO locks", "block" if locks else "pass", "DERIVED",
                "; ".join(map(str, locks)) or "GO would be accepted now", cs_at, 5, "arm")
        sets = [s for s in st.get("rehearsals") or [] if scan and s.get("scan") == scan and not s.get("legacy_pair")]
        if not sets:
            add("scene.rehearsal", "scene", "Rehearsal set", "unknown", "DERIVED", "after rehearsal", None, None,
                "rehearsal")
        else:
            s = sets[0]
            v, ok = s.get("verdict"), s.get("offset_ok")
            status = ("unknown" if s.get("running") or v is None else "block" if v in ("fails", "fails with drift")
                      else "pass" if v == "robust" and ok else "warn")
            add("scene.rehearsal", "scene", "Rehearsal set", status, "DERIVED",
                "running" if s.get("running") else f"{v or '-'}; start offset {'ok' if ok else 'failed' if ok is False else '-'}",
                None, None, "rehearsal")
        conf = None
        if info:
            try:
                conf = (json.loads((Path(info["dir"]) / "hub_start.json").read_text()) or {}).get("confirm")
            except (OSError, ValueError):
                conf = None
        if practice:
            add("operator.confirmations", "operator", "Start checklist", "pass", "OPERATOR", "practice", now, None,
                "session")
        elif info is None:
            add("operator.confirmations", "operator", "Start checklist", "unknown", "OPERATOR", "at the session start",
                None, None, "session")
        else:
            miss = [k for k in CONFIRM_ROBOT if not (conf or {}).get(k)]
            add("operator.confirmations", "operator", "Start checklist", "block" if miss else "pass", "OPERATOR",
                ("missing: " + ", ".join(miss)) if miss else "gantry, E-stop, remote, rope confirmed", info.get("started"),
                None, "session")
        return {"slot": slot, "scan": scan, "session": sess, "t": now, "checks": out}


def serve(hub: Hub, port: int, host="127.0.0.1"):
    page = (Path(__file__).with_name("hub.html")).read_text()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _auth(self):
            if parse_qs(urlparse(self.path).query).get("t", [""])[0] != hub.token:
                self.send_response(403)
                self.end_headers()
                return False
            return True

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code, obj):
            self._send(code, wire.dumps(obj))

        def do_GET(self):
            if not self._auth():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            arg = lambda k, d=None: q.get(k, [d])[0]  # noqa: E731
            if u.path.startswith("/api/v1/"):
                try:
                    self.get_v1(u.path[len("/api/v1/"):].strip("/").split("/"), arg)
                except Exception as exc:  # strict JSON even when something breaks
                    self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                return
            if u.path == "/":
                self._send(200, NOTICE.replace("__TOKEN__", hub.token), "text/html; charset=utf-8")
            elif u.path in ("/legacy", "/legacy/"):
                self._send(200, page.replace("__TOKEN__", hub.token), "text/html; charset=utf-8")
            elif u.path == "/api/state":
                self._send(200, json.dumps(hub.state(), default=str))
            elif u.path == "/api/command":
                try:
                    cmd, out = hub.command(json.loads(q.get("req", ["{}"])[0]))
                    self._send(200, json.dumps({"ok": True, "cmd": " ".join(cmd), "dir": str(out)}))
                except Exception as exc:
                    self._send(200, json.dumps({"ok": False, "error": str(exc)}))
            elif u.path in ("/panel", "/panel/", "/panel/state", "/panel/map"):
                code, data, ctype = hub.panel_proxy("GET", u.path[len("/panel"):] or "/")
                self._send(code, data, ctype)
            elif u.path in ("/sim-panel", "/sim-panel/", "/sim-panel/state", "/sim-panel/map"):
                code, data, ctype = hub.panel_proxy("GET", u.path[len("/sim-panel"):] or "/", slot="sim")
                self._send(code, data, ctype)
            elif u.path == "/api/grid":
                g = hub.scan_grid(q.get("s", [""])[0])
                self._send(200 if g else 404, json.dumps(g or {}, default=float))
            elif u.path == "/api/trajectory":
                tr = hub.trajectory(q.get("s", [""])[0])
                self._send(200 if tr else 404, json.dumps(tr or {}, default=float))
            elif u.path.startswith("/scan/"):
                parts = u.path.split("/", 3)
                f = hub.scan_file(parts[2], parts[3]) if len(parts) == 4 else None
                if f is None:
                    self._send(404, "{}")
                else:
                    ctype = "image/png" if f.suffix == ".png" else "application/json"
                    self._send(200, f.read_bytes(), ctype)
            elif u.path == "/report":
                text = hub.report(q.get("s", [""])[0])
                self._send(200 if text else 404, text or "no report (yet)", "text/plain; charset=utf-8")
            else:
                self._send(404, "{}")

        def get_v1(self, parts, arg):
            def num(k, d=-1):
                try:
                    return int(arg(k, d))
                except (TypeError, ValueError):
                    return d
            if parts == ["state"]:
                self._json(200, hub.state_v1())
            elif parts == ["readiness"]:
                self._json(200, hub.readiness(arg("scan"), arg("slot")))
            elif parts == ["sessions"]:
                self._json(200, hub.sessions_v1(offset=max(0, num("offset", 0)), limit=min(200, max(1, num("limit", SESSIONS_PAGE)))))
            elif parts == ["scan", "live"]:
                self._json(200, hub.scan_live())
            elif len(parts) == 3 and parts[0] == "scan" and parts[2] == "scene":
                sc = hub.scan_scene(parts[1])
                self._json(200 if sc else 404, sc or {"ok": False, "error": "no such scan"})
            elif len(parts) == 3 and parts[0] == "rehearsal" and parts[2] == "track":
                self._json(*hub.track(parts[1]))
            elif len(parts) == 3 and parts[0] == "live" and parts[2] == "view":
                self._json(*hub.live_view(parts[1], num("cloud_after"), num("mem_after")))
            elif len(parts) == 3 and parts[0] == "panel" and parts[2] == "state":
                self._json(*hub.panel_state_v1(parts[1]))
            elif len(parts) == 3 and parts[0] == "session" and parts[2] == "manifest":
                self._json(*hub.manifest_v1(parts[1]))
            else:
                self._json(404, {"ok": False, "error": "unknown endpoint"})

        def do_POST(self):
            if not self._auth():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                msg = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(msg, dict):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                self._send(400, '{"ok": false, "error": "a JSON object body"}')
                return
            path = urlparse(self.path).path
            if path.startswith("/api/v1/"):
                try:
                    self.post_v1(path[len("/api/v1/"):].strip("/"), msg)
                except Exception as exc:
                    self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                return
            if path in ("/panel/act", "/sim-panel/act"):  # E-STOP and the ladder's buttons, straight on to the panel
                slot = "robot" if path == "/panel/act" else "sim"
                if slot == "robot" and msg.get("act") == "g" and hub.live("sim"):
                    self._send(409, '{"ok": false, "error": "a rehearsal is running: GO when it has ended"}')
                    return
                code, data, ctype = hub.panel_proxy("POST", "/act", body=json.dumps(msg).encode(), slot=slot)
                self._send(code, data, ctype)
                return
            if path == "/api/start":
                out = hub.start(msg)
            elif path == "/api/stop":
                out = hub.stop(msg.get("slot"))
            elif path == "/api/send_goal":
                try:
                    out = hub.send_goal(msg.get("goal") or [], scan=msg.get("scan"))
                except (TypeError, ValueError):
                    out = {"ok": False, "error": "goal must be two numbers"}
            elif path == "/api/start_check":
                try:
                    out = hub.start_check(msg.get("scan"), [float(v) for v in msg.get("goal") or [2.0, 0.0]])
                except (TypeError, ValueError):
                    out = {"ok": False, "error": "goal must be two numbers"}
            elif path == "/api/doctor":
                out = hub.run_doctor()
            elif path == "/api/rehearse":
                out = hub.rehearse(msg)
            elif path == "/api/scan":
                try:
                    goal = [float(v) for v in msg.get("goal") or [2.0, 0.0]]
                    out = hub.scan(goal=goal, seconds=min(30.0, max(3.0, float(msg.get("seconds") or 8.0))))
                except (TypeError, ValueError):
                    out = {"ok": False, "error": "goal must be two numbers"}
            else:
                out = {"ok": False, "error": "unknown action"}
            self._send(200, json.dumps(out, default=str))

        def post_v1(self, path, msg):
            if path == "act":
                slot = msg.get("slot") or "robot"
                if not msg.get("act"):
                    self._json(400, {"ok": False, "error": "act missing"})
                    return
                code, reply = hub.act(slot, msg)
                hub.bg_wake.set()
                self._json(code, reply)
                return
            handlers = {"scan": hub.scan_v1, "practice/scan": hub.practice_scan, "start": hub.start_v1,
                        "stop": lambda m: hub.stop_v1(m.get("slot")), "rehearse": hub.rehearse_set,
                        "start_check": lambda m: hub.start_check_v1(m.get("scan"), m.get("slot") or "robot"),
                        "send_goal": hub.send_goal_v1, "doctor": lambda m: hub.run_doctor()}
            if path not in handlers:
                self._json(404, {"ok": False, "error": "unknown endpoint"})
                return
            out = handlers[path](msg)
            if not out.get("ok") and out.get("error") and "reason" not in out:
                out["reason"] = out["error"]
            hub.bg_wake.set()
            self._json(200, out)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    ap = argparse.ArgumentParser(prog="g1 lab", description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8769)
    ap.add_argument("--robot", default="lab-g1")
    ap.add_argument("--iface", default=None, help="the robot link (default: the robot profile's)")
    ap.add_argument("--day", default=str(ROOT / "runs" / f"lab-day-{time.strftime('%Y%m%d')}"))
    ap.add_argument("--headless", action="store_true", help="the deploy service for Deploy Studio: no page to open "
                                                            "(/ says so; the old page stays at /legacy)")
    ap.add_argument("--no-studio-link", action="store_true", help="do not link the day directory into Studio's "
                                                                  "robot-sessions (a development hub)")
    args = ap.parse_args(argv)
    if args.iface is None:
        try:
            args.iface = json.loads((ROOT / "robots" / f"{args.robot}.json").read_text()).get("iface", "eno1")
        except OSError:
            args.iface = "eno1"
    hub = Hub(args)
    server = serve(hub, args.port)
    hub.background()
    base = f"http://127.0.0.1:{args.port}"
    print(f"g1 lab: {base}/api/v1/state?t=... (token: {hub.token[:3]}..., {'G1_PANEL_TOKEN' if os.environ.get('G1_PANEL_TOKEN') else 'the panel token file'})"
          + ("" if args.headless else f"\n  legacy page {base}/legacy?t={hub.token}")
          + f"\n  day {hub.day}\n  Studio link {hub.studio_link or '(none)'}", flush=True)
    (hub.day / "hub_url.txt").write_text(f"{base}/{'api/v1/state' if args.headless else 'legacy'}?t={hub.token}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
