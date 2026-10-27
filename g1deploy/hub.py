#!/usr/bin/env python3
"""g1 lab — the robot day's home page: pick a deploy profile, rehearse it in simulation, start the robot session, and
find every session of the day with its attempts, report and Studio replay.

`./g1 lab [--port 8769] [--robot lab-g1] [--day runs/lab-day-YYYYMMDD]` serves one page on 127.0.0.1:PORT (reached
from the operator's tailnet devices like the console's panel; every request needs the token: G1_PANEL_TOKEN, or a
random one printed at start). The hub never drives the robot itself. It starts the console (`g1 sim|robot ...
--profile P --arm --panel PORT`) as its own process group, so the console keeps running if the hub stops; on the
robot the runner still waits for the panel's Arm, and every motion command stays on the panel (E-STOP, Start SONIC,
GO). A robot session is ended from its panel; the hub stops a robot session only before Arm or after an E-STOP.

Sessions land in the day directory, which is also linked into Deploy Studio's robot-sessions directory so its Replay
lists each one (Studio rescans every 10 s).

The real scene: `Scan now`, and every robot session start, runs tools/scan_scene.py first (a read-only lidar scan of the
standing robot's surroundings, Deploy Studio's twin of it saved as a Studio arena, and a sim course of that twin), and
every robot session that ends gets a twin of its final map too. The page shows the latest scan and its twin, and the
sim rehearsal offers the scanned twins first, so rehearsals run in the scene the robot is actually in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import attempts as A, machine, profiles, workspace  # noqa: E402

STUDIO_DATA = Path(os.environ.get("G1_STUDIO_DATA", Path.home() / "research-data/g1-deploy-studio"))
STUDIO_PORT = int(os.environ.get("G1_STUDIO_PORT", "8790"))
PANEL_PORTS = {"robot": 8770, "sim": 8771}
# the automatic rehearsal in a scanned twin: the robot-day ladder without a hand on it (the panel shows it live)
REHEARSE_SCRIPT = "wait:init,key:s,wait:control,wait:ready,sleep:3,key:g,wait:arrived,sleep:3"
SCAN_FRESH_S = 15 * 60  # a robot session reuses the scan its goal was picked on when it is younger than this
REHEARSAL_COURSES = ("lab-door-real", "rp-door-rope", "rp-rope-behind", "lab-site-2m", "open-2m", "open-3m", "slalom",
                     "pillar", "box-low")


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
                        "student": raw.get("student"),
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


class Hub:
    def __init__(self, args):
        self.args = args
        self.day = Path(args.day).resolve()
        self.day.mkdir(parents=True, exist_ok=True)
        self.token = os.environ.get("G1_PANEL_TOKEN") or secrets.token_urlsafe(9)
        self.proc, self.current = None, None
        self.lock = threading.Lock()
        self.doctor = {"running": False, "output": None, "t": None}
        self.scan_job = {"running": False, "name": None, "error": None, "t": None, "then": None}
        self.scans_dir = self.day / "scans"
        self.twinned = set()  # robot sessions whose final map already has a twin
        self._pc = (0.0, None)
        self.studio_link = self.link_studio()

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
        return {"iface": iface, "carrier": carrier, "livox_driver": livox, "robot": self.args.robot}

    def running(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is not None:
                self.current["exit"] = self.proc.returncode
                self.current["ended"] = time.time()
                self.proc = None
                if self.current["mode"] == "robot" and self.current["dir"] not in self.twinned:
                    self.twinned.add(self.current["dir"])  # what the robot saw on its walks, as a twin too
                    self.scan(source_session=self.current["dir"])
            cur = dict(self.current) if self.current else None
        if cur and self.proc is not None:
            url = Path(cur["dir"]) / "panel_url.txt"
            cur["panel_ready"] = url.exists()
            cur["running"] = True
            cur["panel"] = self.panel_state(cur)
        elif cur:
            cur["running"] = False
        return cur

    def panel_state(self, cur):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{cur['panel_port']}/state?t={self.token}", timeout=1.0) as r:
                s = json.loads(r.read())
            now = next((x for x in s.get("ladder", []) if x["state"] == "now"), None)
            return {"runner": s.get("runner"), "phase": s.get("phase"), "next": now and now["label"],
                    "attempts": len(s.get("attempts", []))}
        except Exception:
            return None

    def state(self):
        sessions = sorted((d for d in self.day.iterdir() if d.is_dir() and (d / "session.json").exists()),
                          key=lambda d: d.stat().st_mtime, reverse=True)
        pc = self.pc()
        du = shutil.disk_usage(self.day)
        return {"day": str(self.day), "token_hint": self.token[:3] + "...", "pc": {
                    "load": pc.get("load"), "cpus": pc.get("cpus"), "gpu": pc.get("gpu"), "busy": pc.get("busy"),
                    "chains": pc.get("chains"), "disk_free_gb": round(du.free / 1e9, 1)},
                "robot_link": self.robot_link(), "profiles": list_profiles(), "courses": [c for c in REHEARSAL_COURSES
                                                                                         if (ROOT / "sim/courses" / c).is_dir()],
                "current": self.running(), "sessions": [session_summary(d, self.day) for d in sessions[:40]],
                "studio": {"port": STUDIO_PORT, "linked": self.studio_link}, "doctor": self.doctor,
                "panel_ports": PANEL_PORTS, "default_robot": self.args.robot, "plan": self.plan(),
                "scans": self.list_scans(), "scan_job": dict(self.scan_job, then=bool(self.scan_job.get("then"))),
                "default_profile": self.default_profile()}

    def default_profile(self):
        """The profile the page selects first: private/lab_hub.json {"default_profile": NAME}, else `deploy`."""
        try:
            return str(json.loads((ROOT / "private/lab_hub.json").read_text())["default_profile"])
        except (OSError, ValueError, KeyError, TypeError):
            return "deploy"

    # -- the real scene ----------------------------------------------------------------------------------
    def list_scans(self):
        out = []
        for d in sorted(self.scans_dir.glob("*/scan.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
            try:
                out.append(json.loads(d.read_text()))
            except (OSError, ValueError):
                continue
        return out

    def scan(self, goal=None, seconds=8.0, then=None, source_session=None):
        """tools/scan_scene.py in the background; `then`: a robot start request to launch when the scan is done
        (whatever the scan found: the panel's own checks still gate GO)."""
        if self.scan_job["running"]:
            return {"ok": False, "error": "a scan is running"}
        if source_session is None:
            with self.lock:
                if self.proc is not None and self.proc.poll() is None:
                    return {"ok": False, "error": "a session is running: scan before or after it"}
            if not self.robot_link()["carrier"]:
                return {"ok": False, "error": f"no link on {self.args.iface}: robot off or cable out"}
        name = f"session-{Path(source_session).name}" if source_session else f"scan-{time.strftime('%H%M%S')}"
        cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/scan_scene.py"), "--out", str(self.scans_dir / name)]
        if source_session:
            cmd += ["--from-session", str(source_session)]
        else:
            cmd += ["--robot", self.args.robot, "--goal", *(str(float(v)) for v in (goal or [2.0, 0.0])),
                    "--seconds", str(float(seconds))]
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
        return out

    def rehearse(self, req):
        """The automatic rehearsal of a goal in a scanned twin: `g1 sim --task TWIN --goal A L --script ...`."""
        scan, goal = req.get("scan"), req.get("goal")
        if not scan or not goal:
            return {"ok": False, "error": "pick a scan and a goal first"}
        r = dict(req, mode="sim", course="scan:" + scan, script=REHEARSE_SCRIPT, label="rehearse")
        res = self.start(r)
        if res.get("ok"):
            (Path(res["dir"]) / "rehearsal.json").write_text(json.dumps(
                {"scan": scan, "goal": [float(goal[0]), float(goal[1])], "profile": req.get("profile")}) + "\n")
        return res

    def scan_file(self, name, rel):
        if rel not in ("site.png", "twin.json", "scan.json", "site.json", "course/task.json", "memory.npz"):
            return None
        p = (self.scans_dir / name / rel).resolve()
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
                if self.scans_dir.resolve() not in task.parents or not task.is_file():
                    raise ValueError(f"no scanned twin {course[5:]}")
                cmd += ["--task", str(task)]
            elif not (ROOT / "sim/courses" / course).is_dir():
                raise ValueError(f"no course {course}")
            else:
                cmd += ["--layout", course]
            if req.get("goal"):
                cmd += ["--goal", *(str(float(v)) for v in req["goal"])]
        else:
            cmd += ["--robot", req.get("robot") or self.args.robot]
            goal = req.get("goal") or [1.5, 0.0]
            cmd += ["--goal", *(str(float(v)) for v in goal)]
        if mode == "sim" and req.get("script"):
            cmd = [c for c in cmd if c != "--arm"] + ["--script", req["script"], "--step-timeout", "60"]
        if req.get("note"):
            cmd += ["--note", str(req["note"])[:300]]
        if req.get("operator"):
            cmd += ["--operator", str(req["operator"])[:60]]
        return cmd, out

    def start(self, req):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return {"ok": False, "error": "a session is running: end it from its panel first"}
        if req.get("mode") == "robot":
            need = ("gantry", "estop", "remote", "clear")
            missing = [k for k in need if not (req.get("confirm") or {}).get(k)]
            if missing:
                return {"ok": False, "error": "confirm the robot checklist first: " + ", ".join(missing)}
            if not self.robot_link()["carrier"]:
                return {"ok": False, "error": f"no link on {self.args.iface}: robot off or cable out"}
            fresh = None
            if req.get("scene_scan"):
                f = self.scans_dir / str(req["scene_scan"]) / "scan.json"
                if f.is_file() and time.time() - f.stat().st_mtime < SCAN_FRESH_S:
                    fresh = req["scene_scan"]
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
        env = dict(os.environ, G1_PANEL_TOKEN=self.token, G1_HUB_PORT=str(self.args.port))
        log = open(out / "console.out", "w")
        # its own process group and session: the console (and the runner it owns) outlive the hub
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        with self.lock:
            self.proc = proc
            self.current = {"pid": proc.pid, "mode": req["mode"], "profile": req["profile"], "dir": str(out),
                            "cmd": " ".join(cmd), "started": time.time(), "panel_port": PANEL_PORTS[req["mode"]],
                            "course": req.get("course")}
        return {"ok": True, "dir": str(out), "cmd": " ".join(cmd)}

    def stop(self):
        cur = self.running()
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

    def run_doctor(self):
        if self.doctor["running"]:
            return {"ok": False, "error": "doctor is running"}
        self.doctor = {"running": True, "output": None, "t": time.time()}

        def go():
            try:
                r = subprocess.run([str(ROOT / "g1"), "doctor", "--robot", self.args.robot], cwd=str(ROOT),
                                   capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)
                out = (r.stdout + r.stderr)[-12000:]
            except Exception as exc:
                out = f"doctor failed: {exc}"
            self.doctor = {"running": False, "output": out, "t": time.time()}
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

        def do_GET(self):
            if not self._auth():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                self._send(200, page.replace("__TOKEN__", hub.token), "text/html; charset=utf-8")
            elif u.path == "/api/state":
                self._send(200, json.dumps(hub.state(), default=str))
            elif u.path == "/api/command":
                try:
                    cmd, out = hub.command(json.loads(q.get("req", ["{}"])[0]))
                    self._send(200, json.dumps({"ok": True, "cmd": " ".join(cmd), "dir": str(out)}))
                except Exception as exc:
                    self._send(200, json.dumps({"ok": False, "error": str(exc)}))
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

        def do_POST(self):
            if not self._auth():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                msg = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(400, '{"ok": false}')
                return
            path = urlparse(self.path).path
            if path == "/api/start":
                out = hub.start(msg)
            elif path == "/api/stop":
                out = hub.stop()
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

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    ap = argparse.ArgumentParser(prog="g1 lab", description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8769)
    ap.add_argument("--robot", default="lab-g1")
    ap.add_argument("--iface", default=None, help="the robot link (default: the robot profile's)")
    ap.add_argument("--day", default=str(ROOT / "runs" / f"lab-day-{time.strftime('%Y%m%d')}"))
    args = ap.parse_args(argv)
    if args.iface is None:
        try:
            args.iface = json.loads((ROOT / "robots" / f"{args.robot}.json").read_text()).get("iface", "eno1")
        except OSError:
            args.iface = "eno1"
    hub = Hub(args)
    server = serve(hub, args.port)
    print(f"g1 lab: http://127.0.0.1:{args.port}/?t={hub.token}\n  day {hub.day}\n  Studio link "
          f"{hub.studio_link or '(no Studio data dir)'}", flush=True)
    (hub.day / "hub_url.txt").write_text(f"http://127.0.0.1:{args.port}/?t={hub.token}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
