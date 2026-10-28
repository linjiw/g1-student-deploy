#!/usr/bin/env python3
"""g1 lab — the robot day's home page: pick a deploy profile, rehearse it in simulation, start the robot session, and
find every session of the day with its attempts, report and Studio replay.

`./g1 lab [--port 8769] [--robot lab-g1] [--day runs/lab-day-YYYYMMDD]` serves one page on 127.0.0.1:PORT (reached
from the operator's tailnet devices like the console's panel; every request needs the token: G1_PANEL_TOKEN, or a
random one printed at start). The hub never drives the robot itself. It starts the console (`g1 sim|robot ...
--profile P --arm --panel PORT`) as its own process group, so the console keeps running if the hub stops; on the
robot the runner still waits for the panel's Arm, and every motion command stays on the panel (E-STOP, Start SONIC,
GO). A robot session is ended from its panel; the hub stops a robot session only before Arm or after an E-STOP.
The page embeds the running session's panel (served through the hub at /panel/, same origin, so its confirmations
work) with its E-STOP bar: one page from the scan to GO. A restarted hub re-attaches to a session still running.

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
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import attempts as A, machine, profiles, registration, workspace  # noqa: E402

STUDIO_DATA = Path(os.environ.get("G1_STUDIO_DATA", Path.home() / "research-data/g1-deploy-studio"))
STUDIO_PORT = int(os.environ.get("G1_STUDIO_PORT", "8790"))
PANEL_PORTS = {"robot": 8770, "sim": 8771}
REHEARSAL_PORT_BASE = 5656  # hub sims run isolated (g1 sim --port-base): they may run beside a live robot session
# GPU memory a rehearsal chain needs beside a live robot session: its planner (a larger bundle grows to ~7-8 GB on
# no-safe resampling, tools/replay_plans.py measurements) and a second SONIC runner (~1 GB)
REHEARSAL_GPU_MIB = int(os.environ.get("G1_REHEARSAL_GPU_MIB", "8500"))


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


class Adopted:
    """A console the hub did not start in this run (it restarted while the session went on): poll() by pid."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        if self.returncode is None and not Path(f"/proc/{self.pid}").exists():
            self.returncode = 0
        return self.returncode


class Hub:
    def __init__(self, args):
        self.args = args
        self.day = Path(args.day).resolve()
        self.day.mkdir(parents=True, exist_ok=True)
        self.token = os.environ.get("G1_PANEL_TOKEN") or secrets.token_urlsafe(9)
        # one robot session and one rehearsal at a time, side by side (the rehearsal on its own ports)
        self.slots = {"robot": {"proc": None, "info": None}, "sim": {"proc": None, "info": None}}
        self.lock = threading.Lock()
        self.doctor = {"running": False, "output": None, "t": None}
        self.scan_job = {"running": False, "name": None, "error": None, "t": None, "then": None}
        self.scans_dir = self.day / "scans"
        self.twinned = set()  # robot sessions whose final map already has a twin
        self.pair_cache = {}  # (exact, drifted) rehearsal names -> rehearsal_pairs row
        self.adopt()
        self._pc = (0.0, None)
        self.studio_link = self.link_studio()

    def adopt(self):
        """Re-attach to a console session of this day that is still running (the hub restarted under it)."""
        for pid, cmd in machine._procs({os.getpid()}):
            m = re.search(r"g1deploy/console\.py (sim|robot)\b.*--record (\S+)", cmd)
            if not m or Path(m.group(2)).resolve().parent != self.day:
                continue
            port = re.search(r"--panel (\d+)", cmd)
            prof = re.search(r"--profile (\S+)", cmd)
            task = re.search(r"--task (\S+)", cmd)
            d = Path(m.group(2)).resolve()
            course = ("scan:" + Path(task.group(1)).parents[1].name) if task and "/scans/" in task.group(1) else None
            self.slots[m.group(1)] = {"proc": Adopted(pid), "info": {
                "pid": pid, "mode": m.group(1), "profile": prof.group(1) if prof else None, "dir": str(d), "cmd": cmd,
                "started": d.stat().st_mtime, "panel_port": int(port.group(1)) if port else PANEL_PORTS[m.group(1)],
                "course": course, "adopted": True}}
            print(f"g1 lab: re-attached to the running {m.group(1)} session {d.name} (pid {pid})", flush=True)

    def live(self, slot):
        """The running session of a slot (robot | sim), or None."""
        s = self.slots[slot]
        return s["info"] if s["proc"] is not None and s["proc"].poll() is None else None

    def walking(self):
        """The live robot is walking (or about to): no rehearsal or scan beside it then."""
        cur = self.live("robot")
        ps = self.panel_state(cur) if cur else None
        return bool(ps and ps.get("phase") in ("PRESCAN", "RUN"))

    def panel_proxy(self, method, path, query="", body=None, slot="robot"):
        """A running session's panel, through the hub (same origin as the page that embeds it): /panel/ is the robot
        session's, /sim-panel/ the rehearsal's."""
        cur = self.live(slot)
        if cur is None:
            return 404, b'{"error": "no session running"}', "application/json"
        prefix = "/panel" if slot == "robot" else "/sim-panel"
        if path == "/":  # this kit's current page (a console started earlier serves the page of its own start)
            page = (Path(__file__).with_name("panel.html")).read_text()
            page = page.replace("__TOKEN__", self.token).replace("__HUB_PORT__", "")
            for p_ in ("/act?t=", "/state?t=", "/map?t="):  # its requests go back through the hub
                page = page.replace(f"'{p_}", f"'{prefix}{p_}")
            return 200, page.encode(), "text/html; charset=utf-8"
        url = f"http://127.0.0.1:{cur['panel_port']}{path}?t={self.token}"
        req = urllib.request.Request(url, data=body, method=method)
        try:
            with urllib.request.urlopen(req, timeout=3.0) as r:
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

    def running(self):
        """Both slots: the session's info, running or the last one ended, with its panel's state."""
        out = {}
        for slot in ("robot", "sim"):
            twin = None
            with self.lock:
                sl = self.slots[slot]
                if sl["proc"] is not None and sl["proc"].poll() is not None:
                    sl["info"].update(exit=sl["proc"].returncode, ended=time.time())
                    sl["proc"] = None
                    d = sl["info"]["dir"]
                    if slot == "robot" and d not in self.twinned and (Path(d) / "node_memory.npz").exists():
                        self.twinned.add(d)  # what the robot saw on its walks, as a twin too (no map before Arm)
                        twin = d
                info = dict(sl["info"]) if sl["info"] else None
                live = sl["proc"] is not None
            if twin:
                self.scan(source_session=twin)
            if info:
                info["running"] = live
                if live:
                    info["panel_ready"] = (Path(info["dir"]) / "panel_url.txt").exists()
                    info["panel"] = self.panel_state(info)
            out[slot] = info
        return out

    def panel_state(self, cur):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{cur['panel_port']}/state?t={self.token}", timeout=1.0) as r:
                s = json.loads(r.read())
            now = next((x for x in s.get("ladder", []) if x["state"] == "now"), None)
            return {"runner": s.get("runner"), "phase": s.get("phase"), "next": now and now["label"],
                    "attempts": len(s.get("attempts", [])), "interlock": "rehearsal_running" in s,
                    "rehearsal_running": s.get("rehearsal_running"), "goal_rel": s.get("goal_rel"),
                    "xy": (s.get("node") or {}).get("xy") or s.get("xy")}
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
                "live": (live := self.running()), "current": self.current_view(live),
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

    # -- the real scene ----------------------------------------------------------------------------------
    def list_scans(self):
        out = []
        for d in sorted(self.scans_dir.glob("*/scan.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
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
        return dict(out, scan=scan, goal_scan=goal, t=time.time())

    def record_check(self, check):
        cur = self.live("robot")
        if cur is not None:
            with open(Path(cur["dir"]) / "start_check.jsonl", "a") as f:
                f.write(json.dumps(check, default=float) + "\n")

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
            fr = json.loads((d / "rehearsal.json").read_text()).get("from_robot")
        except (OSError, ValueError):
            fr = None
        if fr:
            back = lambda xy: np.round(registration.rotate(np.asarray(xy, float).reshape(-1, 2) - fr["t"], -fr["theta_rad"]), 3).tolist()  # noqa: E731
            out["walked"] = back(out["walked"]) if out["walked"] else out["walked"]
            out["plans"] = [back(pl) for pl in out["plans"]]
        if (d / "rehearsal.json").exists() and json.loads((d / "rehearsal.json").read_text()).get("drift"):
            out["plans"] = []  # planned in the drifted odometry's frame, not the scene's
        return out

    def rehearse(self, req):
        """The automatic rehearsal of a goal in a scanned twin: `g1 sim --task TWIN --goal A L --script ...`, then the
        same with odometry drift (DRIFT) when it ends (`pair`, default on).
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
        import numpy as np

        def key(x):
            r = x["rehearsal"]
            return (r["scan"], tuple(r["goal"]), r.get("profile"), json.dumps(r.get("from_robot"), sort_keys=True))
        exact = {key(x): x for x in sessions if x.get("rehearsal") and not x["rehearsal"].get("drift") and x["ended"]}
        out = {}
        for x in sessions:
            r = x.get("rehearsal")
            if not r or not r.get("drift") or not x["ended"] or key(x) not in exact:
                continue
            e = exact[key(x)]
            if (e["name"], x["name"]) in self.pair_cache:  # ended sessions do not change
                out[e["name"]] = self.pair_cache[(e["name"], x["name"])]
                continue
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
            out[e["name"]] = self.pair_cache[(e["name"], x["name"])] = {
                "drift": x["name"], "exact_result": res(e), "drift_result": res(x), "apart_m": way, "verdict": verdict}
        return out

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
                if req.get("task_file"):  # the twin moved to where the live robot stands (rehearse from_robot)
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
        return cmd, out

    def start(self, req):
        slot = req.get("mode")
        if slot not in self.slots:
            return {"ok": False, "error": "mode must be sim or robot"}
        if self.live(slot):
            return {"ok": False, "error": ("a robot session is live: end it from its panel first" if slot == "robot"
                                           else "a rehearsal is running: wait for it (about a minute)")}
        if slot == "sim" and self.live("robot"):
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
            return {"ok": False, "error": "a rehearsal is running: start the robot session when it ends (about a minute)"}
        if req.get("mode") == "robot":
            need = ("gantry", "estop", "remote", "clear")
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
            self.slots[slot] = {"proc": proc, "info": {
                "pid": proc.pid, "mode": slot, "profile": req["profile"], "dir": str(out), "cmd": " ".join(cmd),
                "started": time.time(), "panel_port": PANEL_PORTS[slot], "course": req.get("course"),
                "rehearsal": bool(req.get("script"))}}
        return {"ok": True, "dir": str(out), "cmd": " ".join(cmd)}

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
        it is relative to the robot now (the panel's Set goal)."""
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
