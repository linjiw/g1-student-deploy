"""What a session directory holds as evidence, and how its recording is doing.

manifest(d) / write(d): manifest.json (v2), one row per kind of record: present, partial, missing or n/a, whether this
session requires it, its size, its time span and gaps (recorded streams), and the reason. The hub writes it when a
session ends and on request; the console and the recorder write the files it lists.

recording_health(d): the recorder seen live (from rec/stats.json, its heartbeat, and the chunk files): expected vs
active streams, rates, last write, drops, gaps, bytes, disk free, errors, finalized.

What a session requires (the operators' decision of 2026-10-28, encoded here):
  robot      lowstate, lowcmd, lidar_raw, node_status, node_events, plans, console_events, attempts
             (+ lio_odom, lio_cloud with odometry zmq | fused)
  practice   the same (+ the LIO streams with zmq | fused | lio); a plain `g1 sim` session: as practice
  rehearsal  the same + its sim truth (sim.npz; rec/odostate live)
  optional   video, runner_log, report (written at the end)
The other records (node_memory, start_checks, scan, identity, operator_confirmations; the rehearsal set in force with
a robot attempt) are required where they apply, as before. A record that does not apply to this session as it went
(no Arm: no node; no GO: no attempt) is n/a with its reason and not required. A session is complete only when every
required record is present (not partial), the recording was finalized by the recorder (rec/finalized.json) and the
console closed the session (console_summary.json); a session still running is "recording", never complete.

  python -m g1deploy.evidence SESSION_DIR [--write | --health]
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ITEMS = ("lowstate", "lowcmd", "lidar_raw", "lio_cloud", "lio_odom", "node_status", "node_events", "plans",
         "node_memory", "console_events", "attempts", "start_checks", "scan", "rehearsal", "identity",
         "operator_confirmations", "runner_log", "report", "video")
LIO_ODOM = ("zmq", "fused", "lio")  # odometry modes with a LIO pose stream (and its cloud)
ROBOT_LIO_ODOM = ("zmq", "fused")  # on the robot (lio is the simulator's stand-in)
REQUIRED_STREAMS = ("lowstate", "lowcmd", "lidar_raw", "node_status", "node_events", "plans", "console_events",
                    "attempts")
REQUIRED_WHERE_THEY_APPLY = ("node_memory", "start_checks", "scan", "identity", "operator_confirmations")
OPTIONAL = ("video", "runner_log", "report")
# manifest id -> recorder stream directory (rec/<dir>) for the recorded streams
REC_DIRS = {"lowstate": "lowstate", "lowcmd": "lowcmd", "lidar_raw": "lidar", "lio_cloud": "zmq_cloud",
            "lio_odom": "zmq_odom"}
# recording health: (id, recorder stream) in display order; the ids of the manifest where there is one
HEALTH_STREAMS = (("lowstate", "lowstate"), ("lowcmd", "lowcmd"), ("lidar_raw", "lidar"), ("lio_odom", "zmq_odom"),
                  ("lio_cloud", "zmq_cloud"), ("node_status", "node_status"), ("sim_truth", "odostate"),
                  ("secondary_imu", "secondary_imu"), ("lidar_imu", "lidar_imu"), ("student_stream", "zmq_pose"),
                  ("runner_debug", "zmq_debug"))
HEARTBEAT_STALE_S = 3.0  # stats.json older than this: the recorder is not running (or its disk is stuck)
EVENT_STREAM_STALE_S = 5.0  # an event-driven stream (no expected rate) counts as active this long after a sample
GAP_PARTIAL_S, GAP_PARTIAL_FRAC = 1.0, 0.05  # silent longer than max(1 s, 5 % of its span): partial
TAIL_PARTIAL_S, TAIL_PARTIAL_FRAC = 10.0, 0.05  # a stream that stopped this long before its reference: partial
TAIL_REFERENCE = {"lidar_raw": "lowstate", "lio_odom": "lowstate", "lio_cloud": "lowstate", "node_status": "lowcmd"}
DISK_WARN_GB = 5.0
EOCD = b"PK\x05\x06"  # a whole .npz (zip) ends with its end-of-central-directory record


def _size(p: Path):
    try:
        if p.is_file():
            return p.stat().st_size
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        pass
    return None


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _rows(p: Path):
    out = []
    try:
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def _pid_alive(pid, marker: bytes):
    """pid runs and its command line mentions `marker` (a reused pid is not the process we look for)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        return marker in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def _r(v, n=2):
    return None if v is None else round(float(v), n)


# ------------------------------------------------------------------------------------------------ the session
def session_info(d) -> dict:
    """Mode, kind (robot | practice | rehearsal | sim), odometry and how far the session went, from its files."""
    d = Path(d)
    info = _json(d / "session.json") or {}
    hub = _json(d / "hub_start.json") or {}
    reh = _json(d / "rehearsal.json")
    meta = _json(d / "rec" / "meta.json") or {}
    mode = info.get("mode") or hub.get("mode") or ("sim" if (d / "sim.npz").exists() or reh else None)
    rehearsal = reh is not None or d.name.startswith("rehearse")
    practice = not rehearsal and bool(info.get("practice") or hub.get("practice") or (info.get("hub") or {}).get("practice")
                                      or (d / "practice.json").exists() or d.name.startswith("practice-"))
    kind = "rehearsal" if rehearsal else "practice" if practice else "robot" if mode == "robot" else "sim"
    odom = info.get("odom") or hub.get("odom")
    lio = odom in (ROBOT_LIO_ODOM if kind == "robot" else LIO_ODOM)
    return {"dir": d, "info": info, "hub": hub, "rehearsal": reh, "meta": meta, "mode": mode, "kind": kind,
            "practice": practice, "robot": mode == "robot", "odom": odom, "lio": lio,
            "runner": (d / "runner.log").exists(), "node": (d / "node.log").exists(),
            "armed": (d / "runner.log").exists() or (d / "node.log").exists(),
            "console_pid": info.get("console_pid"), "closed": (d / "console_summary.json").exists()}


def policy(kind: str, lio: bool) -> dict:
    """id -> "required" | "optional" for a session of this kind (before what applies to it as it went)."""
    req = set(REQUIRED_STREAMS) | set(REQUIRED_WHERE_THEY_APPLY)
    if lio:
        req |= {"lio_odom", "lio_cloud"}
    if kind in ("robot", "rehearsal"):  # robot: the rehearsal set in force (or the override) with each attempt
        req.add("rehearsal")             # rehearsal: its sim truth
    return {i: "required" if i in req else "optional" for i in ITEMS}


def recorder_state(d, s: dict | None = None) -> dict:
    """Is the recorder of session `d` running, finalized, or gone without finalizing?"""
    d = Path(d)
    rec = d / "rec"
    stats = _json(rec / "stats.json") or {}
    meta = (s["meta"] if s is not None else _json(rec / "meta.json")) or {}
    fin = _json(rec / "finalized.json")
    legacy = bool(meta) and "pid" not in meta  # recorded before finalized.json: a clean stop wrote stopped_unix
    finalized = fin is not None or (legacy and meta.get("stopped_unix") is not None)
    now = time.time()
    try:
        hb = stats.get("t_unix") or (rec / "stats.json").stat().st_mtime
    except OSError:
        hb = None
    hb_age = None if hb is None else max(0.0, now - hb)
    pid = stats.get("pid") or meta.get("pid")
    pid_alive = _pid_alive(pid, b"recorder")
    fresh = hb_age is not None and hb_age < HEARTBEAT_STALE_S
    running = not finalized and (pid_alive if pid else fresh)
    if not rec.is_dir() or not (meta or stats):
        state = "absent"
    elif finalized:
        state = "finalized"
    elif running and not fresh:
        state = "stalled"  # the process runs but its heartbeat stopped (stuck disk?)
    elif running:
        state = "recording"
    else:
        state = "not finalized"
    return {"state": state, "finalized": finalized, "alive": state == "recording", "pid": pid, "pid_alive": pid_alive,
            "heartbeat_age_s": _r(hb_age), "legacy": legacy, "stats": stats, "meta": meta,
            "finalized_reason": (fin or {}).get("reason") or ("stopped (recorded before finalized.json)"
                                                             if finalized and fin is None else None),
            "clean": (fin or {}).get("clean")}


# ------------------------------------------------------------------------------------------------ streams on disk
def _chunk_whole(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            if f.tell() < 22:
                return False
            f.seek(-22, 2)
            return f.read(4) == EOCD
    except OSError:
        return False


def _chunk_t(p: Path):
    import numpy as np

    try:
        with np.load(p) as z:
            t = z["t"]
            return (float(t[0]), float(t[-1])) if len(t) else None
    except Exception:  # truncated, not a zip, no t: unusable, reported by the caller
        return None


def stream_on_disk(path: Path) -> dict:
    """A recorded stream as it is on disk: chunks (or JSON lines), the usable ones, its time span, what was skipped."""
    out = {"exists": path.exists(), "bytes": _size(path), "chunks": 0, "unreadable": [], "unfinished": 0,
           "first_t": None, "last_t": None, "span_s": None, "rows": None, "bad_lines": 0}
    if path.is_dir():
        files = sorted(p for p in path.glob("[0-9]*.npz") if ".tmp" not in p.name)
        out["unfinished"] = len(list(path.glob("*.tmp.npz")))
        good, bad = [], []
        for p in files:
            (good if _chunk_whole(p) else bad).append(p)
        first = next((ft for ft in (_chunk_t(p) for p in good[:3]) if ft), None)
        last = None
        for p in reversed(good):  # a cut chunk at the end is skipped, the one before it counts
            last = _chunk_t(p)
            if last:
                break
            bad.append(p)
        out["chunks"] = len(files)
        out["unreadable"] = sorted({p.name for p in bad})
        if first and last:
            out["first_t"], out["last_t"] = first[0], last[1]
    elif path.is_file():  # node_status.jsonl
        ts = []
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                ts.append(float(json.loads(line)["t"]))
            except (ValueError, KeyError, TypeError):
                out["bad_lines"] += 1
        out["rows"] = len(ts)
        if ts:
            out["first_t"], out["last_t"] = ts[0], ts[-1]
    if out["first_t"] is not None:
        out["span_s"] = round(out["last_t"] - out["first_t"], 2)
    return out


def _stream_status(id_, live: dict, disk: dict, ends: dict) -> tuple[str, str | None]:
    """present | partial with its reasons, from the files and the recorder's own counters."""
    why, notes = [], []
    if disk["unreadable"]:
        why.append(f"{len(disk['unreadable'])} chunk(s) unreadable (cut at the end of the recording?), skipped: "
                   + ", ".join(disk["unreadable"][:3]))
    if disk.get("bad_lines"):
        notes.append(f"{disk['bad_lines']} cut line(s) skipped")
    if live.get("dropped"):
        reasons = ", ".join(f"{k} {v}" for k, v in (live.get("drop_reasons") or {}).items())
        why.append(f"{live['dropped']} samples dropped by the recorder ({reasons or 'see rec/stats.json'})")
    span = disk.get("span_s") or 0.0
    gap_total = live.get("gap_total_s") or 0.0
    if gap_total > max(GAP_PARTIAL_S, GAP_PARTIAL_FRAC * span):
        why.append(f"silent {gap_total:.1f} s in {live.get('gaps')} gap(s) longer than {live.get('gap_s')} s")
    ref = TAIL_REFERENCE.get(id_)
    if ref and ends.get(ref) and disk.get("last_t") is not None:
        r_first, r_last = ends[ref]
        early = r_last - disk["last_t"]
        if early > max(TAIL_PARTIAL_S, TAIL_PARTIAL_FRAC * (r_last - r_first)):
            why.append(f"stopped {early:.0f} s before {ref}")
    if disk.get("unfinished"):
        notes.append(f"{disk['unfinished']} unfinished chunk write(s) (.tmp) ignored")
    return ("partial" if why else "present"), ("; ".join(why + notes) or None)


# ------------------------------------------------------------------------------------------------ the manifest
def manifest(d) -> dict:
    """The evidence of session directory `d` (a console session: robot, practice, rehearsal or sim). Manifest v2: the
    v1 fields (name, dir, t, complete, items[id, status, bytes, path, reason]) plus, per item, required, policy,
    span_s, gaps and dropped, and the session's state, kind, finalization and the reasons it is not complete."""
    d = Path(d)
    s = session_info(d)
    info, meta, reh, attempts = s["info"], s["meta"], s["rehearsal"], _rows(d / "attempts.jsonl")
    robot, practice, armed, kind = s["robot"], s["practice"], s["armed"], s["kind"]
    pol = policy(kind, s["lio"])
    rs = recorder_state(d, s)
    live = rs["stats"].get("streams") or {}
    rec = d / "rec"
    items = []

    def add(id_, path, status=None, reason=None, note=None, **extra):
        p = d / path if path else None
        size = _size(p) if p is not None else None
        if status is None:
            status = "present" if size else "missing"
            if size == 0:
                reason = f"{path} is empty"
        row = {"id": id_, "required": pol[id_] == "required" and status != "n/a", "policy": pol[id_],
               "status": status, "bytes": size if status in ("present", "partial") else None,
               "path": str(p) if p is not None else None, "reason": reason if status != "present" else note,
               "span_s": None, "gaps": None, "dropped": None}
        row.update(extra)
        items.append(row)
        return row

    # -- recorded streams
    disk = {id_: stream_on_disk(rec / sub) for id_, sub in REC_DIRS.items()}
    disk["node_status"] = stream_on_disk(rec / "node_status.jsonl")
    ends = {k: (v["first_t"], v["last_t"]) for k, v in disk.items() if v["first_t"] is not None}
    no_rec = "no recording (rec/meta.json absent: --no-record?)"

    def add_stream(id_, path, stream, why_missing):
        dk, lv = disk[id_], live.get(stream) or {}
        if not dk["exists"] or not dk["bytes"]:
            return add(id_, path, "missing", why_missing if meta else no_rec, gaps=lv.get("gaps"), dropped=lv.get("dropped"))
        if dk["first_t"] is None:
            return add(id_, path, "missing", "no readable chunk (" + ", ".join(dk["unreadable"][:3]) + ")"
                       if dk["unreadable"] else f"{path} has no readable sample", gaps=lv.get("gaps"),
                       dropped=lv.get("dropped"))
        status, why = _stream_status(id_, lv, dk, ends)
        return add(id_, path, status, why, note=why, span_s=dk["span_s"], gaps=lv.get("gaps"), dropped=lv.get("dropped"),
                   chunks=dk["chunks"] or dk["rows"], unreadable=dk["unreadable"] or None)

    add_stream("lowstate", "rec/lowstate", "lowstate", "the recorder saw no messages")
    if s["runner"] or (rec / "lowcmd").is_dir():
        add_stream("lowcmd", "rec/lowcmd", "lowcmd", "the recorder saw no LowCmd")
    else:
        add("lowcmd", "rec/lowcmd", "n/a", "the session ended before Arm (the runner never ran)")
    if meta and meta.get("lidar") is False:
        add("lidar_raw", "rec/lidar", "missing" if pol["lidar_raw"] == "required" else "n/a",
            "recorded without raw lidar (--no-record-lidar)")
    else:
        add_stream("lidar_raw", "rec/lidar", "lidar", "no raw lidar frames recorded")
    if not s["lio"]:
        why = f"odometry {s['odom'] or 'unknown'}: no LIO stream"
        add("lio_cloud", "rec/zmq_cloud", "n/a", why)
        add("lio_odom", "rec/zmq_odom", "n/a", why)
    else:
        zmq = " ".join(meta.get("zmq") or [])
        add_stream("lio_cloud", "rec/zmq_cloud", "zmq_cloud", "the recorder did not record the LIO cloud"
                   if "cloud=" not in zmq else "no LIO cloud messages recorded")
        add_stream("lio_odom", "rec/zmq_odom", "zmq_odom", "no LIO odometry messages recorded")
    if s["node"]:
        add_stream("node_status", "rec/node_status.jsonl", "node_status", "no node status recorded")
    else:
        add("node_status", "rec/node_status.jsonl", "n/a", "the session ended before Arm (no node ran)")
    # -- the node's, the console's and the hub's records
    if not s["node"]:
        for id_, path in (("node_events", "node.log"), ("plans", "node_plans.json"), ("node_memory", "node_memory.npz")):
            add(id_, path, "n/a", "the session ended before Arm (no node ran)")
    else:
        add("node_events", "node.log", None, "node.log missing")
        if (d / "node_plans.json").exists():
            add("plans", "node_plans.json")
        elif (d / "plans").is_dir():
            add("plans", "plans")
        else:
            add("plans", "node_plans.json", "missing", "the node wrote no plan log (it writes it at exit)")
        add("node_memory", "node_memory.npz", None, "no final node memory (written at node exit)")
    add("console_events", "console_events.jsonl", None, "this console writes no structured events")
    went = [r for r in _rows(d / "console_events.jsonl") if r.get("act") == "g" and r.get("result") == "done"]
    if (d / "attempts.jsonl").exists():
        add("attempts", "attempts.jsonl")
    elif went:
        add("attempts", "attempts.jsonl", "missing", f"GO was accepted {len(went)} time(s) but attempts.jsonl is missing")
    else:
        add("attempts", "attempts.jsonl", "n/a", "no attempt (GO never pressed)")
    if (d / "start_check.jsonl").exists():
        add("start_checks", "start_check.jsonl")
    elif reh is not None:
        add("start_checks", "start_check.jsonl", "n/a", "a rehearsal starts from its course's own start")
    elif robot or practice:
        add("start_checks", "start_check.jsonl", "missing" if attempts else "n/a",
            "no start check before a walk" if attempts else "no walk, no start check")
    else:
        add("start_checks", "start_check.jsonl", "n/a", "a sim session on a course")
    scan = None
    if (d / "scene_scan.txt").exists():
        scan = (d / "scene_scan.txt").read_text().strip()
    elif reh is not None:
        scan = reh.get("scan")
    if scan:
        sd = d.parent / "scans" / scan
        row = add("scan", None, "present" if sd.is_dir() else "missing",
                  None if sd.is_dir() else f"scan {scan} not found beside the session")
        row.update(path=str(sd), bytes=_size(sd) if sd.is_dir() else None)
    elif robot:
        add("scan", None, "missing", "no scan recorded for this session (scene_scan.txt)")
    else:
        add("scan", None, "n/a", "a sim session on a course")
    if reh is not None:
        add("rehearsal", "sim.npz", None, "the rehearsal's sim truth (sim.npz) is written when it ends")
    elif robot or practice:
        tied = [a for a in attempts if a.get("rehearsal") or a.get("rehearsal_override")]
        if tied:
            add("rehearsal", "attempts.jsonl")
        else:
            add("rehearsal", None, "missing" if attempts else "n/a",
                "no rehearsal set (or override) recorded with the attempts" if attempts else "no walk")
    else:
        add("rehearsal", None, "n/a", "not a rehearsal")
    if info.get("deploy_identity"):
        add("identity", "session.json")
    else:
        add("identity", "session.json", "missing", "session.json has no deploy identity")
    conf = info.get("confirm") or info.get("confirmations")
    if conf:
        add("operator_confirmations", "session.json")
    elif robot:
        add("operator_confirmations", "session.json", "missing", "the start checklist is not stored in session.json")
    else:
        add("operator_confirmations", None, "n/a", "simulation: no robot checklist")
    if armed:
        add("runner_log", "runner.log", None, "runner.log missing")
    else:
        add("runner_log", "runner.log", "n/a", "the session ended before Arm (no runner)")
    if (d / "report.md").exists():
        add("report", "report.md")
    elif s["closed"]:
        add("report", "report.md", "missing", "the session ended without a report")
    else:
        add("report", "report.md", "n/a", "written when the session ends")
    videos = sorted(d.glob("*.mp4"))
    if videos:
        add("video", videos[0].name)
    else:
        add("video", None, "n/a", "no video is recorded in v1")

    # -- the session's state and completeness
    console_alive = _pid_alive(s["console_pid"], b"console")
    running = (rs["state"] in ("recording", "stalled") or console_alive) and not s["closed"]
    has_rec = rs["state"] != "absent"
    missing = [i["id"] for i in items if i["required"] and i["status"] == "missing"]
    partial = [i["id"] for i in items if i["required"] and i["status"] == "partial"]
    why = []
    if running:
        state = "recording"
        why.append("the session is still running")
    else:
        state = "ended" if s["closed"] else "lost"
        if not s["closed"]:
            why.append("the console did not close the session (no console_summary.json): it was killed or lost power")
        if has_rec and not rs["finalized"]:
            why.append("the recording was not finalized (no rec/finalized.json): the recorder was killed or lost "
                       "power; everything up to its last whole chunk is usable")
    if missing:
        why.append("required records missing: " + ", ".join(missing))
    if partial:
        why.append("required records partial: " + ", ".join(partial))
    complete = not running and s["closed"] and (rs["finalized"] or not has_rec) and not missing and not partial
    return {"version": 2, "name": d.name, "dir": str(d), "t": time.time(), "complete": bool(complete), "state": state,
            "kind": kind, "mode": s["mode"], "odom": s["odom"], "finalized": rs["finalized"] if has_rec else None,
            "finalized_reason": rs["finalized_reason"], "recorder": rs["state"], "console_closed": s["closed"],
            "missing_required": missing, "partial_required": partial, "incomplete_reasons": why, "items": items}


def write(d) -> dict:
    m = manifest(d)
    p = Path(d) / "manifest.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1) + "\n")
    tmp.replace(p)
    return m


# ------------------------------------------------------------------------------------------------ live health
def _expected_hz(stream, live, sim):
    if "expected_hz" in live:
        return live["expected_hz"]
    from g1deploy import recorder as R

    if stream == "node_status":
        return 5.0
    return (R.SIM_HZ.get(stream) if sim else None) or R.NOMINAL_HZ.get(stream)


def recording_health(d) -> dict:
    """The recording of session `d` as it is now: {streams: [{id, stream, required, policy, active, state, rate_hz,
    expected_hz, last_write_age_s, last_sample_age_s, samples, dropped, gaps, gap_total_s, bytes, chunks,
    write_errors, pending}], disk_free_gb, errors, problems, ok, finalized, recorder_alive, state, ...}.

    Recording state: recording | stalled (its process runs, its heartbeat stopped) | finalized | not finalized (gone
    without finalizing) | absent. Stream state: active | waiting (required, no sample yet) | stalled (no sample for
    max(1 s, 3 expected periods) + the 1 s heartbeat) | stopped (the recording ended) | off (not recorded by
    configuration) | absent (optional, never seen) | n/a (does not apply to this session)."""
    d = Path(d)
    s = session_info(d)
    rs = recorder_state(d, s)
    stats, meta = rs["stats"], rs["meta"]
    live_streams = stats.get("streams") or {}
    pol = policy(s["kind"], s["lio"])
    sim = s["mode"] != "robot"
    now = time.time()
    hb_age = rs["heartbeat_age_s"] or 0.0
    recording = rs["state"] in ("recording", "stalled")
    zmq = " ".join(meta.get("zmq") or [])
    streams, problems = [], []
    for id_, stream in HEALTH_STREAMS:
        lv = live_streams.get(stream) or {}
        if id_ in pol:
            policy_ = pol[id_]
        else:
            policy_ = "required" if id_ == "sim_truth" and s["kind"] == "rehearsal" else "optional"
        applies, off = True, None
        if id_ == "lowcmd" and not s["runner"]:
            applies = False  # before Arm
        elif id_ == "node_status" and not s["node"]:
            applies = False
        elif id_ in ("lio_odom", "lio_cloud") and not s["lio"]:
            applies = False
        elif id_ == "sim_truth" and not sim:
            applies = False
        elif id_ == "lidar_raw" and meta.get("lidar") is False:
            off = "recorded without raw lidar (--no-record-lidar)"
        elif stream.startswith("zmq_") and meta and f"{stream[4:]}=" not in zmq:
            off = "not among the recorder's ZMQ streams"
        required = policy_ == "required" and applies
        expected = _expected_hz(stream, lv, sim)
        samples = lv.get("count") or 0
        age = lv.get("age_s")
        age_now = None if age is None else age + hb_age
        stale = max(1.0, 3.0 / expected) + 1.0 if expected else EVENT_STREAM_STALE_S
        if not applies:
            state = "n/a"
        elif off:
            state = "off"
        elif not recording:
            state = "stopped" if samples else "absent"
        elif not samples:
            state = "waiting" if required else "absent"
        else:
            state = "active" if age_now is not None and age_now <= stale else "stalled"
        lw = lv.get("last_write_unix")
        row = {"id": id_, "stream": stream, "required": required, "policy": policy_, "active": state == "active",
               "state": state, "rate_hz": lv.get("rate_hz"), "expected_hz": expected,
               "last_write_age_s": _r(now - lw) if lw else None, "last_sample_age_s": _r(age_now),
               "samples": samples, "dropped": lv.get("dropped") or 0, "gaps": lv.get("gaps") or 0,
               "gap_total_s": lv.get("gap_total_s") or 0.0, "max_gap_s": lv.get("max_gap_s"),
               "bytes": lv.get("bytes") or 0, "chunks": lv.get("chunks"), "write_errors": lv.get("write_errors") or 0,
               "pending": lv.get("pending") or 0, "reason": off}
        streams.append(row)
        if required and state == "stalled":
            problems.append(f"{id_}: no sample for {age_now if age_now is None else round(age_now, 1)} s "
                            f"(expected {expected or '-'} Hz)")
        elif required and state == "off":
            problems.append(f"{id_} is required but {off}")
        if row["dropped"]:
            reasons = ", ".join(f"{k} {v}" for k, v in (lv.get("drop_reasons") or {}).items())
            problems.append(f"{id_}: {row['dropped']} samples dropped ({reasons})")
    try:
        free = shutil.disk_usage(d if d.exists() else d.parent).free
    except OSError:
        free = (stats.get("disk") or {}).get("free_bytes")
    errors = [f"{k}: {v}" for k, v in (stats.get("errors") or {}).items()]
    if rs["state"] == "not finalized":
        problems.append("the recorder stopped without finalizing (killed, crashed or power loss): the data up to its "
                        "last whole chunk is on disk")
    elif rs["state"] == "stalled":
        problems.append(f"no recorder heartbeat for {hb_age:.0f} s while its process runs (disk stalled?)")
    elif rs["state"] == "absent" and not s["closed"]:
        problems.append("no recording for this session (yet)")
    busy = (stats.get("writer") or {}).get("busy_s")
    if recording and busy and busy > 5.0:
        problems.append(f"the chunk writer has been busy {busy:.0f} s (slow disk): its queues grow, then samples drop")
    if recording and stats.get("paused"):
        problems.append(f"recording paused ({stats['paused']}): samples are dropped and counted")
    if free is not None and free < DISK_WARN_GB * 1e9:
        problems.append(f"disk free {free / 1e9:.1f} GB")
    return {"dir": str(d), "name": d.name, "at": now, "kind": s["kind"], "mode": s["mode"], "odom": s["odom"],
            "state": rs["state"], "recorder_alive": rs["alive"], "finalized": rs["finalized"],
            "finalized_reason": rs["finalized_reason"], "clean": rs["clean"],
            "recorder": {"pid": rs["pid"], "pid_alive": rs["pid_alive"], "heartbeat_age_s": rs["heartbeat_age_s"],
                         "state": stats.get("state"), "elapsed_s": stats.get("elapsed_s"), "writer_busy_s": busy,
                         "paused": stats.get("paused"), "format": stats.get("format")},
            "disk_free_gb": _r(None if free is None else free / 1e9),
            "quota_gb": _r(stats["quota_bytes"] / 1e9) if stats.get("quota_bytes") else None,
            "bytes_total": stats.get("bytes_total") or sum(r["bytes"] for r in streams),
            "dropped_total": sum(r["dropped"] for r in streams), "errors": errors,
            "error_log": stats.get("error_log") or [], "problems": problems,
            "ok": rs["state"] in ("recording", "finalized") and not problems and not errors, "streams": streams}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fn = write if "--write" in sys.argv else recording_health if "--health" in sys.argv else manifest
    print(json.dumps(fn(args[0]), indent=1))
