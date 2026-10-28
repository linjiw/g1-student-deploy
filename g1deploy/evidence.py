"""What a session directory holds as evidence: manifest.json, one row per kind of record, each present, missing or
n/a with its size and the reason (the hub writes it when a session ends and on request; the console and the recorder
write the files it lists).

  python -m g1deploy.evidence SESSION_DIR [--write]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ITEMS = ("lowstate", "lowcmd", "lidar_raw", "lio_cloud", "lio_odom", "node_status", "node_events", "plans",
         "node_memory", "console_events", "attempts", "start_checks", "scan", "rehearsal", "identity",
         "operator_confirmations", "runner_log", "report", "video")
LIO_ODOM = ("zmq", "fused", "lio")  # odometry modes with a LIO pose stream (and its cloud)


def _size(p: Path):
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
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


def manifest(d) -> dict:
    """The evidence of session directory `d` (a console session: robot, practice, rehearsal or sim)."""
    d = Path(d)
    info = _json(d / "session.json") or {}
    meta = _json(d / "rec" / "meta.json") or {}
    reh = _json(d / "rehearsal.json")
    mode = info.get("mode") or ("sim" if (d / "sim.npz").exists() or reh else None)
    robot = mode == "robot"
    odom = info.get("odom")
    armed = (d / "runner.log").exists() or (d / "node.log").exists()
    attempts = _rows(d / "attempts.jsonl")
    practice = bool(info.get("practice") or (info.get("hub") or {}).get("practice") or d.name.startswith("practice-"))
    items = []

    def add(id_, path, status=None, reason=None):
        p = d / path if path else None
        size = _size(p) if p is not None else None
        if status is None:
            status = "present" if size else "missing"
            if size == 0:
                reason = f"{path} is empty"
        items.append({"id": id_, "status": status, "bytes": size if status == "present" else None,
                      "path": str(p) if p is not None and status == "present" else (str(p) if p is not None else None),
                      "reason": reason if status != "present" else None})

    rec = d / "rec"
    for id_, sub in (("lowstate", "lowstate"), ("lowcmd", "lowcmd")):
        if (rec / sub).is_dir():
            add(id_, f"rec/{sub}")
        elif not meta:
            add(id_, f"rec/{sub}", "missing", "no recording (rec/meta.json absent: --no-record?)")
        else:
            add(id_, f"rec/{sub}", "missing", "the recorder saw no messages" if id_ == "lowstate" else
                "no LowCmd recorded (the runner never ran)" if not armed else "the recorder saw no LowCmd")
    if meta and meta.get("lidar") is False:
        add("lidar_raw", "rec/lidar", "n/a", "recorded without raw lidar (--no-record-lidar)")
    elif (rec / "lidar").is_dir():
        add("lidar_raw", "rec/lidar")
    else:
        add("lidar_raw", "rec/lidar", "missing", "no raw lidar frames recorded")
    if odom not in LIO_ODOM:
        why = f"odometry {odom or 'unknown'}: no LIO stream"
        add("lio_cloud", "rec/zmq_cloud", "n/a", why)
        add("lio_odom", "rec/zmq_odom", "n/a", why)
    else:
        zmq = " ".join(meta.get("zmq") or [])
        if (rec / "zmq_cloud").is_dir():
            add("lio_cloud", "rec/zmq_cloud")
        else:
            add("lio_cloud", "rec/zmq_cloud", "missing", "the recorder did not record the LIO cloud" if "cloud=" not in zmq
                else "no LIO cloud messages recorded")
        add("lio_odom", "rec/zmq_odom", None, "no LIO odometry messages recorded")
    if armed:
        add("node_status", "rec/node_status.jsonl", None, "no node status recorded")
    else:
        add("node_status", "rec/node_status.jsonl", "n/a", "the session ended before Arm (no node ran)")
    if not armed:
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
    if (d / "attempts.jsonl").exists():
        add("attempts", "attempts.jsonl")
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
        items.append({"id": "scan", "status": "present" if sd.is_dir() else "missing",
                      "bytes": _size(sd) if sd.is_dir() else None, "path": str(sd),
                      "reason": None if sd.is_dir() else f"scan {scan} not found beside the session"})
    elif robot:
        add("scan", None, "missing", "no scan recorded for this session (scene_scan.txt)")
    else:
        add("scan", None, "n/a", "a sim session on a course")
    if reh is not None:
        add("rehearsal", "sim.npz", None, "the rehearsal's sim truth (sim.npz) is written when it ends")
    elif robot or practice:
        tied = [a for a in attempts if a.get("rehearsal")]
        if tied:
            add("rehearsal", "attempts.jsonl")
        else:
            add("rehearsal", None, "missing" if attempts else "n/a",
                "no rehearsal set recorded with the attempts" if attempts else "no walk")
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
    elif (d / "console_summary.json").exists():
        add("report", "report.md", "missing", "the session ended without a report")
    else:
        add("report", "report.md", "n/a", "written when the session ends")
    videos = sorted(d.glob("*.mp4"))
    if videos:
        add("video", videos[0].name)
    else:
        add("video", None, "n/a", "no video is recorded in v1")
    return {"name": d.name, "dir": str(d), "t": time.time(), "complete": not any(i["status"] == "missing" for i in items),
            "items": items}


def write(d) -> dict:
    m = manifest(d)
    p = Path(d) / "manifest.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1) + "\n")
    tmp.replace(p)
    return m


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = (write if "--write" in sys.argv else manifest)(args[0])
    print(json.dumps(out, indent=1))
