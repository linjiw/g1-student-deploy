#!/usr/bin/env python3
"""Summarise one `g1 sim` / `g1 robot` session: what happened, how healthy every stream was, and what to look at.

Reads what the console, node, runner and recorder wrote into the session directory (each part optional):
  session.json, console_summary.json, node.log, node_plans.json, runner.log, rec/ (g1deploy/recorder.py),
  sim.npz (simulation truth), notes.jsonl (`g1 note`), runner_csv/ (runner --enable-csv-logs)
and writes report.json + report.md next to them. Gates are the ones the docs set (READINESS.md, HARDWARE.md):
planner p95 <= 0.32 s, lowstate >= 400 Hz on the robot, lidar ~10 Hz, one catch-up reset per stream start.

usage: session_report.py <session dir> [--quiet]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1deploy import contract as C  # noqa: E402
from g1deploy.recording import open_session  # noqa: E402

LATENCY_GATE_S = 0.32
TILT_ESTOP = 0.75


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def stamped_lines(path):
    """console Proc logs: '<monotonic> <text>' per line."""
    out = []
    try:
        for line in Path(path).read_text(errors="replace").splitlines():
            t, _, text = line.partition(" ")
            try:
                out.append((float(t), text))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def tilt_of(q):
    q = np.asarray(q, dtype=float)
    return np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1, 1))


def q_stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return None
    return {"n": int(len(x)), "p50": round(float(np.median(x)), 4), "p95": round(float(np.quantile(x, 0.95)), 4),
            "max": round(float(x.max()), 4)}


# ------------------------------------------------------------------------------------------- sections
def node_section(d: Path, t0):
    events = []
    for t, text in stamped_lines(d / "node.log"):
        if text.startswith('{"event"'):
            try:
                e = json.loads(text)
            except json.JSONDecodeError:
                continue
            e["t_rel"] = round(e.get("t", t) - t0, 2) if t0 is not None else None
            events.append(e)
    counts = {}
    for e in events:
        counts[e["event"]] = counts.get(e["event"], 0) + 1
    # legs: each 'prescan' (a GO) starts one, the next arrived / stand / estop / node_exit ends it. 'run' (the first
    # plan) is emitted only once per node process, so it cannot delimit walks.
    legs, cur, goal = [], None, None
    for e in events:
        k = e["event"]
        if k == "goal":
            goal = e.get("goal_odom")
            if cur is not None:
                cur["retargets"] = cur.get("retargets", 0) + 1
        if k == "prescan":
            cur = {"start_t": e["t_rel"], "goal_odom": None if goal is None else [round(v, 2) for v in goal]}
        elif k == "run" and cur is not None:
            cur["first_plan_s"] = round(float(e.get("first_plan_s", float("nan"))), 3)
        elif cur is not None and k in ("arrived", "stand", "estop", "node_exit", "stand_deferred"):
            if k == "stand_deferred":
                cur["stand_deferred"] = True
                continue
            cur.update(end={"arrived": "arrived", "stand": f"stand ({e.get('reason')})", "estop": f"estop ({e.get('reason')})",
                            "node_exit": f"node exit ({e.get('result')})"}[k], end_t=e["t_rel"])
            if k == "arrived":
                cur["arrive_dist_m"] = round(float(e.get("dist", float("nan"))), 3)
            cur["duration_s"] = round(cur["end_t"] - cur["start_t"], 2) if cur["start_t"] is not None else None
            legs.append(cur)
            cur = None
    if cur is not None:
        legs.append(dict(cur, end="still running at the end of the log"))
    plans = load_json(d / "node_plans.json", {}) or {}
    rec = plans.get("plans") if isinstance(plans, dict) else None
    lat = q_stats([p.get("latency_s") for p in rec]) if rec else None
    rules = {}
    for p in rec or []:
        rules[p.get("rule")] = rules.get(p.get("rule"), 0) + 1
    exit_ev = next((e for e in reversed(events) if e["event"] == "node_exit"), {})
    return {"events": counts, "legs": legs, "plan_latency_s": lat,
            "latency_gate": None if not lat else ("PASS" if lat["p95"] <= LATENCY_GATE_S else "FAIL"),
            "rules": rules, "rejected_plans": sum(1 for p in rec or [] if p.get("rejected")),
            "overruns": exit_ev.get("overruns"), "skipped_ticks": exit_ev.get("skipped"),
            "max_lag_s": exit_ev.get("max_lag_s"),
            "timeline": [f"{e['t_rel']:7.2f}  {e['event']}" + (f" ({e['reason']})" if "reason" in e else "")
                         for e in events if e["event"] not in ("dq_clipped", "stall_risk", "command")][:80]}


def runner_section(d: Path):
    lines = stamped_lines(d / "runner.log")
    text = [t for _, t in lines]
    timing = {"lowstate_age_ms": [], "policy_us": []}
    for t in text:
        if t.startswith("Loop timing"):
            m = re.search(r"LowState age: ([0-9.]+)ms", t)
            p = re.search(r"Policy: ([0-9]+)us", t)
            if m:
                timing["lowstate_age_ms"].append(float(m.group(1)))
            if p:
                timing["policy_us"].append(float(p.group(1)))
    return {"lines": len(text), "init_done": any("Init Done" in t for t in text),
            "control": any("transitioning to CONTROL" in t for t in text),
            "catch_up_resets": sum("Catch-up: Reset" in t for t in text),
            "emergency_stop": any(("EMERGENCY STOP" in t or "Emergency Stop" in t) for t in text),
            "emergency_stop_t": next((tt for tt, t in lines if "EMERGENCY STOP" in t or "Emergency Stop" in t), None),
            "stream_enabled": sum("ZMQ STREAMING MODE: ENABLED" in t for t in text),
            "errors": [t[:160] for t in text if re.search(r"\b(error|abort|fatal|exception)\b", t, re.I)][:10],
            "lowstate_age_ms": q_stats(timing["lowstate_age_ms"]), "policy_us": q_stats(timing["policy_us"])}


def recording_section(d: Path, mode, estop_t=None):
    if not (d / "rec").is_dir():
        return None
    rec = open_session(d)
    stats = load_json(rec.dir / "stats.json", {}) or {}
    out = {"streams": stats.get("streams", {}), "errors": stats.get("errors", {}), "checks": {}}
    names = rec.streams()
    low = rec.stream("lowstate", keys=["tick", "imu_quat", "q", "dq", "tau", "temp"]) if "lowstate" in names else {}
    if low.get("t") is not None and len(low["t"]) > 1:
        ticks = low["tick"].astype(np.int64)
        uniq = np.unique(ticks)
        step = int(np.median(np.diff(uniq))) if len(uniq) > 1 else 1
        expected = int(uniq[-1] - uniq[0]) // max(step, 1) + 1
        rate = (len(uniq) - 1) / (low["t"][-1] - low["t"][0])  # distinct robot samples per second
        out["checks"]["lowstate"] = {"rate_hz": round(float(rate), 1), "tick_step": step,
                                     "duplicates_pct": round(100 * (len(ticks) - len(uniq)) / len(ticks), 2),
                                     "missing_samples": int(expected - len(uniq)),
                                     "missing_pct": round(100 * (expected - len(uniq)) / max(expected, 1), 3),
                                     "gate": "n/a (sim)" if mode == "sim" else ("PASS" if rate >= 400 else "FAIL")}
        tilt = tilt_of(low["imu_quat"])
        before = low["t"] < estop_t if estop_t is not None else np.ones(len(tilt), bool)
        # after an E-STOP the robot drops into the gantry on purpose: judge tilt only before it
        out["checks"]["tilt_max_rad"] = round(float(tilt[before].max()), 3) if before.any() else None
        if estop_t is not None:
            out["checks"]["tilt_max_after_estop_rad"] = round(float(tilt[~before].max()), 3) if (~before).any() else None
        temp = low["temp"][:, :29, 0].astype(float)
        hot = np.argsort(temp.max(0))[::-1][:5]
        if temp.max() > 0:  # the simulator reports no temperatures
            out["checks"]["motor_temp_max_c"] = {C.MJ_JOINTS[i]: float(temp[:, i].max()) for i in hot}
            out["checks"]["motor_temp_rise_c"] = round(float((temp[-1] - temp[0]).max()), 1)
        tau = np.abs(low["tau"][:, :29])
        top = np.argsort(tau.max(0))[::-1][:5]
        out["checks"]["tau_est_max_nm"] = {C.MJ_JOINTS[i]: round(float(tau[:, i].max()), 1) for i in top}
    if "lowcmd" in names and low.get("t") is not None:
        cmd = rec.stream("lowcmd", keys=["q", "kp"])
        if len(cmd.get("t", [])) > 1:
            j = np.clip(np.searchsorted(low["t"], cmd["t"]), 0, len(low["t"]) - 1)
            active = (cmd["kp"][:, :29] > 0).all(1)
            if active.any():
                err = cmd["q"][active, :29] - low["q"][j[active], :29]
                rms = np.sqrt((err ** 2).mean(0))
                worst = np.argsort(rms)[::-1][:5]
                out["checks"]["pd_tracking_rms_rad"] = {
                    "all_joints": round(float(np.sqrt((err ** 2).mean())), 4),
                    "worst": {C.MJ_JOINTS[i]: round(float(rms[i]), 4) for i in worst},
                    "basis": "lowcmd q target vs the next lowstate q (the PD's own tracking; SONIC's reference is in zmq_pose)"}
            out["checks"]["lowcmd_rate_hz"] = round(float((len(cmd["t"]) - 1) / (cmd["t"][-1] - cmd["t"][0])), 1)
    if "lidar" in names:
        li = rec.stream("lidar", keys=["stamp_sec", "stamp_nanosec", "width"])
        if len(li.get("t", [])) > 1:
            stamps = li["stamp_sec"] + li["stamp_nanosec"] * 1e-9
            ds = np.diff(stamps)
            nominal = float(np.median(ds))
            out["checks"]["lidar"] = {"rate_hz": round(float((len(li["t"]) - 1) / (li["t"][-1] - li["t"][0])), 2),
                                      "frames": int(len(li["t"])), "nominal_period_s": round(nominal, 3),
                                      "skipped_at_source": int(np.round(ds[ds > 1.5 * nominal] / nominal - 1).sum()),
                                      "points_per_frame": int(np.median(li["width"])),
                                      "header_minus_recv_s": round(float(np.median(stamps - li["t_unix"])), 3),
                                      "gate": "PASS" if (len(li["t"]) - 1) / (li["t"][-1] - li["t"][0]) >= 8 else
                                      ("CHECK (sim runs below real time while recording)" if mode == "sim" else "CHECK")}
    status = rec.node_status()
    if status:
        out["checks"]["node_status_samples"] = len(status)
    return out


def truth_section(d: Path, session, status):
    sim = d / "sim.npz"
    if not sim.exists():
        return None
    z = np.load(sim)
    pos, contact = z["root_pos"], z["contact_n"]
    goal = np.asarray(session.get("goal") or [np.nan, np.nan], dtype=float)
    path = float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum())
    out = {"path_m": round(path, 2), "final_xy": np.round(pos[-1, :2], 3).tolist(),
           "min_pelvis_z_m": round(float(pos[:, 2].min()), 3), "fell": bool(pos[:, 2].min() < 0.38),
           "max_obstacle_contact_n": round(float(contact.max()), 1),
           "launch_goal_true_dist_m": round(float(np.linalg.norm(pos[-1, :2] - goal)), 3)}
    low = np.flatnonzero(pos[:, 2] < 0.38)
    out["fall_t"] = float(z["wall"][low[0]]) if len(low) and "wall" in z.files else None
    if status:  # odometry error: node's own xy vs simulator truth at the same instant
        rec = open_session(d)
        odo = rec.stream("odostate", keys=["position"]) if "odostate" in rec.streams() else {}
        if len(odo.get("t", [])):
            errs = []
            for s in status:
                xy = s["status"].get("xy")
                if not xy:
                    continue
                j = min(int(np.searchsorted(odo["t"], s["t"])), len(odo["t"]) - 1)
                errs.append(float(np.linalg.norm(np.asarray(xy) - odo["position"][j, :2])))
            if errs:
                out["odometry_error_m"] = {"final": round(errs[-1], 3), "max": round(max(errs), 3),
                                           "basis": "node status xy vs rt/odostate (simulator truth)"}
    return out


def build(d: Path):
    d = Path(d)
    session = load_json(d / "session.json", {}) or {}
    summary = load_json(d / "console_summary.json", {}) or {}
    t0 = session.get("t0")
    mode = session.get("mode", "unknown")
    rep = {"session": str(d), "mode": mode, "task": session.get("task"), "launch_goal": session.get("goal"),
           "meta": {k: session.get(k) for k in ("robot", "operator", "note", "git", "bundle_manifest_sha256",
                                                "runner_commit", "mount_calib", "argv", "host", "started")
                    if k in session},
           "console_events": summary.get("events", []), "final_status": summary.get("status", {})}
    rep["node"] = node_section(d, t0)
    rep["runner"] = runner_section(d)
    rep["recording"] = recording_section(d, mode, rep["runner"]["emergency_stop_t"])
    rec_status = open_session(d).node_status() if (d / "rec").is_dir() else []
    rep["truth"] = truth_section(d, session, rec_status)
    notes = d / "notes.jsonl"
    rep["notes"] = [json.loads(l) for l in notes.read_text().splitlines() if l.strip()] if notes.exists() else []
    rep["runner_csv"] = sorted(p.name for p in (d / "runner_csv").rglob("*.csv"))[:40] if (d / "runner_csv").is_dir() else None
    rep["files"] = {p.name: p.stat().st_size for p in sorted(d.iterdir()) if p.is_file()}
    rep["size_mb"] = round(sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e6, 1)
    rep["flags"] = flags(rep)
    return rep


def flags(rep):
    """Things an operator or researcher should look at, worst first."""
    out = []
    n, r, rec, tr = rep["node"], rep["runner"], rep.get("recording") or {}, rep.get("truth") or {}
    if n["events"].get("estop"):
        out.append("node supervisor E-STOP fired: read the timeline")
    if r["emergency_stop"]:
        out.append("runner reached EMERGENCY STOP / damping")
    if tr.get("fell"):
        after_estop = r["emergency_stop_t"] is not None and tr.get("fall_t") and tr["fall_t"] >= r["emergency_stop_t"] - 0.2
        out.append("simulated robot dropped after the E-STOP (expected: damping)" if after_estop
                   else "simulated robot FELL (pelvis below 0.38 m) before any E-STOP")
    if (tr.get("odometry_error_m") or {}).get("max", 0) > 0.25:
        out.append(f"odometry error up to {tr['odometry_error_m']['max']} m vs truth: arrival was judged on a wrong pose")
    if tr.get("max_obstacle_contact_n", 0) > 5:
        out.append(f"obstacle contact {tr['max_obstacle_contact_n']} N (> 5 N Isaac threshold)")
    if n["plan_latency_s"] and n["latency_gate"] == "PASS" and n["plan_latency_s"]["p95"] > 0.8 * LATENCY_GATE_S:
        out.append(f"planner p95 {n['plan_latency_s']['p95']} s is within 20 % of the {LATENCY_GATE_S} s gate")
    if n["latency_gate"] == "FAIL":
        out.append(f"planner p95 {n['plan_latency_s']['p95']} s > {LATENCY_GATE_S} s gate")
    if n["events"].get("plan_dropped"):
        out.append(f"{n['events']['plan_dropped']} plans dropped (sanity/stale)")
    if n["events"].get("stand_deferred"):
        out.append("stand deferred near a remembered ceiling (phantom or real overhead cells)")
    starts = max(r["stream_enabled"], len(n["legs"]))  # every walk opens a new stream window: one reset each
    if starts and r["catch_up_resets"] > starts:
        out.append(f"{r['catch_up_resets']} catch-up resets for {starts} stream starts/walks: stream fell behind")
    chk = rec.get("checks", {})
    if chk.get("lowstate", {}).get("gate") == "FAIL":
        out.append(f"lowstate {chk['lowstate']['rate_hz']} Hz < 400 Hz")
    if chk.get("lowstate", {}).get("missing_pct", 0) > 1:
        out.append(f"{chk['lowstate']['missing_pct']} % lowstate samples missing in the recording")
    if str(chk.get("lidar", {}).get("gate", "")).startswith("CHECK"):
        out.append(f"lidar {chk['lidar']['rate_hz']} Hz (expected ~10)")
    if chk.get("motor_temp_max_c") and max(chk["motor_temp_max_c"].values()) >= 70:
        out.append(f"motor temperature reached {max(chk['motor_temp_max_c'].values()):.0f} °C (runner cuts off at 90 °C)")
    if (chk.get("tilt_max_rad") or 0) > 0.5:
        out.append(f"pelvis tilt reached {chk['tilt_max_rad']} rad (E-stop at {TILT_ESTOP})")
    if rec.get("errors"):
        out.append(f"recorder errors: {rec['errors']}")
    if rep.get("recording") is None:
        out.append("no rec/ directory: raw robot data was not recorded")
    return out


def markdown(rep):
    L = [f"# Session report: {Path(rep['session']).name}", "",
         f"- mode **{rep['mode']}**, task `{rep['task']}`, launch goal {rep['launch_goal']}, size {rep['size_mb']} MB"]
    for k, v in rep["meta"].items():
        L.append(f"- {k}: `{v}`")
    L += ["", "## Look at first", ""] + ([f"- {f}" for f in rep["flags"]] or ["- nothing flagged"])
    n = rep["node"]
    L += ["", "## Walks (node legs)", "", "| # | start s | end | duration s | arrive dist m | retargets |",
          "|---|---|---|---|---|---|"]
    for i, leg in enumerate(n["legs"], 1):
        L.append(f"| {i} | {leg.get('start_t')} | {leg.get('end')} | {leg.get('duration_s')} | "
                 f"{leg.get('arrive_dist_m', '')} | {leg.get('retargets', 0)} |")
    L += ["", f"Planner latency: {n['plan_latency_s']} → gate **{n['latency_gate']}** (p95 ≤ {LATENCY_GATE_S} s). "
          f"Rules: {n['rules']}. Rejected plans: {n['rejected_plans']}. Loop overruns {n['overruns']}, "
          f"skipped ticks {n['skipped_ticks']}, max lag {n['max_lag_s']} s.", "",
          "Node events: " + ", ".join(f"{k} {v}" for k, v in sorted(n["events"].items()))]
    r = rep["runner"]
    L += ["", "## Runner", "", f"Init done {r['init_done']}, control {r['control']}, stream starts {r['stream_enabled']}, "
          f"catch-up resets {r['catch_up_resets']}, emergency stop {r['emergency_stop']}. LowState age ms "
          f"{r['lowstate_age_ms']}, policy µs {r['policy_us']}."]
    if r["errors"]:
        L += [""] + [f"    {e}" for e in r["errors"]]
    rec = rep.get("recording")
    if rec:
        L += ["", "## Recorded streams", "", "| stream | samples | rate Hz | max gap s | MB |", "|---|---|---|---|---|"]
        for k, v in rec["streams"].items():
            L.append(f"| {k} | {v.get('count')} | {v.get('rate_hz', '')} | {v.get('max_gap_s', '')} | "
                     f"{round((v.get('bytes') or 0) / 1e6, 1)} |")
        L += ["", "```json", json.dumps(rec["checks"], indent=1), "```"]
    if rep.get("truth"):
        L += ["", "## Simulation truth", "", "```json", json.dumps(rep["truth"], indent=1), "```"]
    if rep["notes"]:
        L += ["", "## Operator notes", ""] + [f"- {x.get('time', '')} {x.get('text', '')}" for x in rep["notes"]]
    L += ["", "## Node timeline (s from console start)", "", "```"] + n["timeline"] + ["```"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    d = Path(args.session)
    rep = build(d)
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    (d / "report.md").write_text(markdown(rep))
    if not args.quiet:
        legs = rep["node"]["legs"]
        print(f"report: {d / 'report.md'}")
        print(f"  walks: {len(legs)}  " + "  ".join(f"[{l.get('end')} {l.get('duration_s')} s]" for l in legs))
        print(f"  planner latency {rep['node']['plan_latency_s']}  gate {rep['node']['latency_gate']}")
        if rep.get("recording"):
            c = rep["recording"]["checks"]
            print(f"  lowstate {c.get('lowstate', {}).get('rate_hz')} Hz, lidar {c.get('lidar', {}).get('rate_hz')} Hz, "
                  f"tilt max {c.get('tilt_max_rad')} rad, size {rep['size_mb']} MB")
        for f in rep["flags"]:
            print(f"  ! {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
