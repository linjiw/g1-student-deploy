"""Is this PC quiet enough for a real-time run? A snapshot of its load, taken before a session starts.

The real-time chain (simulator, SONIC runner, node) shares the PC with whatever else runs. On 2026-10-26 three
offline replays running beside a batch of `g1 sim` runs (load ~17 on 28 cores) doubled the node's loop lag
(0.2 -> 0.4-0.7 s) and plan latency (0.08 -> 0.12-0.14 s) and produced an E-stop 2.9 s into a walk; 17 runs had to be
redone (DEPLOY_KIT.md §11). The console records this snapshot in every session (`session.json` "machine") and, by
policy (`--busy`), warns, waits or refuses when the PC is busy. Another deploy chain already running is always refused
(two chains on one DDS domain cross-talk), unless the check is off.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

LOAD_FRAC = 0.35  # 1-min load average above this share of the cores = busy (the bad batch: 17 / 28 = 0.6)
GPU_UTIL = 60  # % GPU utilisation (median of a short sample) above this = busy; idle desktop + lidar stack ~35 %
# another real-time chain: two on one DDS domain cross-talk (and on the robot, two runners)
CHAIN = (r"sim/realtime/robot_sim\.py", r"g1deploy/node\.py", r"g1_deploy_onnx_ref", r"g1deploy/console\.py")
# heavy offline jobs of this kit that compete with a real-time run; G1_HEAVY_JOBS adds regexes (comma-separated)
HEAVY = (r"tools/replay_plans\.py", r"tools/bench_planner\.py", r"tools/profile_\w+\.py", r"tools/render_session\.py",
         r"tools/debug_plan\.py", r"tools/lio_replay\.py")


def heavy_patterns():
    extra = [p.strip() for p in os.environ.get("G1_HEAVY_JOBS", "").split(",") if p.strip()]
    return HEAVY + tuple(extra)


def _procs(skip=()):
    """(pid, cmdline) of every process but the skipped ones (Linux /proc)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in skip:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if cmd:
            out.append((int(d.name), cmd))
    return out


def _ancestors(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return seen


def _gpu(samples=3, gap_s=0.2):
    """Median GPU utilisation (%), memory used (MiB) and compute processes, or None without nvidia-smi."""
    util, mem = [], None
    try:
        for i in range(samples):
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            u, m = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
            util.append(u)
            mem = m
            if i + 1 < samples:
                time.sleep(gap_s)
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        apps = []
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                apps.append({"pid": int(parts[0]), "name": parts[1], "mem_mib": float(parts[2] or 0)})
    except Exception:  # no GPU tools: the snapshot says so instead of guessing
        return None
    util.sort()
    return {"util_pct": util[len(util) // 2], "mem_used_mib": mem, "compute_apps": apps}


def snapshot(own_pid=None, gpu=True):
    """The PC's load now: load averages, cores, GPU, other deploy chains and heavy kit jobs (own process tree
    excluded)."""
    skip = _ancestors(own_pid or os.getpid())
    procs = _procs(skip)
    find = lambda pats: [{"pid": p, "cmd": c[:160]} for p, c in procs if any(re.search(x, c) for x in pats)]  # noqa: E731
    load1, load5, load15 = os.getloadavg()
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load": [round(load1, 2), round(load5, 2), round(load15, 2)],
            "cpus": os.cpu_count(), "gpu": _gpu() if gpu else None, "chains": find(CHAIN), "heavy": find(heavy_patterns())}


def busy_reasons(snap, load_frac=LOAD_FRAC, gpu_util=GPU_UTIL):
    """Why the PC is not quiet (empty = quiet). The first reason, if it is another chain, is a hard refusal."""
    out = []
    if snap.get("chains"):
        out.append("another deploy chain is running: " + "; ".join(f"pid {c['pid']} {c['cmd'][:60]}" for c in snap["chains"]))
    if snap.get("heavy"):
        out.append("heavy offline jobs: " + "; ".join(f"pid {c['pid']} {c['cmd'][:60]}" for c in snap["heavy"]))
    cpus = snap.get("cpus") or 1
    if snap.get("load") and snap["load"][0] > load_frac * cpus:
        out.append(f"load {snap['load'][0]} on {cpus} cores (> {load_frac:.0%})")
    g = snap.get("gpu")
    if g and g.get("util_pct") is not None and g["util_pct"] > gpu_util:
        out.append(f"GPU {g['util_pct']:.0f} % busy (> {gpu_util} %)")
    return out


def check(policy="warn", wait_s=300.0, poll_s=10.0, log=print, own_pid=None):
    """Apply the `--busy` policy. Returns (snapshot, reasons, ok). warn: go on, reasons recorded; wait: re-check
    every poll_s until quiet or wait_s passes, then refuse; refuse: stop when busy; off: no check (ok, no snapshot).
    Another chain running refuses under every policy but off."""
    if policy == "off":
        return None, [], True
    t_end = time.monotonic() + wait_s
    while True:
        snap = snapshot(own_pid)
        reasons = busy_reasons(snap)
        if not reasons:
            return snap, [], True
        hard = bool(snap.get("chains"))
        if policy == "wait" and not hard and time.monotonic() < t_end:
            log(f"PC busy ({'; '.join(reasons)}): waiting for it to settle")
            time.sleep(poll_s)
            continue
        ok = policy == "warn" and not hard
        return snap, reasons, ok


LAG_SUSPECT_S = 0.35  # node loop lag above this: the run's timing is suspect (clean runs ~0.2 s, busy PC 0.4-0.7 s)
