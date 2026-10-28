"""Robot-day workspace for the operator panel: the safety ladder, the live checks and the near-body reading of the
node's map, from the console's state. Pure functions of one context dict (Session.workspace_ctx), so the panel, the
tests and a dry run in simulation all see the same logic (tests/test_workspace.py).

The ladder is the robot-day procedure of docs/DEPLOY_KIT.md as live steps: systems up, Arm, the stand ramp, Start
SONIC, the planner warm, Clear map, a goal, GO, the walk, the operator's tag. The first step not done is the one to do
now; the panel puts its button first. Nothing here sends anything: the console's existing actions do, with their own
refusals (GO before CONTROL, before the planner is warm, in E-STOP).
"""

from __future__ import annotations

import json
import math

from g1deploy.machine import LAG_SUSPECT_S

TEMP_WARN_C, TEMP_BAD_C = 60, 75  # ankles reached 80-83 C after ~10 min standing (2026-10-24); the runner cuts at 90
LATENCY_GATE_S = 0.25
NEAR_BODY_M = 0.5  # obstacle evidence this close to the pelvis, outside the rear blind: a person, the gantry, a leak
BEHIND_M = (1.0, 1.6)  # tall returns here behind: the spotter or rope just outside the blind (kcsc backs up)
CLEAR_MAP_SCAN_S = 10.0  # node.CLEAR_MAP_SCAN_S: a walk waits for this much fresh scan after Clear map


MAP_CODES = {" ": "unknown", ".": "ground", "1": "low < 0.3 m", "2": "obstacle < 1.0 m", "3": "tall >= 1.0 m",
             "^": "overhead over free floor", "s": "obstacle seen only from up close (self-return signature)"}


def strict(obj):
    """`obj` as plain JSON values with every non-finite float as None (numpy arrays and scalars become lists and
    numbers). The kit's v1 endpoints (node `view`, panel `/state` and `/act`) answer strict JSON: a NaN in a pose or a
    goal distance, likely exactly when odometry fails, must not break a browser's JSON.parse."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict(v) for v in obj]
    if hasattr(obj, "tolist"):  # a numpy array or scalar
        return strict(obj.tolist())
    return str(obj)


def dumps(obj) -> str:
    """Strict, compact JSON (strict())."""
    return json.dumps(strict(obj), allow_nan=False, separators=(",", ":"))


def code_grid(support, ceiling, lo, res, center_xy, half_m=4.0, res_m=0.1, near=None, lock=None):
    """A memory's support / ceiling layers around `center_xy` as one character per `res_m` cell (MAP_CODES; the
    highest support and the lowest ceiling of the memory cells inside), world axes: rows[i][j] is the cell at
    x0 + (i + 0.5) res, y0 + (j + 0.5) res. `near`: the memory's near-only mask (cells seen only from up close).
    `lock` is held while the window is copied (the node's perception thread writes the layers)."""
    import contextlib

    import numpy as np

    n = support.shape[0]
    k = max(1, int(round(res_m / res)))
    rr = k * res
    w = int(math.ceil(2 * half_m / rr)) * k  # memory cells per side, a multiple of k
    i0 = int(math.floor((center_xy[0] - half_m - lo[0]) / res))
    j0 = int(math.floor((center_xy[1] - half_m - lo[1]) / res))
    sup, ceil = np.full((w, w), np.nan), np.full((w, w), np.nan)
    nr = np.zeros((w, w), dtype=bool)
    a0, a1, b0, b1 = max(i0, 0), min(i0 + w, n), max(j0, 0), min(j0 + w, n)
    if a1 > a0 and b1 > b0:
        with lock if lock is not None else contextlib.nullcontext():
            sup[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = support[a0:a1, b0:b1]
            ceil[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = ceiling[a0:a1, b0:b1]
            if near is not None:
                nr[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = (near() if callable(near) else near)[a0:a1, b0:b1]
    m = w // k
    with np.errstate(invalid="ignore"):
        obs = sup > 0.05
    n_obs = obs.reshape(m, k, m, k).sum(axis=(1, 3))
    n_near = (obs & nr).reshape(m, k, m, k).sum(axis=(1, 3))
    hi = np.where(np.isnan(sup), -np.inf, sup).reshape(m, k, m, k).max(axis=(1, 3))
    lo_c = np.where(np.isnan(ceil), np.inf, ceil).reshape(m, k, m, k).min(axis=(1, 3))
    code = np.full((m, m), " ")
    code[hi > -np.inf] = "."
    code[(hi > 0.05) & (hi < 0.3)] = "1"
    code[(hi >= 0.3) & (hi < 1.0)] = "2"
    code[hi >= 1.0] = "3"
    code[np.isfinite(lo_c) & (hi <= 0.05)] = "^"
    code[(n_obs > 0) & (n_near == n_obs)] = "s"  # every obstacle return of the cell came from closer than near_m
    return {"rows": ["".join(r) for r in code], "res": rr, "half": m * rr / 2, "x0": float(lo[0] + i0 * res),
            "y0": float(lo[1] + j0 * res), "codes": MAP_CODES}


def near_body(m, near_m=NEAR_BODY_M, behind=BEHIND_M):
    """Obstacle cells of a node map (StudentNode.map_snapshot) near the robot, in the robot's frame (ahead, left)."""
    if not m or not m.get("ok"):
        return None
    x, y, yaw = m["robot"]
    res, x0, y0 = m["res"], m["x0"], m["y0"]
    c, s = math.cos(yaw), math.sin(yaw)
    half_blind = 0.5 * float(m.get("rear_blind_deg") or 0.0)
    blind_r = float(m.get("rear_blind_range_m") or 0.0)
    close, back, self_like = [], [], 0
    for i, row in enumerate(m["rows"]):
        for j, ch in enumerate(row):
            if ch == "s":
                self_like += 1
            if ch not in "23":
                continue
            dx, dy = x0 + (i + 0.5) * res - x, y0 + (j + 0.5) * res - y
            ahead, left = c * dx + s * dy, -s * dx + c * dy
            r = math.hypot(ahead, left)
            az = math.degrees(math.atan2(left, ahead))
            in_blind = half_blind > 0 and abs(az) > 180.0 - half_blind and (blind_r <= 0 or r < blind_r)
            if r < near_m and not in_blind:
                close.append((round(ahead, 2), round(left, 2)))
            elif ch == "3" and abs(az) > 90.0 and behind[0] <= r < behind[1]:
                back.append((round(ahead, 2), round(left, 2)))
    return {"close": len(close), "close_cells": close[:20], "behind_tall": len(back), "near_m": near_m,
            "behind_m": list(behind), "self_like": self_like}


def _check(key, label, level, detail=""):
    return {"key": key, "label": label, "level": level, "detail": detail}


def checks(ctx):
    """Live GO/NO-GO items: level ok | warn | bad | na (not applicable here)."""
    robot = ctx["mode"] == "robot"
    ns, rec = ctx.get("node_status") or {}, ctx.get("rec_stats") or {}
    streams = rec.get("streams", {}) if isinstance(rec, dict) else {}
    out = []
    busy = ctx.get("machine_busy") or []
    out.append(_check("pc", "Deploy PC quiet", "warn" if busy else "ok",
                      "; ".join(busy) if busy else "no heavy jobs; real-time numbers are trustworthy"))
    procs = ctx.get("procs") or {}
    if "rec" in procs:
        low = streams.get("lowstate", {})
        rate, age = low.get("rate_hz"), low.get("age_s")
        if not procs["rec"]:
            out.append(_check("rec", "Recording", "bad", "the recorder stopped: this session is not being recorded"))
        elif rec.get("errors"):
            out.append(_check("rec", "Recording", "warn", f"recorder errors: {rec.get('errors')}"))
        else:
            out.append(_check("rec", "Recording", "ok" if rate else "warn",
                              f"lowstate {rate:.0f} Hz" if rate else "waiting for the first streams"))
        if robot or rate:
            fresh = rate and age is not None and age < 0.5
            out.append(_check("lowstate", "Robot state (lowstate)", "ok" if fresh and rate > (200 if robot else 50) else "bad",
                              f"{rate:.0f} Hz" if fresh else "no fresh lowstate: robot link down?"))
    else:
        out.append(_check("rec", "Recording", "na", "recording off (--no-record)"))
    age = ns.get("cloud_age_s")
    if age is not None and age < 1e8:
        lvl = "ok" if age < 0.3 else "warn" if age < 1.0 else "bad"
        out.append(_check("lidar", "Lidar to the planner", lvl, f"latest cloud {age:.2f} s old"
                          + (f", {ns['lidar_frames']} frames" if ns.get("lidar_frames") else "")))
    else:
        li = streams.get("lidar", {})
        if li.get("rate_hz") and (li.get("age_s") or 9) < 1.0:
            out.append(_check("lidar", "Lidar", "ok", f"{li['rate_hz']:.0f} Hz recorded (planner not up yet)"))
        else:
            out.append(_check("lidar", "Lidar", "warn" if not ctx.get("node_ready") else "bad",
                              "no lidar frames yet" + ("" if ctx.get("node_ready") else " (the planner starts on Arm)")))
    if robot:
        rm = ctx.get("remote") or {}
        out.append(_check("remote", "Wireless remote (Select = E-STOP)", "ok" if rm.get("connected") else "bad",
                          f"connected{', last ' + rm['last'] if rm.get('last') else ''}" if rm.get("connected")
                          else "not seen: the remote cannot stop the robot; check it is on and paired"))
        t = ctx.get("temps")
        if t:
            hot = max(v for v in (t.get("ankle_c"), t.get("knee_c"), t.get("max_c")) if v is not None)
            lvl = "ok" if hot < TEMP_WARN_C else "warn" if hot < TEMP_BAD_C else "bad"
            out.append(_check("temps", "Motor temperatures", lvl,
                              f"ankles {t.get('ankle_c')} C, knees {t.get('knee_c')} C, hottest {t.get('max_c')} C "
                              f"({t.get('max_joint')})" + ("" if lvl == "ok" else
                              f"; rest the robot below {TEMP_WARN_C} C before GO (sit it or end the session)")))
        else:
            out.append(_check("temps", "Motor temperatures", "warn", "no reading yet (from the remote watcher)"))
    else:
        out.append(_check("temps", "Motor temperatures", "na", "the simulator reports none"))
    nb = ctx.get("near_body")
    walking = ns.get("phase") in ("PRESCAN", "RUN")
    if walking:
        pass  # near the body is normal mid-walk (a door frame); these checks are for the stand before GO
    elif nb is not None:
        if nb["close"]:
            out.append(_check("near", "Clear around the robot", "warn",
                              f"{nb['close']} obstacle cells within {nb['near_m']} m of the pelvis (outside the rear "
                              "blind): a person, the gantry or self returns. Step back, then Clear map"))
        else:
            out.append(_check("near", "Clear around the robot", "ok", f"nothing within {nb['near_m']} m outside the blind"
                              + (f" ({nb['self_like']} cells seen only from up close: self returns, masked by the guards)"
                                 if nb.get("self_like") else "")))
        if nb["behind_tall"]:
            out.append(_check("behind", "Something tall just behind", "warn",
                              f"{nb['behind_tall']} tall cells {nb['behind_m'][0]}-{nb['behind_m'][1]} m behind, just "
                              "outside the blind (a spotter, the rope, the gantry, furniture): the student may back up "
                              "into them. Spotter to the side, rope high"))
    else:
        out.append(_check("near", "Clear around the robot", "na", "open the map (no map from the planner yet)"))
    lag = ns.get("max_lag_s")
    if lag is not None:
        out.append(_check("lag", "Planner loop timing", "ok" if lag <= LAG_SUSPECT_S else "warn",
                          f"max loop lag {lag:.2f} s" + ("" if lag <= LAG_SUSPECT_S else
                                                        " (> 0.35 s: timing suspect; something else loads the PC?)")))
    lat = ns.get("latency_s")
    if lat is not None:
        out.append(_check("latency", "Plan latency", "ok" if lat <= LATENCY_GATE_S else "warn",
                          f"latest plan {lat:.2f} s" + ("" if lat <= LATENCY_GATE_S else
                                                       f" (> {LATENCY_GATE_S} s: a no-safe replan or a slow planner)")))
    return out


def ladder(ctx):
    """The robot-day steps, each done | now | todo | blocked, with what to do. Steps 1-5 come once per session; Clear
    map, goal, GO, the walk and its tag repeat for every attempt (a closed attempt is tagged before the next one)."""
    robot = ctx["mode"] == "robot"
    st = ctx["runner_state"]
    procs = ctx.get("procs") or {}
    att = ctx.get("attempts") or []
    last = att[-1] if att else None
    walking = last is not None and last.get("result") is None
    pending_tag = last is not None and not walking and last.get("tag") is None
    t, cleared, goal_t = ctx.get("t", 0.0), ctx.get("cleared_t"), ctx.get("goal_t")
    cycle = last["t_end"] if last is not None and not walking else ctx.get("control_t")
    scan_left = None if cleared is None else max(0.0, CLEAR_MAP_SCAN_S - (t - cleared))
    checks_ = {c["key"]: c["level"] for c in ctx.get("checks") or []}
    sys_bad = [k for k in ("rec", "lowstate", "remote") if checks_.get(k) == "bad"]
    if walking or pending_tag:
        map_done = goal_done = go_done = True
    else:
        map_done = cleared is not None and (cycle is None or cleared >= cycle) and scan_left == 0
        if "goal_ok" in ctx:  # the console's GO gate decides: a goal read back from the node in this cycle
            goal_done = bool(ctx["goal_ok"])
        else:  # a robot's launch goal (--goal) never counts: it was anchored at Arm, in a frame nobody checked
            goal_done = goal_t is not None and (cycle is None or goal_t >= cycle
                                                or (not robot and last is None and goal_t == 0.0))
        go_done = False
    steps = [
        ("systems", "Systems up", (not sys_bad and "rec" in procs) or not robot,
         "recorder, robot state, lidar" + (", remote" if robot else "") + (f": fix {', '.join(sys_bad)}" if sys_bad else "")),
        ("arm", "Arm: start the runner", "runner" in procs,
         "the gantry takes the weight; the runner releases Unitree's controller and ramps to the stand pose"),
        ("init", "Stand ramp done", st.get("init"), "the runner ramps the joints to the stand pose"),
        ("sonic", "Start SONIC", st.get("control"), "the robot stands on SONIC in the gantry; lower it to take its weight"),
        ("planner", "Student planner warm", ctx.get("node_ready"), "about 25 s for kcsc, longer for larger bundles"),
        ("map", "Clear map (nobody within 1 m)", map_done,
         "spotter to the side, rope held high; the node maps again and GO waits 10 s"
         + (f" ({scan_left:.0f} s of scan left)" if scan_left else "")),
        ("goal", "Goal set", goal_done,
         "ahead / left in metres from the robot now (or tap the map), or Send goal from Studio after its start check; "
         "a goal counts once the node has read it back, for this attempt only; leave ~0.5 m spare: leg odometry "
         "under-counts"),
        ("go", "GO", go_done or walking, "; ".join(ctx.get("go_locks") or [])
         or "the student walks to the goal; E-STOP or Select on any doubt"),
        ("walk", "Walk ends", pending_tag, f"attempt {last['n']} walking" if walking else
         "arrived, stopped, E-STOP or takeover"),
        ("tag", "Tag the attempt", False, "clean / touched / stumble / caught / aborted / setup issue, plus a note"),
    ]
    out, now_set = [], False
    for key, label, done, detail in steps:
        state = "done" if done else "todo"
        if not done and not now_set:
            state, now_set = "now", True
        out.append({"key": key, "label": label, "state": state, "detail": detail})
    if st.get("estop"):
        for s in out:
            if s["state"] in ("now", "todo"):
                s["state"] = "blocked"
        out.insert(0, {"key": "estop", "label": "E-STOP: the runner is damping", "state": "now",
                       "detail": "robot in the gantry; end the session, check the robot, start a new session"})
    return out
