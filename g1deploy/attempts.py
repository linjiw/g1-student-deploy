"""The attempt log of a console session (attempts.jsonl): one row per GO, for the operator panel and the report.

An attempt opens when GO reaches the node and closes by itself when the walk ends: the node reports ARRIVED or
STOPPED, the runner goes to E-STOP, the operator takes over or asks for a stand, or the session ends. While it is
open it keeps the walk's extremes from the node's status (tilt, pelvis height, plan latency, node lag) and its plan
and no-safe counts. The operator labels it afterwards from the panel: an outcome tag and a free note.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

TAGS = ("clean", "touched", "stumble", "caught", "aborted", "setup issue")
TAG_HELP = {"clean": "walked as meant, nothing touched", "touched": "brushed or hit something",
            "stumble": "lost balance or stumbled, recovered", "caught": "the gantry, rope or spotter held the robot",
            "aborted": "the operator stopped it (wrong plan, unsafe)", "setup issue": "not the student: setup, map, rope"}


class Attempts:
    def __init__(self, path: Path | None, profile: str | None = None):
        self.path, self.profile = path, profile
        self.rows: list[dict] = []
        self.open: dict | None = None

    # -- life cycle --------------------------------------------------------------------------------------
    def start(self, t: float, status: dict, goal_rel=None):
        """GO reached the node. A still-open attempt ends here (a new GO after a stand without an end)."""
        if self.open is not None:
            self.close(t, "superseded", status)
        xy = [round(float(v), 3) for v in status["xy"]] if status.get("xy") is not None else None
        self.open = {"n": len(self.rows) + 1, "profile": self.profile, "t_go": round(t, 2), "t_end": None,
                     "result": None, "goal_rel": goal_rel, "goal": status.get("goal"), "start_xy": xy,
                     "start_phase": status.get("phase"), "left_start": False, "max_tilt": None, "min_pelvis_z": None,
                     "max_latency_s": None, "max_lag_s": None, "plans": 0, "no_safe_plans": 0, "end_goal_dist_m": None,
                     "walked_m": None, "tag": None, "note": ""}
        self.rows.append(self.open)
        self._plans_seen = status.get("plans") or 0
        self.save()

    def update(self, t: float, status: dict, estop: bool = False):
        """Each console tick: track the walk, close it when it has ended."""
        a = self.open
        if a is None:
            return
        ph = status.get("phase")
        if ph != a["start_phase"] or ph in ("PRESCAN", "RUN"):
            a["left_start"] = True

        def ext(key, val, fn):
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                a[key] = round(float(val) if a[key] is None else fn(a[key], float(val)), 3)
        if ph == "RUN":
            ext("max_tilt", status.get("tilt_rad"), max)
            ext("min_pelvis_z", status.get("pelvis_z_m"), min)
            ext("max_latency_s", status.get("latency_s"), max)
        ext("max_lag_s", status.get("max_lag_s"), max)
        plans = status.get("plans") or 0
        if plans > self._plans_seen:
            a["plans"] += plans - self._plans_seen
            if status.get("no_safe"):
                a["no_safe_plans"] += 1  # the latest plan only: a lower bound when several land between ticks
            self._plans_seen = plans
        if estop:
            self.close(t, "E-STOP", status)
        elif a["left_start"] and ph in ("ARRIVED", "STOPPED"):
            self.close(t, ph.lower(), status)

    def close(self, t: float, result: str, status: dict | None = None):
        a = self.open
        if a is None:
            return
        status = status or {}
        a.update(t_end=round(t, 2), result=result, duration_s=round(t - a["t_go"], 1))
        if status.get("goal_dist_m") is not None:
            a["end_goal_dist_m"] = round(float(status["goal_dist_m"]), 2)
        if status.get("xy") is not None and a["start_xy"] is not None:
            a["walked_m"] = round(math.dist(status["xy"], a["start_xy"]), 2)
        self.open = None
        self.save()

    def tag(self, n: int, tag: str | None = None, note: str | None = None) -> bool:
        """The operator's label of attempt `n` (1-based); an empty tag clears it."""
        if not 1 <= n <= len(self.rows) or (tag and tag not in TAGS):
            return False
        row = self.rows[n - 1]
        if tag is not None:
            row["tag"] = tag or None
        if note is not None:
            row["note"] = str(note)[:500]
        self.save()
        return True

    # -- output ----------------------------------------------------------------------------------------
    def save(self):
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in self.public()))
        tmp.replace(self.path)

    def public(self) -> list[dict]:
        return [{k: v for k, v in r.items() if k not in ("start_phase", "left_start")} for r in self.rows]


def load(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except OSError:
        return []
