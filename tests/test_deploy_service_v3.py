"""The deploy service's contract v3 (the 2026-10-28 review): session receipts, rehearsal sets with an identity and a
state (and the review's cases for picking the one in force), passage gates, the act lookup, session detail with labels
of ended sessions, a profile's route configuration, recording health, session-bound acts and GO's rehearsal gate.
Fake consoles, profiles and bundles in tmp; non-identity frames."""

import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from g1deploy import evidence, frames as F, hub, wire

from .test_deploy_service import (OFFSET, P_C, TOKEN, FakeConsole, coded_map, go_live, make_hub, make_scan,
                                  rehearsal_run, scene)

RUNS = ("exact", "drift", "offset")
GATE = {"a": [1.25, -0.6], "b": [1.25, 0.0]}  # the straight walk to [2.5, -0.3] crosses it at y = -0.15


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("G1_PANEL_TOKEN", TOKEN)
    monkeypatch.setattr(hub, "STUDIO_DATA", tmp_path / "no-studio")
    monkeypatch.setattr(hub, "gpu_free_mib", lambda: 12000.0)
    monkeypatch.setattr(hub.machine, "_procs", lambda skip=(): [])


def make_bundle(root, rules_route=None):
    """A bundle: MANIFEST.json, a student config (deployed format, or a rules block naming `rules_route`) and the route
    sources the hub reads its defaults from."""
    src = root / "flow/src/motion2scene_flow/closed_loop"
    src.mkdir(parents=True)
    (root / "config").mkdir()
    (root / "MANIFEST.json").write_text(json.dumps({"files": {}, "name": root.name}))
    (src / "route.py").write_text("class SensorRoute:\n    def __init__(self, a, b, m, *, inflate_m=0.35, step_over_m=0.25,"
                                  " period_s=0.5, grow_unknown_m=0.0, **_):\n        pass\n")
    (src / "route_cost.py").write_text("DEFAULTS = dict(\n    step_over_m=0.25,\n    hard_m=0.30,\n"
                                       "    hard_fallback_m=(0.20,),\n    w_clear=4.0,\n)\n")
    cfg = {"planner": {}, "episode": {"goal_lookahead_m": 1.5}}
    if rules_route is None:
        cfg["route_variants"] = {"sensor": {"route_source": "sensor", "route": {"period_s": 0.1}}}
    else:
        cfg["block"] = {"route_source": "sensor_cost", "route": rules_route}
    (root / "config/student.json").write_text(json.dumps(cfg))
    return root


@pytest.fixture
def profs(tmp_path, monkeypatch):
    """Profiles p1 (the deployed-format bundle) and p2 (a rules bundle, its route margin lowered by the profile)."""
    pdir = tmp_path / "profiles"
    pdir.mkdir()
    b1 = make_bundle(tmp_path / "bundle-a")
    b2 = make_bundle(tmp_path / "bundle-b", rules_route={"step_over_m": 0.05, "hard_fallback_m": [0.2, 0.15]})
    rules = tmp_path / "rules_plugin.py"
    rules.write_text("# a rules plugin\n")
    (pdir / "p1.json").write_text(json.dumps({"name": "p1", "bundle": str(b1), "guards": "deploy", "goal_lookahead": 1.0}))
    (pdir / "p2.json").write_text(json.dumps({"name": "p2", "bundle": str(b2), "rules": str(rules), "guards": "deploy",
                                              "rules_overrides": {"route": {"hard_m": 0.25}}}))
    monkeypatch.setattr(hub.profiles, "SEARCH", (pdir,))
    return pdir


def ident(h, scan="scan-a", goal=(2.5, -0.3), profile="p1", odom="leg", **over):
    pi = hub.profile_identity(profile)
    d = {"scan": scan, "scan_memory_sha": hub.file_sha(h.scans_dir / scan / "memory.npz"), "goal_scene": list(goal),
         "profile": profile, "profile_sha": pi["sha"], "bundle_sha": pi["bundle"]["manifest_sha"],
         "rules_sha": (pi["rules"] or {}).get("sha"), "odom": odom,
         "starts": {"exact": [0.0, 0.0, 0.0], "drift": [0.0, 0.0, 0.0], "offset": OFFSET}, "drift_model": "lab-g1-leg"}
    d.update(over)
    return d


def make_set(h, name, started, identity, results=("arrived",) * 3, kinds=RUNS, **kw):
    for i, kind in enumerate(kinds):
        rehearsal_run(h, f"rehearse-{name}-{kind}", kind, name, result=results[i], identity=identity,
                      offset=OFFSET if kind == "offset" else None, started=started + i, profile=identity["profile"],
                      goal=identity["goal_scene"], **kw)


def sets_by_name(h):
    return {s["name"]: s for s in h.rehearsal_sets(h.all_sessions())}


# ------------------------------------------------------------------------------------------------- frames: passage
def test_crosses_a_passage_gate():
    a, b = [1.0, -0.3], [1.0, 0.3]
    assert F.crosses([[0, 0], [2, 0]], a, b) and F.crosses([[2, 0.1], [0, 0.1]], a, b)
    assert not F.crosses([[0, 0], [0.9, 0]], a, b)  # stops short
    assert not F.crosses([[0, 0], [1, 1], [2, 0]], a, b)  # around, above the gate
    assert F.crosses([[0, 0], [1, 0], [2, 0]], a, b)  # a sample on the gate's line
    assert not F.crosses([[1, -0.3], [1, 0.3]], a, b)  # along the gate only
    assert not F.crosses([[0, 0]], a, b) and not F.crosses([[0, 0], [np.nan, 0]], a, b)


# ------------------------------------------------------------------------------------------------- A: receipts
def test_session_receipt_from_session_json_and_before_it(tmp_path, profs):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    d = h.day / "robot-100000-p2"
    go_live(h, "robot", d, 1, profile="p2", odom="fused", operator="lw")
    (d / "session.json").unlink()  # the console has not written it yet
    pi = hub.profile_identity("p2")
    (d / "hub_start.json").write_text(json.dumps({"mode": "robot", "profile": "p2", "t": 1700000000.0, "operator": "lw",
                                                  "confirm": {"gantry": True},
                                                  "kit": {"kit_commit": "k0", "kit_dirty": False,
                                                          "private_commit": "p9"}}))
    (d / "scene_scan.txt").write_text("scan-a\n")
    rc = h.receipt(d, h.live("robot"))
    assert rc["source"] == "hub" and rc["profile"] == {"name": "p2", "sha": pi["sha"]}  # the profile file, until it runs
    assert rc["rules"]["sha"] == pi["rules"]["sha"] and rc["private_commit"] == "p9" and rc["kit"]["commit"] == "k0"
    mc = {"rot": [1, 0, 0]}
    (d / "session.json").write_text(json.dumps({
        "mode": "robot", "robot": "lab-g1", "operator": "lw", "odom": "fused", "bundle": "/x/bundle-b",
        "deploy_rules": "/x/rules_plugin.py", "profile": {"name": "p2", "sha256": "f" * 64},
        "bundle_manifest_sha256": "b" * 64, "deploy_identity": {"rules_sha256": "r" * 64},
        "git": {"commit": "c1", "dirty": ["g1deploy/hub.py"]}, "runner_commit": "", "mount_calib": {"content": mc},
        "confirm": {"gantry": True, "estop": True}}))
    rc = h.receipt(d, h.live("robot"))
    assert rc["source"] == "session.json" and rc["kind"] == "robot" and rc["slot"] == "robot" and rc["robot"] == "lab-g1"
    assert rc["profile"]["sha"] == "f" * 64 and rc["bundle"] == {"name": "bundle-b", "manifest_sha": "b" * 64}
    assert rc["rules"] == {"name": "rules_plugin.py", "sha": "r" * 64} and rc["odom"] == "fused"
    assert rc["kit"] == {"commit": "c1", "dirty": True} and rc["runner_commit"] is None
    assert rc["mount_calib_sha"] == hub.json_sha(mc) and rc["confirmations"]["estop"] and rc["missing"] == []
    assert rc["scan"] == {"name": "scan-a", "memory_sha": hub.file_sha(h.scans_dir / "scan-a" / "memory.npz")}
    st = h.refresh()
    assert st["live"]["robot"]["receipt"]["session"] == d.name and "session_receipt" in st["capabilities"]
    for cap in ("rehearsal_identity", "rehearsal_match", "passage", "act_status", "session_detail", "session_tag",
                "profile_route"):
        assert cap in st["capabilities"]
    sha = hub.file_sha(h.scans_dir / "scan-a" / "memory.npz")
    assert st["scans"][0]["memory_sha"] == sha and h.scan_scene("scan-a")["memory_sha"] == sha
    assert h.receipt_v1(d.name)[1]["profile"]["sha"] == "f" * 64 and h.receipt_v1("nope")[0] == 404
    # a practice session's receipt names the robot odometry it practises, and its start offset
    p = h.day / "practice-110000-p1"
    p.mkdir()
    (p / "session.json").write_text(json.dumps({"mode": "sim", "odom": "lio", "profile": {"name": "p1", "sha256": "1"}}))
    (p / "practice.json").write_text(json.dumps({"scan": "scan-a", "start_offset": OFFSET, "odom": "zmq"}))
    rc = h.receipt(p)
    assert rc["kind"] == "practice" and rc["odom"] == "zmq" and rc["sim_odom"] == "lio" and rc["start_offset"] == OFFSET


# ------------------------------------------------------------------------------------------------- B: sets and match
def test_rehearse_set_records_identity_and_passage(tmp_path, profs, monkeypatch):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    launched = []
    monkeypatch.setattr(hub.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or SimpleNamespace(pid=7, poll=lambda: 0))
    monkeypatch.setattr(hub.threading, "Thread", lambda target, args, daemon: SimpleNamespace(start=lambda: target(*args)))
    assert "passage" in h.rehearse_set({"scan": "scan-a", "goal_scene": [2.5, -0.3], "profile": "p1",
                                        "passage": {"a": [1, 0], "b": [1, 0.05]}})["error"]  # 5 cm: not a gate
    r = h.rehearse_set({"scan": "scan-a", "goal_scene": [2.5, -0.3], "profile": "p1", "odom": "fused", "passage": GATE})
    assert r["ok"] and len(launched) == 3
    metas = [json.loads((d / "rehearsal.json").read_text()) for d in sorted(h.day.glob("rehearse*"))]
    idn = metas[0]["identity"]
    assert all(m["identity"] == idn and m["passage"]["a"] == GATE["a"] for m in metas)
    assert idn == ident(h, odom="fused") and idn["profile_sha"] == hub.file_sha(profs / "p1.json")
    rec = json.loads((h.day / "rehearsal_sets.jsonl").read_text().splitlines()[-1])
    assert rec["identity"] == idn and rec["finished"] >= rec["t"] and rec["cancelled"] is False
    # the fake runs never wrote session.json: the set has no run records, so it failed (never a verdict)
    st = sets_by_name(h)[r["name"]]
    assert st["state"] == "failed" and st["verdict"] is None and st["runs"]["exact"]["state"] == "error"


def test_set_states_verdicts_per_run_and_timing(tmp_path, profs):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    t0 = 1_800_000_000.0
    make_set(h, "set-a", t0, ident(h))
    make_set(h, "set-b", t0 + 100, ident(h), kinds=("exact", "drift"))  # stopped after two runs
    (h.day / "rehearsal_sets.jsonl").write_text(json.dumps({"name": "set-b", "error": "stopped", "cancelled": True,
                                                            "t": t0 + 100, "runs": {}}) + "\n")
    make_set(h, "set-c", t0 + 200, ident(h), lag=0.6)  # the PC was busy: loop lag 0.6 s
    make_set(h, "set-d", t0 + 300, ident(h), results=("arrived", "stopped", "arrived"))
    s = sets_by_name(h)
    a = s["set-a"]
    assert a["state"] == "complete" and a["verdict"] == "robust" and a["all_passed"] and a["timing_valid"] is True
    run = a["runs"]["drift"]
    assert run["state"] == "done" and run["arrival"] and run["passage"] is None and run["contact_max_n"] == 4.5
    assert run["final_dist_m"] == pytest.approx(0.0, abs=0.02) and run["timing_valid"] and run["passed"]
    assert "_path" not in run and run["ran"]["profile_sha"] == a["identity"]["profile_sha"]
    b = s["set-b"]
    assert b["state"] == "partial" and b["verdict"] is None and b["all_passed"] is None and b["cancelled"]
    assert b["runs"]["offset"]["state"] == "pending" and b["runs"]["exact"]["passed"]
    c = s["set-c"]
    assert c["state"] == "complete" and c["timing_valid"] is False and "suspect" in c["runs"]["exact"]["timing_note"]
    d = s["set-d"]
    assert d["verdict"] == "fails with drift" and d["all_passed"] is False and d["offset_ok"] is True
    # newest first by start time
    assert [x["name"] for x in h.rehearsal_sets(h.all_sessions())] == ["set-d", "set-c", "set-b", "set-a"]
    json.loads(wire.dumps(h.rehearsal_sets(h.all_sessions())))


def test_passage_scores_a_detour_as_a_failure(tmp_path, profs):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    n = 200
    s = np.linspace(0, 1, n)
    around = np.r_[np.c_[1.25 * s, 1.0 * s], np.c_[1.25 + 1.25 * s, 1.0 - 1.3 * s]]  # via (1.25, 1.0): above the gate
    for i, kind in enumerate(RUNS):
        rehearsal_run(h, f"rehearse-s1-{kind}", kind, "set-s1", identity=ident(h), passage=GATE, started=10.0 + i,
                      offset=OFFSET if kind == "offset" else None, walk=around if kind == "exact" else None, n=n)
    st = sets_by_name(h)["set-s1"]
    assert st["state"] == "complete" and st["passage"]["a"] == GATE["a"]
    ex = st["runs"]["exact"]
    assert ex["arrival"] and ex["passage"] == "missed" and not ex["passed"]  # arrived, but around the gap
    assert st["runs"]["drift"]["passage"] == "crossed" and st["runs"]["offset"]["passage"] == "crossed"
    assert st["verdict"] == "missed passage" and st["all_passed"] is False
    code, tr = h.track("rehearse-s1-exact")
    assert code == 200 and tr["passage"] == "missed" and tr["arrived"]


def test_match_the_reviews_cases(tmp_path, profs):
    """Newer failure over older pass; an in-progress rerun; changed odometry; changed profile contents under the same
    name; a different start; a set with an identity field missing."""
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    t0 = 1_800_000_000.0
    make_set(h, "set-1300-old-pass", t0, ident(h))
    m = h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="leg")
    assert m["complete"]["name"] == "set-1300-old-pass" and m["complete"]["verdict"] == "robust" and m["running"] is None
    assert h.rehearsal_match("scan-a", [2.55, -0.3], profile="p1", odom="leg")["complete"]  # within 0.08 m
    far = h.rehearsal_match("scan-a", [2.6, -0.3], profile="p1", odom="leg")
    assert far["complete"] is None and far["others"][0]["mismatch"] == ["goal_scene"]
    # newer failure over older pass: the newest complete set counts, whatever its name says
    make_set(h, "set-0900-new-failure", t0 + 3600, ident(h), results=("stopped", "stopped", "arrived"))
    m = h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="leg")
    assert m["complete"]["name"] == "set-0900-new-failure" and m["complete"]["verdict"] == "fails"
    old = next(o for o in m["others"] if o["name"] == "set-1300-old-pass")
    assert old["mismatch"] == [] and "older than set-0900-new-failure" in old["reasons"][0]
    # an in-progress rerun: shown as running, the newest complete one still counts
    run = h.day / "rehearse-set-rerun-exact"
    rehearsal_run(h, run.name, "exact", "set-rerun", identity=ident(h), started=t0 + 7200, ended=False)
    go_live(h, "sim", run, 1, rehearsal=True)
    (run / "session.json").write_text(json.dumps({"mode": "sim"}))
    m = h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="leg")
    assert m["running"]["name"] == "set-rerun" and m["running"]["state"] == "running" and m["running"]["verdict"] is None
    assert m["complete"]["name"] == "set-0900-new-failure"
    h.slots["sim"] = {"proc": None, "info": None}
    # changed odometry
    m = h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="fused")
    assert m["complete"] is None and all("odom" in o["mismatch"] for o in m["others"] if o["name"].startswith("set-1"))
    # changed profile contents under the same name: another sha
    (profs / "p1.json").write_text(json.dumps({"name": "p1", "bundle": json.loads((profs / "p1.json").read_text())["bundle"],
                                               "guards": "deploy", "goal_lookahead": 1.2}))
    m = h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="leg")
    assert m["complete"] is None and "profile_sha" in m["others"][0]["mismatch"]
    # a different start: a set whose offset run started elsewhere does not count for this start offset
    make_set(h, "set-other-start", t0 + 9000, ident(h, starts={"exact": [0, 0, 0], "drift": [0, 0, 0],
                                                                "offset": [0.3, 0.0, 0.0]}))
    m = h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="leg", offset=OFFSET)
    assert m["complete"] is None and "starts" in next(o for o in m["others"] if o["name"] == "set-other-start")["mismatch"]
    assert h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="leg", offset=[0.3, 0, 0])["complete"]["name"] == \
        "set-other-start"
    # a set with an identity field missing never counts, and says which
    bad = ident(h)
    del bad["odom"]
    make_set(h, "set-no-odom", t0 + 9500, bad)
    o = next(o for o in h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="leg")["others"]
             if o["name"] == "set-no-odom")
    assert "odom" in o["missing"] and "odom missing from the set's identity" in o["reasons"]
    # a set from before receipts (derived identity: its scan memory at the time is unknown)
    for i, kind in enumerate(RUNS):
        rehearsal_run(h, f"rehearse-legacy-{kind}", kind, "set-legacy", started=t0 + 9900 + i,
                      offset=OFFSET if kind == "offset" else None)
    o = next(o for o in h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="leg")["others"]
             if o["name"] == "set-legacy")
    assert "scan_memory_sha" in o["missing"]
    # a passage the goal needs: a set scored without it does not count
    p = h.rehearsal_match("scan-a", [2.5, -0.3], profile="p1", odom="leg", offset=[0.3, 0, 0], passage=GATE)
    assert p["complete"] is None and "passage" in next(x for x in p["others"] if x["name"] == "set-other-start")["mismatch"]
    json.loads(wire.dumps(p))


# ------------------------------------------------------------------------------------------------- F: route config
def test_profile_route_config(tmp_path, profs):
    h = make_hub(tmp_path)
    code, r1 = h.profile_route("p1")
    assert code == 200 and r1["basis"] == "bundle default" and r1["route"] == "sensor"
    p = r1["params"]
    assert p["route_margin_m"] == 0.35 and p["step_over_m"] == 0.25 and p["grow_unknown_m"] == 0.1
    assert p["goal_lookahead_m"] == 1.0 and p["guards"] == "deploy" and p["robot_radius_m"] == hub.GOAL_CLEAR_M
    assert p["ceiling_below_m"] == 1.35 and p["ceiling_near_m"] == 0.8  # the node's own stand deferral
    assert r1["sha"] == hub.file_sha(profs / "p1.json")
    code, r2 = h.profile_route("p2")
    q = r2["params"]
    assert r2["basis"] == "profile" and r2["route"] == "sensor_cost" and q["route_margin_m"] == 0.25
    assert q["step_over_m"] == 0.05 and q["route_fallback_m"] == [0.2, 0.15] and q["grow_unknown_m"] == 0.0
    assert q["goal_lookahead_m"] == 1.5 and r2["sources"]["grow_unknown_m"] == "not used by this route"
    assert r2["params_sha"] != r1["params_sha"] and h.profile_route("nope")[0] == 404
    assert "description" not in json.dumps(r2) and "rules_plugin" not in json.dumps(r2)


# ------------------------------------------------------------------------------------------------- D: act lookup
def test_act_lookup_through_the_console(tmp_path):
    h = make_hub(tmp_path)
    con = FakeConsole(act_results={"a-1": {"act": "g", "result": "done", "reason": "GO: attempt 1", "t_mono": 5.0}})
    old = FakeConsole(state={"runner": "CONTROL", "phase": "WAIT", "attempts": [], "acts": [
        {"id": "a-2", "act": "s", "result": "refused", "reason": "not armed", "t_mono": 3.0}]})
    try:
        d = h.day / "robot-120000-deploy"
        go_live(h, "robot", d, con.port)
        h.act("robot", {"act": "g", "id": "a-1", "operator": "lw", "rehearsal_override": {"operator": "lw"}})
        code, r = h.act_lookup("robot", "a-1")
        assert code == 200 and r["known"] and r["result"] == "done" and r["lookup"] == "console act_result"
        assert r["session"] == d.name and r["hub"]["relayed"] and r["hub"]["code"] == 200
        r = h.act_lookup("robot", "zz")[1]
        assert r["known"] is False and r["hub"] == {"relayed": False}
        # a console without act records: its last 20 acts, then its event log
        h.slots["robot"]["info"]["panel_port"] = old.port
        r = h.act_lookup("robot", "a-2")[1]
        assert r["known"] and r["result"] == "refused" and r["lookup"].startswith("console state.acts")
        (d / "console_events.jsonl").write_text(json.dumps({"id": "a-0", "act": "c", "result": "done", "reason": None,
                                                            "t_mono": 1.0}) + "\n")
        r = h.act_lookup("robot", "a-0")[1]
        assert r["known"] and r["lookup"] == "console_events.jsonl"
        assert h.act_lookup("nope", "x")[0] == 404
    finally:
        con.close()
        old.close()


# ------------------------------------------------------------------------------------------------- F01: session-bound acts
def test_acts_and_requests_bound_to_their_session(tmp_path, monkeypatch):
    h = make_hub(tmp_path)
    con = FakeConsole()
    try:
        go_live(h, "sim", h.day / "practice-130000-deploy", con.port, practice=True)
        code, r = h.act("sim", {"act": "c", "id": "x1", "expect": {"session": "robot-old"}})
        assert code == 409 and r["result"] == "refused" and "robot-old" in r["reason"] and not con.acts
        code, r = h.act("sim", {"act": "estop", "source": "studio", "expect": {"session": "robot-old"}})
        assert code == 200 and con.acts[-1]["act"] == "estop"  # a stop always goes through
        code, r = h.act("sim", {"act": "c", "id": "x2", "session": "practice-130000-deploy"})
        assert code == 200 and r["session"] == "practice-130000-deploy" and con.acts[-1]["id"] == "x2"
        r = h.send_goal_v1({"slot": "sim", "scan": "scan-a", "goal_scene": [1, 0], "check_id": "c",
                            "expect": {"session": "other"}})
        assert not r["ok"] and r["session_mismatch"]
        audit = [json.loads(x) for x in (h.day / "hub_actions.jsonl").read_text().splitlines()]
        assert audit[0]["code"] == 409 and audit[0]["reply"]["session_mismatch"]
    finally:
        con.close()


# ------------------------------------------------------------------------------------------------- GO's rehearsal gate
def test_robot_go_needs_a_complete_matching_set_or_an_override(tmp_path, profs):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    odom_from_scene = F.compose(P_C, F.inverse(OFFSET))
    con = FakeConsole(live_map=coded_map(F.apply(odom_from_scene, scene()), P_C[:2], robot=P_C))
    try:
        d = h.day / "robot-140000-p1"
        go_live(h, "robot", d, con.port, profile="p1", odom="leg")
        pi = hub.profile_identity("p1")
        (d / "session.json").write_text(json.dumps({"mode": "robot", "odom": "leg", "profile": {"name": "p1", "sha256": pi["sha"]},
                                                    "bundle_manifest_sha256": pi["bundle"]["manifest_sha"],
                                                    "deploy_identity": {"rules_sha256": None}}))
        code, r = h.act("robot", {"act": "g", "id": "g1", "operator": "lw"})
        assert code == 409 and r["rehearsal_required"] and "send_goal" in r["reason"] and not con.acts
        r = h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3], "check_id": "gc-1",
                            "passage": GATE})
        assert r["ok"] and r["rehearsal"]["match"] is False and r["passage"]["a"] == GATE["a"]
        # a goal check bound to another profile configuration, or expired, sends nothing
        n = len(con.acts)
        assert "another profile" in h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3],
                                                    "check_id": "gc-x", "config_sha": "0" * 64})["reason"]
        assert "expired" in h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3],
                                            "check_id": "gc-y", "check_expires_at": time.time() - 1})["reason"]
        assert len(con.acts) == n and h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3],
                                                      "check_id": "gc-1", "config_sha": pi["sha"]})["ok"]
        r = h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3], "check_id": "gc-1",
                            "passage": GATE})
        con.state = dict(con.state, goal_seq=r["goal_odom"]["seq"])
        code, r = h.act("robot", {"act": "g", "id": "g2", "operator": "lw"})
        assert code == 409 and "no complete rehearsal set" in r["reason"]
        # the override goes to the console with GO (the attempt row stores it)
        code, r = h.act("robot", {"act": "g", "id": "g3", "operator": "lw",
                                  "rehearsal_override": {"reason": "rehearsed on the bench"}})
        assert code == 200 and con.acts[-1]["act"] == "g"
        ov = con.acts[-1]["rehearsal"]["override"]
        assert ov["operator"] == "lw" and ov["reason"] == "rehearsed on the bench" and "no complete" in ov["without"]
        # a complete set scored on this passage, with this profile's contents and odometry: GO goes through
        make_set(h, "set-1", 100.0, ident(h), passage=GATE)
        code, r = h.act("robot", {"act": "g", "id": "g4", "operator": "lw"})
        assert code == 200 and con.acts[-1]["rehearsal"]["match"] and con.acts[-1]["rehearsal"]["name"] == "set-1"
        assert con.acts[-1]["rehearsal"]["verdict"] == "robust" and "override" not in con.acts[-1]["rehearsal"]
        # a goal set elsewhere since (the panel): not the goal the set was for
        con.state = dict(con.state, goal_seq=1)
        code, r = h.act("robot", {"act": "g", "id": "g5", "operator": "lw"})
        assert code == 409 and "set elsewhere" in r["reason"]
        # the send_goal record keeps the scan, the gate and the rehearsal in force
        rec = [json.loads(x) for x in (d / "start_check.jsonl").read_text().splitlines() if '"send_goal"' in x][-1]
        assert rec["scan"] == "scan-a" and rec["passage"]["b"] == GATE["b"] and rec["rehearsal"]["match"] is False
    finally:
        con.close()


def test_practice_go_records_the_rehearsal_and_never_refuses(tmp_path, profs):
    h = make_hub(tmp_path)
    con = FakeConsole()
    try:
        go_live(h, "sim", h.day / "practice-150000-p1", con.port, practice=True, profile="p1")
        code, r = h.act("sim", {"act": "g", "id": "g1", "operator": "lw"})
        assert code == 200 and con.acts[-1]["rehearsal"]["match"] is False and con.acts[-1]["rehearsal"]["reason"]
    finally:
        con.close()


# ------------------------------------------------------------------------------------------------- E: detail and labels
def practice_session(h, name="practice-160000-p1", gate=GATE, via=None):
    """An ended practice session: one attempt, its send_goal record (scan, goal, gate), sim truth through the gate."""
    d = h.day / name
    d.mkdir(parents=True)
    t0 = 1000.0
    (d / "session.json").write_text(json.dumps({"mode": "sim", "odom": "fused", "t0": t0,
                                                "profile": {"name": "p1", "sha256": "1" * 64}}))
    (d / "practice.json").write_text(json.dumps({"scan": "scan-a", "start_offset": OFFSET, "scene_from_course": OFFSET,
                                                 "odom": "fused"}))
    (d / "scene_scan.txt").write_text("scan-a\n")
    (d / "console_summary.json").write_text("{}")
    (d / "attempts.jsonl").write_text(json.dumps({"n": 1, "t_go": 10.0, "t_end": 20.0, "result": "arrived",
                                                  "check_id": "gc-9", "tag": "clean", "note": "",
                                                  "start_check": {"scene_from_odom": [0, 0, 0]}}) + "\n")
    (d / "start_check.jsonl").write_text(json.dumps({"action": "send_goal", "check_id": "gc-9", "ok": True,
                                                     "goal_scene": [2.5, -0.3], "scan": "scan-a", "passage": gate}) + "\n")
    (d / "console_events.jsonl").write_text(json.dumps({"wall": 50.0, "id": "t1", "act": "tag", "source": "panel",
                                                        "operator": "lw", "args": {"n": 1, "tag": "clean"},
                                                        "result": "done"}) + "\n")
    wall = t0 + np.arange(0, 30, 0.01)
    scene_xy = np.c_[np.interp(wall, [t0 + 10, t0 + 20], [0, 2.5]), np.interp(wall, [t0 + 10, t0 + 20], [0, -0.3])]
    course = F.apply(F.inverse(OFFSET), scene_xy) if via is None else via(wall)
    np.savez_compressed(d / "sim.npz", wall=wall, root_pos=np.c_[course, 0.78 + 0 * wall], contact_n=0 * wall)
    return d


def test_session_detail_and_labels_of_an_ended_session(tmp_path, profs, monkeypatch):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    d = practice_session(h)
    code, det = h.session_detail(d.name)
    assert code == 200 and det["ended"] and det["receipt"]["kind"] == "practice" and det["live_slot"] is None
    a = det["attempts"][0]
    assert a["goal_scene"] == [2.5, -0.3] and a["scan"] == "scan-a" and a["passage_gate"]["a"] == GATE["a"]
    assert a["passage"] == "crossed" and a["passage_basis"] == "sim truth"
    assert [x["via"] for x in a["tag_history"]] == ["console"] and a["tag"] == "clean"
    assert det["replay"] == {"studio_id": hub.studio_id(h.day, d), "linked": False}
    assert det["manifest"]["version"] in (1, 2) and "present" in det["manifest"]
    # Review lists every evidence item from the session detail alone (a fresh browser has nothing else; 2026-10-28)
    assert isinstance(det["manifest"]["items"], list) and det["manifest"]["items"] and "summary" in det["manifest"]
    # labels after the end: append-only, the newest wins, every change kept
    code, r = h.tag_v1(d.name, {"n": 1, "tag": "touched", "note": "brushed the left tower", "operator": "lw"})
    assert code == 200 and r["via"] == "service" and r["attempt"]["tag"] == "touched"
    time.sleep(0.01)
    code, r = h.tag_v1(d.name, {"n": 1, "tag": "stumble", "operator": "jw"})
    a = r["attempt"]
    assert a["tag"] == "stumble" and a["note"] == "brushed the left tower"  # a tag alone keeps the note
    assert [(x["via"], x["tag"]) for x in a["tag_history"]] == [("console", "clean"), ("service", "touched"),
                                                                ("service", "stumble")]
    assert len((d / hub.TAGS_FILE).read_text().splitlines()) == 2
    assert h._summary(d)["attempts"][0]["tag"] == "stumble"  # the session list reads the merged labels too
    for bad in ({"n": 1, "tag": "great", "operator": "lw"}, {"n": 1, "tag": "clean"}, {"n": 0, "tag": "clean", "operator": "x"},
                {"n": 1, "operator": "lw"}):
        assert h.tag_v1(d.name, bad)[0] == 400
    assert h.tag_v1(d.name, {"n": 2, "tag": "clean", "operator": "lw"})[0] == 404
    assert h.tag_v1("nope", {"n": 1, "tag": "clean", "operator": "lw"})[0] == 404
    # a console that still runs for this session elsewhere takes its own labels
    monkeypatch.setattr(hub.machine, "_procs", lambda skip=(): [(9, f"python g1deploy/console.py sim --record {d}")])
    assert h.tag_v1(d.name, {"n": 1, "tag": "clean", "operator": "lw"})[0] == 409
    # a detour around the gate is "missed"
    d2 = practice_session(h, "practice-170000-p1",
                          via=lambda w: F.apply(F.inverse(OFFSET), np.c_[np.interp(w, [1010, 1015, 1020], [0, 1.25, 2.5]),
                                                                         np.interp(w, [1010, 1015, 1020], [0, 1.0, -0.3])]))
    assert h.session_detail(d2.name)[1]["attempts"][0]["passage"] == "missed"


def test_labels_of_a_live_session_go_through_its_console(tmp_path):
    h = make_hub(tmp_path)
    con = FakeConsole()
    try:
        d = h.day / "robot-180000-deploy"
        go_live(h, "robot", d, con.port)
        (d / "attempts.jsonl").write_text(json.dumps({"n": 1, "result": "arrived"}) + "\n")
        code, r = h.tag_v1(d.name, {"n": 1, "tag": "clean", "note": "ok", "operator": "lw"})
        assert code == 200 and r["via"] == "console" and not (d / hub.TAGS_FILE).exists()
        act = con.acts[-1]
        assert act["act"] == "tag" and act["n"] == 1 and act["tag"] == "clean" and act["expect"] == {"session": d.name}
        assert h.session_detail(d.name)[1]["ended"] is False
    finally:
        con.close()


def test_robot_attempt_passage_on_odometry(tmp_path):
    h = make_hub(tmp_path)
    d = h.day / "robot-190000-deploy"
    (d / "rec").mkdir(parents=True)
    s_from_o = [0.5, 0.2, 0.3]
    (d / "session.json").write_text(json.dumps({"mode": "robot", "t0": 0.0}))
    (d / "attempts.jsonl").write_text(json.dumps({"n": 1, "t_go": 1.0, "t_end": 5.0, "check_id": "c1",
                                                  "start_check": {"scene_from_odom": s_from_o}}) + "\n")
    (d / "start_check.jsonl").write_text(json.dumps({"action": "send_goal", "check_id": "c1", "ok": True,
                                                     "goal_scene": [2.5, -0.3], "passage": GATE}) + "\n")
    rows = [{"t": 1.0 + 0.5 * i, "status": {"xy": F.apply(F.inverse(s_from_o), [0.3 * i, -0.04 * i]).tolist()}}
            for i in range(9)]
    (d / "rec/node_status.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    a = h.session_detail(d.name)[1]["attempts"][0]
    assert a["passage"] == "crossed" and a["passage_basis"] == "odometry estimate"


# ------------------------------------------------------------------------------------------------- G: recording
def test_recording_health_when_the_kit_has_it(tmp_path, monkeypatch):
    h = make_hub(tmp_path)
    d = h.day / "robot-200000-deploy"
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps({"mode": "robot"}))
    monkeypatch.delattr(evidence, "recording_health", raising=False)
    assert h.recording_v1(d.name)[1]["supported"] is False and "recording_health" not in hub.evidence_capabilities()
    monkeypatch.setattr(evidence, "recording_health", lambda p: {"streams": [], "finalized": True, "dir": str(p)},
                        raising=False)
    code, r = h.recording_v1(d.name)
    assert code == 200 and r["supported"] and r["finalized"] and r["session"] == d.name
    assert "recording_health" in hub.evidence_capabilities()
    monkeypatch.setattr(evidence, "manifest", lambda p: {"version": 2, "items": [], "complete": True})
    assert "manifest_v2" in hub.evidence_capabilities()
    assert h.session_detail(d.name)[1]["recording"]["finalized"] is True
    monkeypatch.setattr(evidence, "recording_health", lambda p: 1 / 0, raising=False)
    assert "ZeroDivisionError" in h.recording_v1(d.name)[1]["error"] and h.recording_v1("nope")[0] == 404


# ------------------------------------------------------------------------------------------------- HTTP
def test_http_v3_routes(tmp_path, profs):
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    make_set(h, "set-1", 100.0, ident(h))
    d = practice_session(h)
    server = hub.serve(h, 0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def get(path):
        try:
            with urllib.request.urlopen(f"{base}{path}{'&' if '?' in path else '?'}t={TOKEN}") as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(path, body):
        req = urllib.request.Request(f"{base}{path}?t={TOKEN}", data=json.dumps(body).encode(), method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        code, m = get("/api/v1/rehearsal/match?scan=scan-a&x=2.5&y=-0.3&profile=p1&odom=leg")
        assert code == 200 and m["complete"]["name"] == "set-1" and m["order"].startswith("newest first")
        assert get("/api/v1/rehearsal/match?scan=scan-a&x=a&y=1")[0] == 400
        code, m = get("/api/v1/rehearsal/match?scan=scan-a&x=2.5&y=-0.3&profile=p1&odom=leg&pa=1.25,-0.6&pb=1.25,0")
        assert code == 200 and m["complete"] is None
        code, det = get(f"/api/v1/session/{d.name}")
        assert code == 200 and det["attempts"][0]["passage"] == "crossed"
        assert get(f"/api/v1/session/{d.name}/receipt")[1]["kind"] == "practice"
        assert get(f"/api/v1/session/{d.name}/recording")[0] == 200
        assert get("/api/v1/session/nope")[0] == 404
        code, rc = get("/api/v1/profile/p1/route")
        assert code == 200 and rc["params"]["route_margin_m"] == 0.35 and get("/api/v1/profile/zz/route")[0] == 404
        code, a = get("/api/v1/act/robot/some-id")
        assert code == 200 and a["known"] is False
        code, r = post(f"/api/v1/session/{d.name}/tag", {"n": 1, "tag": "aborted", "operator": "lw"})
        assert code == 200 and r["attempt"]["tag"] == "aborted"
        assert post(f"/api/v1/session/{d.name}/tag", {"n": 1, "tag": "x", "operator": "lw"})[0] == 400
        assert post("/api/v1/stop", {"slot": "sim", "session": "other"})[1]["session_mismatch"]
        rd = get("/api/v1/readiness?scan=scan-a&x=2.5&y=-0.3")[1]
        assert "checks" in rd, rd
        reh = next(c for c in rd["checks"] if c["id"] == "scene.rehearsal")
        assert reh["status"] == "block" and "no complete rehearsal set" in reh["detail"]  # set-1 was for p1, not deploy
    finally:
        server.shutdown()


# ------------------------------------------------------------------------------------------------- the verifier's defects
def live_session(h, con, name="robot-210000-p1", slot="robot", odom="fused", **info):
    """A live session (robot, or practice in the sim slot) of profile p1 whose session.json names what runs."""
    d = h.day / name
    go_live(h, slot, d, con.port, profile="p1", odom=odom, **info)
    pi = hub.profile_identity("p1")
    (d / "session.json").write_text(json.dumps({"mode": "robot" if slot == "robot" else "sim", "odom": odom,
                                                "profile": {"name": "p1", "sha256": pi["sha"]},
                                                "bundle_manifest_sha256": pi["bundle"]["manifest_sha"],
                                                "deploy_identity": {"rules_sha256": None}}))
    if info.get("practice"):
        (d / "practice.json").write_text(json.dumps({"scan": "scan-a", "start_offset": OFFSET, "odom": odom}))
    return d


def test_robot_go_needs_a_robust_set_whatever_the_odometry(tmp_path, profs):
    """D3: a complete matching set that is not robust (here: fails with drift, on FAST-LIO + legs) needs the override;
    practice records it and never refuses; readiness agrees."""
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    odom_from_scene = F.compose(P_C, F.inverse(OFFSET))
    con = FakeConsole(live_map=coded_map(F.apply(odom_from_scene, scene()), P_C[:2], robot=P_C))
    try:
        live_session(h, con, odom="fused")
        make_set(h, "set-1", 100.0, ident(h, odom="fused"), results=("arrived", "stopped", "arrived"))
        r = h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3], "check_id": "gc-1"})
        assert r["ok"] and r["rehearsal"]["match"] and r["rehearsal"]["verdict"] == "fails with drift"
        con.state = dict(con.state, goal_seq=r["goal_odom"]["seq"])
        code, g = h.act("robot", {"act": "g", "id": "g1", "operator": "lw"})
        assert code == 409 and g["rehearsal_required"] and "not robust: fails with drift" in g["reason"]
        assert not any(a["act"] == "g" for a in con.acts)
        rd = {c["id"]: c for c in h.readiness("scan-a", "robot", [2.5, -0.3])["checks"]}["scene.rehearsal"]
        assert rd["status"] == "block" and "override" in rd["detail"]
        assert rd["overridable"] == "rehearsal_override"  # GO's tick clears it, not a hard lock (verifier N1)
        # without a goal from the page, readiness judges the goal in force (the one sent in this session)
        nogoal = {c["id"]: c for c in h.readiness("scan-a", "robot")["checks"]}["scene.rehearsal"]
        assert nogoal["status"] == "block" and "set-1" in nogoal["detail"]
        code, g = h.act("robot", {"act": "g", "id": "g2", "operator": "lw",
                                  "rehearsal_override": {"reason": "drift is leg odometry's; this walk is FAST-LIO"}})
        ov = con.acts[-1]["rehearsal"]["override"]
        assert code == 200 and ov["operator"] == "lw" and "not robust" in ov["without"]
        assert con.acts[-1]["rehearsal"]["name"] == "set-1"
        # a robust set for the goal: no override needed
        make_set(h, "set-2", 200.0, ident(h, odom="fused"))
        code, g = h.act("robot", {"act": "g", "id": "g3", "operator": "lw"})
        assert code == 200 and con.acts[-1]["rehearsal"]["verdict"] == "robust"
        h.refresh()  # readiness reads the 1 Hz state
        rd = {c["id"]: c for c in h.readiness("scan-a", "robot", [2.5, -0.3])["checks"]}["scene.rehearsal"]
        assert rd["status"] == "pass"
    finally:
        con.close()
    # practice: a failing set is recorded with GO, never a refusal; readiness warns
    h2 = make_hub(tmp_path / "p")
    make_scan(h2, "scan-a")
    con = FakeConsole(live_map=coded_map(F.apply(odom_from_scene, scene()), P_C[:2], robot=P_C))
    try:
        live_session(h2, con, name="practice-220000-p1", slot="sim", practice=True)
        make_set(h2, "set-1", 100.0, ident(h2, odom="fused"), results=("stopped", "stopped", "stopped"))
        r = h2.send_goal_v1({"slot": "sim", "scan": "scan-a", "goal_scene": [2.5, -0.3], "check_id": "gc-1"})
        con.state = dict(con.state, goal_seq=r["goal_odom"]["seq"])
        code, g = h2.act("sim", {"act": "g", "id": "g1", "operator": "lw"})
        assert code == 200 and con.acts[-1]["rehearsal"]["verdict"] == "fails" and "override" not in con.acts[-1]["rehearsal"]
        rd = {c["id"]: c for c in h2.readiness("scan-a", "sim", [2.5, -0.3])["checks"]}["scene.rehearsal"]
        assert rd["status"] == "warn" and "practice" in rd["detail"]
    finally:
        con.close()


def test_start_check_envelope_is_the_goals_set_in_force(tmp_path, profs):
    """D4: within_offset_envelope comes from the set in force for this goal (scan, goal, profile, odometry), only when
    its offset case passed; null otherwise, whatever other sets on the scan did."""
    h = make_hub(tmp_path)
    make_scan(h, "scan-a")
    odom_from_scene = F.compose(P_C, F.inverse(OFFSET))
    con = FakeConsole(live_map=coded_map(F.apply(odom_from_scene, scene()), P_C[:2], robot=P_C))
    try:
        live_session(h, con, odom="fused")
        # another goal on the same scan whose offset case passed: not this goal's
        make_set(h, "set-other-goal", 100.0, ident(h, goal=(2.0, 1.0), odom="fused"))
        c = h.start_check_v1("scan-a", "robot", [2.5, -0.3])
        assert c["ok"] and c["offset_envelope"] is None and c["within_offset_envelope"] is None
        # this goal's set, its offset case failed: no envelope
        make_set(h, "set-goal-offset-failed", 200.0, ident(h, odom="fused"), results=("arrived", "arrived", "stopped"))
        c = h.start_check_v1("scan-a", "robot", [2.5, -0.3])
        assert c["offset_envelope"] is None and c["within_offset_envelope"] is None
        # a newer set for this goal whose offset case passed: inside it (the robot stands 20 deg / 0.2 m off)
        make_set(h, "set-goal-ok", 300.0, ident(h, odom="fused"))
        c = h.start_check_v1("scan-a", "robot", [2.5, -0.3])
        assert c["within_offset_envelope"] is True and c["offset_envelope"]["set"] == "set-goal-ok"
        assert h.start_check_v1("scan-a", "robot")["within_offset_envelope"] is None  # no goal yet
        # the goal sent through send_goal is the start check's goal from then on
        r = h.send_goal_v1({"slot": "robot", "scan": "scan-a", "goal_scene": [2.5, -0.3], "check_id": "gc-1"})
        assert r["check"]["within_offset_envelope"] is True
        assert h.start_check_v1("scan-a", "robot")["offset_envelope"]["set"] == "set-goal-ok"
        far = F.compose(P_C, F.inverse(F.offset_pose(-28.0, 0.4)))
        con.map = coded_map(F.apply(far, scene()), P_C[:2], robot=P_C)
        assert h.start_check_v1("scan-a", "robot")["within_offset_envelope"] is False
        # other odometry than the set's: not the set in force
        h.slots["robot"]["info"]["odom"] = "leg"
        sj = json.loads((h.day / "robot-210000-p1" / "session.json").read_text())
        (h.day / "robot-210000-p1" / "session.json").write_text(json.dumps(dict(sj, odom="leg")))
        assert h.start_check_v1("scan-a", "robot", [2.5, -0.3])["offset_envelope"] is None
    finally:
        con.close()


def test_session_names_resolve_across_lab_days(tmp_path, profs):
    """D1: a Review link from an earlier day opens its session; the newest day that has the name wins; an unknown name
    is a clear 404."""
    h = make_hub(tmp_path)
    runs = h.day.parent
    old = practice_session(SimpleNamespace(day=runs / "lab-day-20261027"), name="practice-090000-p1")
    older = practice_session(SimpleNamespace(day=runs / "lab-day-20261026"), name="practice-090000-p1")
    import os
    os.utime(older.parent, (1e9, 1e9))
    assert h.session_dir("practice-090000-p1") == old.resolve()
    code, det = h.session_detail("practice-090000-p1")
    assert code == 200 and det["day"] == "lab-day-20261027"
    assert det["replay"]["studio_id"] == hub.studio_id(old.parent, old) and "lab-day-20261027" in det["replay"]["studio_id"]
    assert det["attempts"][0]["passage"] == "crossed" and h.receipt_v1("practice-090000-p1")[0] == 200
    code, r = h.tag_v1("practice-090000-p1", {"n": 1, "tag": "touched", "operator": "lw"})
    assert code == 200 and (old / hub.TAGS_FILE).exists() and not (older / hub.TAGS_FILE).exists()
    # today's session of the same name wins over the earlier days'
    today = practice_session(h, name="practice-090000-p1")
    assert h.session_dir("practice-090000-p1") == today.resolve()
    for bad in ("nope", "../lab-day-20261027/practice-090000-p1", "scans", "", ".hidden"):
        code, r = h.session_detail(bad)
        assert code == 404 and r["ok"] is False
    assert "lab-day-*" in h.session_detail("nope")[1]["error"]
    assert h.manifest_v1("nope")[0] == 404 and h.recording_v1("nope")[0] == 404 and h.tag_v1("nope", {})[0] == 404


def test_an_attempt_from_past_the_gate_has_no_passage_to_make(tmp_path, profs):
    """D11: the same goal sent again from where the first walk ended: attempt 2 starts past the gate (n/a), not
    "missed"; the first one went through it."""
    h = make_hub(tmp_path)
    d = practice_session(h)
    rows = [json.loads(x) for x in (d / "attempts.jsonl").read_text().splitlines()]
    rows.append(dict(rows[0], n=2, t_go=22.0, t_end=28.0, check_id="gc-10", tag=None))
    (d / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with open(d / "start_check.jsonl", "a") as f:
        f.write(json.dumps({"action": "send_goal", "check_id": "gc-10", "ok": True, "goal_scene": [2.5, -0.3],
                            "scan": "scan-a", "passage": GATE}) + "\n")
    a1, a2 = h.session_detail(d.name)[1]["attempts"]
    assert a1["passage"] == "crossed" and a1["passage_note"] is None
    assert a2["passage"] == "n/a" and a2["passage_note"] == "started past the gate" and a2["passage_basis"] == "sim truth"
    # the console's own row may carry the gate (its `passage`) without a send_goal record
    rows[1] = dict(rows[1], check_id="gc-none", passage=GATE, goal_scene=[2.5, -0.3])
    (d / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    a2 = h.session_detail(d.name)[1]["attempts"][1]
    assert a2["passage_gate"]["a"] == GATE["a"] and a2["passage"] == "n/a"
    assert F.passage_ahead([0, 0], [2.5, -0.3], GATE["a"], GATE["b"])
    assert not F.passage_ahead([2.4, -0.3], [2.5, -0.3], GATE["a"], GATE["b"])
    assert not F.passage_ahead([1.25, -0.3], [2.5, -0.3], GATE["a"], GATE["b"])  # on the gate's line


def test_lowcmd_stays_required_when_the_console_ran_the_runner(tmp_path):
    """D12: runner.log is optional; the console's own record (a done Arm, or a console started without --arm) says
    the runner ran, and then lowcmd is required."""
    def session(name, argv, acts, runner_log=False):
        d = tmp_path / name
        (d / "rec" / "lowstate").mkdir(parents=True)
        np.savez(d / "rec" / "lowstate" / "000000.npz", t=np.linspace(0.0, 2.0, 5))
        (d / "rec" / "meta.json").write_text(json.dumps({"format": "g1deploy.recording.v1"}))
        (d / "session.json").write_text(json.dumps({"mode": "sim", "odom": "dds", "argv": argv}))
        (d / "console_events.jsonl").write_text("".join(json.dumps(a) + "\n" for a in acts))
        if runner_log:
            (d / "runner.log").write_text("x")
        return {i["id"]: i for i in evidence.manifest(d)["items"]}
    arm = [{"id": "1", "act": "arm", "result": "done"}]
    armed = session("practice-1", ["sim", "--arm"], arm)
    assert evidence.runner_ran(tmp_path / "practice-1") and armed["lowcmd"]["status"] == "missing"
    assert armed["lowcmd"]["required"] and armed["runner_log"]["status"] == "missing"
    launched = session("practice-2", ["sim", "--script", "key:s"], [])  # no --arm: the runner starts at launch
    assert launched["lowcmd"]["status"] == "missing" and launched["lowcmd"]["required"]
    never = session("practice-3", ["sim", "--arm"], [{"id": "1", "act": "arm", "result": "refused"}])
    assert never["lowcmd"]["status"] == "n/a" and not evidence.runner_ran(tmp_path / "practice-3")
    assert evidence.runner_ran(tmp_path / "practice-4") is False
    assert session("practice-5", ["sim", "--arm"], [], runner_log=True)["lowcmd"]["status"] == "missing"
