"""Deploy profiles (g1deploy/profiles.py), the busy-PC guard (g1deploy/machine.py) and their console wiring."""

import json
from types import SimpleNamespace

import pytest

from g1deploy import machine, profiles
from g1deploy.console import parse_args
from tools import session_report


def test_shipped_profiles_load_and_resolve_paths():
    for name in ("deploy", "deploy-full-lidar"):
        prof, path = profiles.load(name)
        assert path.parent.name == "profiles"
        assert prof["bundle"].startswith("/") and prof["guards"] == "deploy" and prof["goal_lookahead"] == 1.0
    assert profiles.load("deploy")[0]["rear_blind_deg"] == 180.0
    assert profiles.load("deploy-full-lidar")[0]["rear_blind_deg"] == 0.0


def test_profile_rejects_unknown_keys_and_bad_values(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"guards": "deploy", "lookahead": 1.0}))
    with pytest.raises(profiles.ProfileError, match="unknown keys"):
        profiles.load(bad)
    bad.write_text(json.dumps({"guards": "strict"}))
    with pytest.raises(profiles.ProfileError, match="must be one of"):
        profiles.load(bad)
    bad.write_text(json.dumps({"candidates": "16"}))
    with pytest.raises(profiles.ProfileError, match="must be int"):
        profiles.load(bad)
    with pytest.raises(profiles.ProfileError, match="no deploy profile"):
        profiles.find("no-such-profile")


def test_sim_profile_fills_options_and_command_line_wins():
    a = parse_args(["sim", "--profile", "deploy", "--layout", "open-2m"])
    assert (a.guards, a.goal_lookahead, a.rear_blind_deg, a.rear_blind_range_m, a.debug_plans) == ("deploy", 1.0, 180.0, 1.0, True)
    assert (a.planner_impl, a.planner_precision) == ("fused", "fp32")  # console defaults, not set by the profile
    assert a.profile["name"] == "deploy" and a.profile["overridden"] == []
    b = parse_args(["sim", "--profile", "deploy", "--layout", "open-2m", "--rear-blind-deg", "0", "--guards", "walls"])
    assert (b.rear_blind_deg, b.guards) == (0.0, "walls") and sorted(b.profile["overridden"]) == ["guards", "rear_blind_deg"]


def test_sim_without_profile_keeps_the_old_defaults():
    a = parse_args(["sim", "--layout", "open-2m"])
    assert (a.guards, a.debug_plans, a.planner_impl, a.planner_precision, a.profile) == ("off", False, "fused", "fp32", None)
    assert a.rear_blind_deg is None and a.goal_lookahead is None


def test_deploy_profile_wins_over_the_robot_profile():
    a = parse_args(["robot", "--robot", "lab-g1", "--profile", "deploy-full-lidar", "--goal", "1", "0"])
    assert a.iface == "eno1" and a.lidar == "livox"  # the robot's own facts
    assert a.rear_blind_deg == 0.0  # the deploy profile's blind, not lab-g1's 180
    b = parse_args(["robot", "--robot", "lab-g1", "--goal", "1", "0"])
    assert (b.rear_blind_deg, b.rear_blind_range_m) == (180, 1.0)  # no profile: the robot profile's blind as before


def test_identity_follows_the_effective_settings():
    base = parse_args(["sim", "--profile", "deploy", "--layout", "open-2m"])
    same = parse_args(["sim", "--layout", "open-2m", "--guards", "deploy", "--goal-lookahead", "1.0",
                       "--rear-blind-deg", "180", "--rear-blind-range-m", "1.0"])
    assert profiles.identity(base)["sha256"] == profiles.identity(same)["sha256"]  # flags or profile: same run
    other = parse_args(["sim", "--profile", "deploy", "--layout", "open-2m", "--goal-lookahead", "1.5"])
    assert profiles.identity(other)["sha256"] != profiles.identity(base)["sha256"]
    off1 = SimpleNamespace(rear_blind_deg=0.0, rear_blind_range_m=1.0)
    off2 = SimpleNamespace(rear_blind_deg=None, rear_blind_range_m=None)
    assert profiles.identity(off1)["sha256"] == profiles.identity(off2)["sha256"]  # no blind is no blind


def test_busy_reasons():
    quiet = {"load": [3.0, 3.0, 3.0], "cpus": 28, "gpu": {"util_pct": 35}, "chains": [], "heavy": []}
    assert machine.busy_reasons(quiet) == []
    loaded = dict(quiet, load=[17.0, 15.0, 10.0])
    assert "load 17.0 on 28 cores" in machine.busy_reasons(loaded)[0]
    gpu = dict(quiet, gpu={"util_pct": 95})
    assert machine.busy_reasons(gpu)[0].startswith("GPU 95 %")
    jobs = dict(quiet, heavy=[{"pid": 7, "cmd": "python tools/replay_plans.py runs/x"}])
    assert machine.busy_reasons(jobs)[0].startswith("heavy offline jobs")


def test_busy_policy(monkeypatch):
    snaps = {"quiet": {"load": [1, 1, 1], "cpus": 28, "gpu": None, "chains": [], "heavy": []}}
    snaps["busy"] = dict(snaps["quiet"], load=[20, 20, 20])
    snaps["chain"] = dict(snaps["quiet"], chains=[{"pid": 9, "cmd": "python g1deploy/node.py"}])
    for name, snap in snaps.items():
        monkeypatch.setattr(machine, "snapshot", lambda own_pid=None, _s=snap: _s)
        _, reasons, warn_ok = machine.check("warn")
        _, _, refuse_ok = machine.check("refuse")
        if name == "quiet":
            assert reasons == [] and warn_ok and refuse_ok
        elif name == "busy":
            assert reasons and warn_ok and not refuse_ok
        else:  # another chain: refused even under warn
            assert reasons and not warn_ok and not refuse_ok
        _, _, wait_ok = machine.check("wait", wait_s=0.0, poll_s=0.0, log=lambda m: None)
        assert wait_ok == (name == "quiet")
    assert machine.check("off") == (None, [], True)


def test_report_flags_a_lagging_or_busy_session():
    rep = {"node": {"events": {}, "plan_latency_s": None, "latency_gate": None, "max_lag_s": 0.52, "legs": []},
           "runner": {"emergency_stop": False, "emergency_stop_t": None, "stream_enabled": 0, "catch_up_resets": 0},
           "recording": {}, "truth": {}, "machine_busy": []}
    assert any("timing is suspect" in f for f in session_report.flags(rep))
    rep["node"]["max_lag_s"], rep["machine_busy"] = 0.2, ["load 17.0 on 28 cores (> 35%)"]
    out = session_report.flags(rep)
    assert not any("timing is suspect" in f for f in out) and any("PC was busy" in f for f in out)
