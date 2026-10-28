"""Recording health and evidence under faults (CONTRACT v3 G; review F08): the recorder's stats.json heartbeat and
finalized.json, recording_health(), manifest v2 (required per kind of session, present / partial / missing / n/a,
complete only when finalized and closed), and the faults: a write error (disk full), the quota, low disk, a slow writer,
an abrupt loss, a cut last chunk, a missing optional stream, a lost console; the console's own files on a full disk.
Fixture-level: a Recorder fed by a thread instead of DDS."""

import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from g1deploy import evidence as E, recorder as R, recording

ROOT = Path(__file__).resolve().parents[1]


def feed(rec, hz=200.0, lowcmd=True, lidar=True, status=True, until=None):
    """DDS stand-in: lowstate (+ lowcmd) at hz, lidar at 10 Hz, node status at 5 Hz, until the recorder stops."""
    k = 0
    t_end = None if until is None else time.monotonic() + until
    while not rec.stop.is_set() and (t_end is None or time.monotonic() < t_end):
        t = time.monotonic()
        rec.writer("lowstate").add(t, tick=k, q=np.full(35, k % 7, np.float32))
        if lowcmd:
            rec.writer("lowcmd").add(t, q=np.zeros(35, np.float32))
        if lidar and k % int(hz / 10) == 0:
            rec.writer("lidar", compress=True).add(t, data=bytes(1500), width=1)
        if status and k % int(hz / 5) == 0:
            rec.status.add(t, {"t": t, "t_unix": time.time(), "status": {"phase": "WAIT"}})
        k += 1
        time.sleep(1.0 / hz)


def run_recorder(out, *extra, seconds=1.2, setup=None, **feed_kw):
    args = R.parse_args(["--iface", "lo", "--out", str(out), "--zmq", "", "--status", "", "--flush-s", "0.25",
                         "--seconds", str(seconds), *extra])
    rec = R.Recorder(args)
    if setup:
        setup(rec)
    rec.start_dds = lambda: threading.Thread(target=feed, args=(rec,), kwargs=feed_kw, daemon=True).start()
    rec.run()
    return rec


def session_dir(root, name="practice-120000-deploy", kind="practice", odom="dds", closed=True, armed=True, **files):
    """A console session's files around a recording (rec/ is filled by the caller)."""
    d = root / name
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps({"mode": "robot" if kind == "robot" else "sim", "odom": odom,
                                                "deploy_identity": {"short": "a"},
                                                "confirm": {"gantry": True, "estop": True}}))
    if kind == "practice":
        (d / "hub_start.json").write_text(json.dumps({"practice": True, "mode": "sim"}))
    if kind == "rehearsal":
        (d / "rehearsal.json").write_text(json.dumps({"scan": "scan-a", "kind": "exact"}))
        (d / "sim.npz").write_bytes(b"x" * 10)
    if armed:
        for f in ("runner.log", "node.log", "node_plans.json", "node_memory.npz"):
            (d / f).write_text("x")
    # the console's own record: Arm and Start SONIC carried out (armed), else only a lease
    acts = [{"id": "a0", "act": "arm", "result": "done"}, {"id": "a1", "act": "s", "result": "done"}] if armed else \
        [{"id": "a0", "act": "lease", "result": "done"}]
    (d / "console_events.jsonl").write_text("".join(json.dumps(a) + "\n" for a in acts))
    if kind != "sim":
        (d / "scene_scan.txt").write_text("scan-a\n")
        (root / "scans" / "scan-a").mkdir(parents=True, exist_ok=True)
        (root / "scans" / "scan-a" / "scan.json").write_text("{}")
    if closed:
        (d / "console_summary.json").write_text("{}")
        (d / "report.md").write_text("# report\n")
    for f, text in files.items():
        (d / f).write_text(text)
    return d


def items(m):
    return {i["id"]: i for i in m["items"]}


# ------------------------------------------------------------------------------------------------ the recorder
def test_recorder_heartbeat_streams_and_finalized(tmp_path):
    seen = {}

    def setup(rec):
        def peek():  # meta.json carries the lidar fields while the recording runs (a cut recording stays decodable)
            time.sleep(1.4)
            seen["meta"] = json.loads((rec.out / "meta.json").read_text())
            seen["stats"] = json.loads((rec.out / "stats.json").read_text())
        rec.lidar_fields, rec.meta_dirty = [{"name": "x", "offset": 0, "datatype": 7, "count": 1}], True
        threading.Thread(target=peek, daemon=True).start()
    run_recorder(tmp_path / "rec", seconds=2.0, setup=setup)
    assert seen["meta"]["lidar_fields"][0]["name"] == "x" and "stopped_unix" not in seen["meta"]
    live = seen["stats"]
    assert live["format"] == R.STATS_FORMAT and live["pid"] == os.getpid() and live["state"] == "recording"
    assert abs(live["t_unix"] - time.time()) < 5 and live["disk"]["free_bytes"] > 0 and not live["finalized"]
    low = live["streams"]["lowstate"]
    assert low["expected_hz"] == 200.0 and 100 < low["rate_hz"] < 260 and low["gap_s"] == 0.1
    assert low["chunks"] >= 3 and low["bytes"] > 0 and low["dropped"] == 0 and low["last_write_unix"] > 0
    assert live["streams"]["node_status"]["count"] >= 4
    st = json.loads((tmp_path / "rec" / "stats.json").read_text())
    fin = json.loads((tmp_path / "rec" / "finalized.json").read_text())
    assert st["finalized"] and st["state"] == "finalized" and fin["clean"] and fin["reason"].startswith("--seconds")
    assert fin["streams"]["lowstate"]["samples"] == st["streams"]["lowstate"]["samples"] > 250
    rows = recording.open_session(tmp_path).stream("lowstate")
    assert len(rows["t"]) == fin["streams"]["lowstate"]["samples"]  # every sample counted is on disk


def test_a_complete_session_and_its_health(tmp_path):
    d = session_dir(tmp_path)
    run_recorder(d / "rec")
    m = E.manifest(d)
    st = items(m)
    assert m["version"] == 2 and m["kind"] == "practice" and m["state"] == "ended" and m["finalized"]
    assert m["complete"], m["incomplete_reasons"]
    assert [i["id"] for i in m["items"]] == list(E.ITEMS)
    for id_ in ("lowstate", "lowcmd", "lidar_raw", "node_status"):
        assert st[id_]["status"] == "present" and st[id_]["required"] and 0.8 < st[id_]["span_s"] < 1.5, st[id_]
        assert st[id_]["gaps"] == 0 and st[id_]["dropped"] == 0
    assert st["lio_odom"]["status"] == "n/a" and not st["lio_odom"]["required"]
    assert st["attempts"]["status"] == "n/a" and not st["attempts"]["required"]  # no GO: does not apply
    assert st["video"]["policy"] == "optional" and st["runner_log"]["policy"] == "optional"
    h = E.recording_health(d)
    assert h["state"] == "finalized" and h["finalized"] and not h["recorder_alive"] and h["ok"], h["problems"]
    rows = {r["id"]: r for r in h["streams"]}
    assert set(rows) >= {"lowstate", "lowcmd", "lidar_raw", "node_status", "lio_odom", "sim_truth"}
    assert rows["lowstate"]["required"] and rows["lowstate"]["state"] == "stopped" and not rows["lowstate"]["active"]
    assert rows["lowstate"]["expected_hz"] == 200.0 and rows["lowstate"]["bytes"] > 0
    assert rows["lio_odom"]["state"] == "n/a" and rows["lidar_imu"]["state"] == "absent"  # optional, never seen
    assert h["disk_free_gb"] > 0 and h["errors"] == []


def test_a_write_error_is_counted_and_the_recording_goes_on(tmp_path):
    def setup(rec):
        w = rec.writer("lidar", compress=True)

        def full(rows):
            raise OSError(errno.ENOSPC, "No space left on device")
        w.write = full
    d = session_dir(tmp_path)
    run_recorder(d / "rec", setup=setup)
    st = json.loads((d / "rec" / "stats.json").read_text())
    li, low = st["streams"]["lidar"], st["streams"]["lowstate"]
    assert li["write_errors"] >= 3 and li["dropped"] == li["samples"] > 5 and li["drop_reasons"] == {"write error": li["dropped"]}
    assert "No space left" in st["errors"]["write:lidar"] and low["dropped"] == 0 and low["chunks"] >= 3
    fin = json.loads((d / "rec" / "finalized.json").read_text())
    assert fin["clean"] is False and fin["dropped_total"] == li["dropped"]
    m = E.manifest(d)
    assert items(m)["lidar_raw"]["status"] == "missing" and not m["complete"]  # nothing of it reached the disk
    h = E.recording_health(d)
    assert any("lidar_raw" in p and "dropped" in p for p in h["problems"]) and h["errors"] and not h["ok"]


def test_quota_and_low_disk_pause_and_count(tmp_path, monkeypatch):
    rec = run_recorder(tmp_path / "q", "--quota-gb", "0.00002")  # 20 kB
    st = json.loads((tmp_path / "q" / "stats.json").read_text())
    assert "quota" in st["errors"] and st["streams"]["lowstate"]["drop_reasons"].get("quota", 0) > 50
    assert st["bytes_total"] < 0.2e6 and rec.paused == "quota"
    du = namedtuple("du", "total used free")
    free = {"v": 0.2e9}
    monkeypatch.setattr(R, "disk_usage", lambda p: du(100e9, 100e9 - free["v"], free["v"]))
    threading.Timer(0.6, lambda: free.update(v=50e9)).start()
    run_recorder(tmp_path / "d", seconds=2.2)
    st = json.loads((tmp_path / "d" / "stats.json").read_text())
    low = st["streams"]["lowstate"]
    assert low["drop_reasons"].get("disk low", 0) > 20 and low["chunks"] >= 2  # paused, then resumed
    assert "resumed" in st["errors"]["disk"] and st["paused"] is None
    assert [e["where"] for e in st["error_log"]].count("disk") == 2


def test_a_slow_writer_grows_its_queue_drops_and_keeps_the_heartbeat(tmp_path):
    seen = {}

    def setup(rec):
        w = rec.writer("lowstate")
        real = w.write

        def slow(rows):
            time.sleep(1.2)
            real(rows)
        w.write = slow

        def peek():
            time.sleep(1.6)
            seen["stats"], seen["at"] = json.loads((rec.out / "stats.json").read_text()), time.time()
        threading.Thread(target=peek, daemon=True).start()
    run_recorder(tmp_path / "rec", "--pending-scale", "0.0001", seconds=2.5, setup=setup)  # lowstate queue: 6.4 kB (~40 rows)
    live = seen["stats"]
    assert seen["at"] - live["t_unix"] < 1.5  # the heartbeat went on while the writer was stuck
    assert live["writer"]["busy_s"] is not None and live["writer"]["busy_s"] > 0.2
    st = json.loads((tmp_path / "rec" / "stats.json").read_text())
    low = st["streams"]["lowstate"]
    assert low["drop_reasons"].get("queue full", 0) > 50 and low["chunks"] >= 1 and st["finalized"]


ABRUPT = """
import sys, threading, time
sys.path.insert(0, {root!r})
import numpy as np
from g1deploy import recorder as R
args = R.parse_args(["--iface", "lo", "--out", {out!r}, "--zmq", "", "--status", "", "--flush-s", "0.25"])
rec = R.Recorder(args)
def feed():
    k = 0
    while True:
        t = time.monotonic()
        rec.writer("lowstate").add(t, tick=k, q=np.zeros(35, np.float32))
        if k % 20 == 0:
            rec.writer("lidar", compress=True).add(t, data=bytes(1500), width=1)
        k += 1
        time.sleep(0.005)
rec.start_dds = lambda: threading.Thread(target=feed, daemon=True).start()
rec.run()
"""


def test_abrupt_loss_and_a_cut_last_chunk(tmp_path):
    d = session_dir(tmp_path, "sim-120000-deploy", kind="sim", armed=False, closed=False)
    proc = subprocess.Popen([sys.executable, "-c", ABRUPT.format(root=str(ROOT), out=str(d / "rec"))])
    try:
        t_end = time.monotonic() + 20
        while time.monotonic() < t_end:
            st = E._json(d / "rec" / "stats.json") or {}
            if (st.get("streams", {}).get("lowstate", {}).get("chunks") or 0) >= 4:
                break
            time.sleep(0.2)
        h = E.recording_health(d)
        rows = {r["id"]: r for r in h["streams"]}
        assert h["state"] == "recording" and h["recorder_alive"] and rows["lowstate"]["active"]
        assert rows["lowcmd"]["state"] == "n/a"  # before Arm
        m = E.manifest(d)
        assert m["state"] == "recording" and not m["complete"] and "still running" in m["incomplete_reasons"][0]
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(10)
    h = E.recording_health(d)
    assert h["state"] == "not finalized" and not h["recorder_alive"] and not h["finalized"]
    assert any("without finalizing" in p for p in h["problems"])
    m = E.manifest(d)
    st = items(m)
    assert m["state"] == "lost" and m["finalized"] is False and not m["complete"]
    assert any("not finalized" in w for w in m["incomplete_reasons"])
    assert st["lowstate"]["status"] == "present" and st["lowstate"]["span_s"] > 0.5  # usable up to the last chunk
    n_before = len(recording.open_session(d).stream("lowstate")["t"])
    chunks = sorted((d / "rec" / "lowstate").glob("[0-9]*.npz"))
    last = chunks[-1]
    last.write_bytes(last.read_bytes()[: last.stat().st_size // 2])  # cut while it was written (no fsync, power loss)
    (d / "rec" / "lowstate" / "999999.tmp.npz").write_bytes(b"PK")  # an unfinished write
    rec = recording.open_session(d)
    rows = rec.stream("lowstate")
    assert 0 < len(rows["t"]) < n_before and rec.skipped and rec.skipped[0][:2] == ("lowstate", last.name)
    st = items(E.manifest(d))
    assert st["lowstate"]["status"] == "partial" and last.name in st["lowstate"]["reason"]
    assert st["lowstate"]["unreadable"] == [last.name] and "unfinished" in st["lowstate"]["reason"]


def test_missing_optional_records_keep_a_session_complete(tmp_path):
    d = session_dir(tmp_path)
    (d / "report.md").unlink()
    (d / "runner.log").unlink()
    (d / "node.log").write_text("x")  # armed: the node ran
    run_recorder(d / "rec")
    m = E.manifest(d)
    st = items(m)
    assert st["report"]["status"] == "missing" and not st["report"]["required"]
    assert st["runner_log"]["status"] == "missing" and not st["runner_log"]["required"]
    assert m["complete"], m["incomplete_reasons"]
    (d / "node_plans.json").unlink()  # a required one
    m = E.manifest(d)
    assert not m["complete"] and m["missing_required"] == ["plans"]


def test_required_records_per_kind_of_session(tmp_path):
    robot = session_dir(tmp_path, "robot-1", kind="robot", odom="fused",
                        **{"attempts.jsonl": json.dumps({"n": 1, "rehearsal": {"set": "s1"}}) + "\n"})
    run_recorder(robot / "rec", "--no-lidar")
    m = E.manifest(robot)
    st = items(m)
    assert st["lio_odom"]["required"] and st["lio_odom"]["status"] == "missing"  # fused on the robot: LIO required
    assert st["lidar_raw"]["status"] == "missing" and "--no-record-lidar" in st["lidar_raw"]["reason"]
    assert st["rehearsal"]["required"] and st["rehearsal"]["status"] == "present"
    assert st["start_checks"]["status"] == "missing" and not m["complete"]
    practice = session_dir(tmp_path, "practice-2", odom="lio")
    assert items(E.manifest(practice))["lio_cloud"]["required"]  # the stand-in counts in practice
    reh = session_dir(tmp_path, "rehearse-3", kind="rehearsal")
    st = items(E.manifest(reh))
    assert st["rehearsal"]["required"] and st["rehearsal"]["status"] == "present" and st["start_checks"]["status"] == "n/a"
    assert E.policy("practice", False)["rehearsal"] == "optional"  # practice: the rehearsal is advisory
    run_recorder(reh / "rec")
    assert {r["id"]: r for r in E.recording_health(reh)["streams"]}["sim_truth"]["required"]
    went = session_dir(tmp_path, "practice-4")  # GO accepted, but no attempts.jsonl
    (went / "console_events.jsonl").write_text(json.dumps({"id": "g1", "act": "g", "result": "done"}) + "\n")
    st = items(E.manifest(went))
    assert st["attempts"]["required"] and st["attempts"]["status"] == "missing"


def test_a_stream_that_dies_mid_session_is_partial(tmp_path):
    d = session_dir(tmp_path)
    t0 = 1000.0
    for sub, n_end in (("lowstate", 60.0), ("lidar", 20.0), ("lowcmd", 60.0)):
        (d / "rec" / sub).mkdir(parents=True)
        np.savez(d / "rec" / sub / "000000.npz", t=np.linspace(t0, t0 + n_end, 50))
    (d / "rec" / "node_status.jsonl").write_text("".join(json.dumps({"t": t0 + k}) + "\n" for k in range(60))
                                                 + '{"t": 10')  # a line cut by the end
    (d / "rec" / "meta.json").write_text(json.dumps({"pid": 1, "lidar": True, "zmq": []}))
    (d / "rec" / "finalized.json").write_text(json.dumps({"reason": "stopped"}))
    (d / "rec" / "stats.json").write_text(json.dumps({"streams": {"lowstate": {"gaps": 3, "gap_total_s": 4.5, "gap_s": 0.1}}}))
    st = items(E.manifest(d))
    assert st["lidar_raw"]["status"] == "partial" and "stopped 40 s before lowstate" in st["lidar_raw"]["reason"]
    assert st["lowstate"]["status"] == "partial" and "silent 4.5 s in 3 gap(s)" in st["lowstate"]["reason"]
    assert st["node_status"]["status"] == "present" and "cut line" in st["node_status"]["reason"]


def test_a_lost_console_leaves_the_recorder_to_finalize_itself(tmp_path):
    args = R.parse_args(["--iface", "lo", "--out", str(tmp_path / "rec"), "--orphan-grace-s", "0"])
    rec = R.Recorder(args)
    rec.check_parent()
    assert rec.orphan_t is None and not rec.stop.is_set()  # our parent is still here
    rec.ppid0 = os.getppid() + 100000  # as if the console that started it were gone
    rec.check_parent()
    assert "exited without stopping the recorder" in rec.errors["parent"] and rec.stats()["state"] == "orphan"
    rec.check_parent()
    assert rec.stop.is_set() and "is gone" in rec.stop_reason


def test_legacy_recordings_read_as_before(tmp_path):
    """Sessions recorded before finalized.json: a clean stop is meta.json's stopped_unix."""
    d = session_dir(tmp_path)
    (d / "rec" / "lowstate").mkdir(parents=True)
    np.savez(d / "rec" / "lowstate" / "000000.npz", t=np.linspace(0, 1, 10))
    (d / "rec" / "meta.json").write_text(json.dumps({"format": R.FORMAT, "lidar": True, "stopped_unix": 1.0}))
    (d / "rec" / "stats.json").write_text(json.dumps({"streams": {"lowstate": {"count": 10, "rate_hz": 9.0}}}))
    h = E.recording_health(d)
    assert h["finalized"] and h["state"] == "finalized" and "before finalized.json" in h["finalized_reason"]
    assert E.manifest(d)["finalized"]
    (d / "rec" / "meta.json").write_text(json.dumps({"format": R.FORMAT, "lidar": True}))
    assert E.recording_health(d)["state"] == "recording"  # its stats.json is fresh: a legacy recorder running
    os.utime(d / "rec" / "stats.json", (time.time() - 60, time.time() - 60))
    assert E.recording_health(d)["state"] == "not finalized"


# ------------------------------------------------------------------------------------------------ the console
class Broken:
    def write(self, *_):
        raise OSError(errno.ENOSPC, "No space left on device")

    def close(self):
        pass


def test_the_console_survives_a_full_disk(tmp_path):
    from g1deploy import console
    from tests.test_console_acts import arm, session

    s = arm(session(tmp_path))
    s.frames, s.events_log = Broken(), Broken()
    s.attempts.path = tmp_path / "gone" / "attempts.jsonl"  # its directory vanished: every save fails
    s.attempts.on_error = lambda exc: s.write_failed("attempts.jsonl", exc)
    s.record_frame("screen")
    rec = s.run_act(s.make_act("h", "panel"))
    assert rec["result"] == "done" and s.act_result(rec["id"])["result"] == "done"  # the act stands, its receipt answers
    s.attempts.start(1.0, {"phase": "RUN", "xy": [0, 0]})
    s.attempts.update(2.0, {"phase": "ARRIVED", "xy": [1, 0], "goal_dist_m": 0.1})
    assert s.attempts.rows[0]["result"] == "arrived" and s.attempts.save_errors >= 2
    st = s.panel_state()
    assert set(st["write_errors"]) == {"console_frames.jsonl", "console_events.jsonl", "attempts.jsonl"}
    assert sum("cannot write" in e for _, e in s.events) == 3  # shown once per file
    # a child's log on a full disk: its stdout is still drained (an undrained pipe would block the runner)
    p = console.Proc("chatty", [sys.executable, "-c", "import sys\nfor i in range(20000): print('x' * 60)"],
                     tmp_path / "chatty.log", dict(os.environ))
    p.f = Broken()
    p.p.wait(20)
    assert p.p.returncode == 0
