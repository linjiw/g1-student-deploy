"""The console's remote handling: the runner leaves the student stream on any D-pad press (2026-10-27: D-left = boxing)."""
import json
import queue
from types import SimpleNamespace

from g1deploy import console


def fake(iface="ZMQ"):
    s = SimpleNamespace(runner_state={"iface": iface, "stream": iface == "ZMQ", "control": True, "estop": False},
                        remote={"connected": True, "samples": 0, "last": None}, inbox=queue.Queue(), events=[],
                        stops=[])
    s.event = s.events.append
    s.estop = s.stops.append
    return s


def press(s, b):
    console.Session.on_remote(s, json.dumps({"remote": "press", "button": b, "t": 0.0}))


def test_d_left_while_streaming_stands_the_student_and_alarms():
    s = fake()
    press(s, "Left")
    assert s.inbox.get_nowait() == ("key", "h", "remote D-left")
    assert "BOXING" in s.events[-1] and s.runner_state["set"] == "Left"


def test_d_up_while_streaming_is_the_takeover():
    s = fake()
    press(s, "Up")
    assert s.inbox.get_nowait() == ("key", "t", "remote D-up")


def test_select_stops_and_punch_keys_are_named_in_the_boxing_set():
    s = fake("GAMEPAD")
    press(s, "Select")
    assert s.stops == ["remote Select"]
    s.runner_state["set"] = "Left"
    press(s, "L2")
    assert "LEFT HOOK" in s.events[-1] and s.inbox.empty()


def test_other_presses_are_logged():
    s = fake()
    press(s, "L2")
    assert s.events[-1] == "remote: L2" and s.inbox.empty()
