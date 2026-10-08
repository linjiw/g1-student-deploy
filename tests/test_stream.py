import numpy as np

from g1deploy import bundle, contract as C
from g1deploy import zmq_stream as Z


def _plan(n=50, seed=0):
    rng = np.random.default_rng(seed)
    q = np.zeros((n, 36))
    q[:, 0] = np.linspace(0.02, 1.0, n)
    q[:, 3] = 1.0
    q[:, 7:] = rng.normal(0, 0.2, (n, 29))
    rot = np.repeat(np.eye(3)[None], n, 0)
    return {"qpos": q, "rotation": rot, "joint_velocity": rng.normal(0, 1, (n, 29))}


def test_pack_roundtrip_and_header():
    msg = Z.pose_message(np.arange(10, 20), np.ones((10, 29)), np.zeros((10, 29)), np.tile([1, 0, 0, 0], (10, 1)),
                         timestamp=1.5)
    assert msg.startswith(b"pose")
    header, f = Z.unpack(msg)
    assert header["v"] == 1 and header["endian"] == "le" and header["count"] == 10
    assert [x["name"] for x in header["fields"]] == ["joint_pos", "joint_vel", "body_quat", "frame_index", "catch_up",
                                                    "timestamp_monotonic"]
    assert f["joint_pos"].dtype == np.float32 and f["frame_index"].dtype == np.int64
    assert np.array_equal(f["frame_index"], np.arange(10, 20))
    assert len(msg) == 4 + 1280 + 10 * 29 * 4 * 2 + 10 * 4 * 4 + 10 * 8 + 1 + 8


def test_stream_is_isaac_order():
    q = np.tile(np.arange(29, dtype=float), (3, 1))
    _, f = Z.unpack(Z.pose_message(np.arange(3), q, q, np.tile([1, 0, 0, 0], (3, 1))))
    assert np.array_equal(f["joint_pos"][0], C.mj_to_isaac(np.arange(29)))


def test_runner_rows_equal_isaac_feed_rows():
    """At playback tick g the runner reads streamed frames g + 5k; they must equal feed.target rows."""
    flow = bundle.load_flow()
    plan = _plan()
    anchor_q = np.zeros(36)
    anchor_q[3] = 1.0
    anchor_q[7:] = 0.05
    anchor_dq = np.full(29, 0.3)
    anchor_tick = 100
    ticks, q, dq, quat = Z.stream_frames(plan, anchor_tick, anchor_q, anchor_dq, anchor_tick,
                                         anchor_tick + 50 + Z.LOOKAHEAD_FRAMES)
    for j in range(0, 12):
        tgt, _ = flow.feed.target(plan, anchor_q, anchor_dq, j, anchor_q)
        g = anchor_tick + j
        rows = [int(np.flatnonzero(ticks == g + 5 * k)[0]) for k in range(10)]
        assert np.allclose(q[rows], tgt[:, :29], atol=1e-6)
        assert np.allclose(dq[rows], tgt[:, 29:58], atol=1e-6)


def test_streamer_always_extends_the_window():
    """A chunk ending at or before the previous end makes the runner reset playback + heading."""
    s = Z.PoseStreamer.__new__(Z.PoseStreamer)
    sent = []
    s.sock = type("S", (), {"send": lambda self, m: sent.append(Z.unpack(m)[1]["frame_index"])})()
    s.sent, s.last_end = 0, None
    plan = _plan(50)
    q0 = np.zeros(36); q0[3] = 1
    s.send_plan(plan, 100, q0, np.zeros(29), 100)
    s.send_plan(_plan(10), 110, q0, np.zeros(29), 110)  # short plan would end earlier
    assert sent[1][-1] > sent[0][-1]
