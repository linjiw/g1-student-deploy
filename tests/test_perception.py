import numpy as np

from g1deploy import bundle
from g1deploy.perception import mid360


def test_mount_matches_flow_sensor_model():
    flow = bundle.load_flow()
    vis = flow.sensor_map.visibility_module()
    m = vis.MID360_INVERTED
    assert np.allclose(mid360.POS_PELVIS, m.pos_pelvis)
    assert np.allclose(mid360.ROT_PELVIS, m.rot_pelvis)
    assert tuple(m.elevation_deg) == mid360.ELEVATION_DEG


def test_ray_grid_matches_raycast_memory():
    flow = bundle.load_flow()
    mem = flow.sensor_map.RayCastMemory([], (0.0, 0.0), mounts=("mid360_inverted",), range_m=4.0)
    _, dirs, rmax = mem.rays[0]
    assert np.allclose(dirs, mid360.ray_grid()) and rmax == 4.0


def test_cloud_memory_sees_box_and_ceiling():
    """A wall-like box and a beam seen from a standing pose land in the right layers."""
    from g1deploy.perception.cloud_memory import MjRayMemory
    from sim.episode import Physics, stand_start_state

    boxes = [{"center_xy": [1.5, 0.0], "half_xy": [0.1, 0.5], "yaw": 0.0, "z_bottom": 0.0, "z_top": 0.6},
             {"center_xy": [2.5, 0.0], "half_xy": [0.075, 0.75], "yaw": 0.0, "z_bottom": 1.15, "z_top": 1.3}]
    sim = Physics(boxes, bundle.sonic_onnx())
    sim.reset(stand_start_state())
    mem = MjRayMemory(sim, (0.0, 0.0), mounts=("mid360_inverted",), latency_s=0.1, range_m=4.0)
    for k in range(20):
        mem.step(k * 0.02, sim.data.qpos[:3], sim.data.qpos[3:7])
    ij = lambda x, y: tuple(np.round((np.array([x, y]) - mem.lo) / mem.res).astype(int))  # noqa: E731
    assert mem.support[ij(1.42, 0.0)] > 0.5  # front face of the low wall
    assert np.nanmin(mem.ceiling[ij(2.45, -0.3)[0]:ij(2.6, 0.3)[0] + 1, ij(2.45, -0.3)[1]:ij(2.6, 0.3)[1] + 1]) < 1.2
    assert mem.stats["self_removed"] > 0 and np.isnan(mem.ceiling[ij(0.3, 0.0)])  # nothing spurious at the robot


def _standing_scanner(mount_error=None):
    from g1deploy.perception.mid360_scan import Mid360Scanner
    from sim.episode import Physics, stand_start_state

    sim = Physics([], bundle.sonic_onnx())
    sim.reset(stand_start_state())
    sc = Mid360Scanner(sim.model, sim.data, mount_error=mount_error, range_noise_m=0.01)
    sc.keep_raw = True
    for k in range(41):
        sc.substep(k * 0.005)
    return sim, sc


def test_unitree_cloud_codec_roundtrip():
    from g1deploy.perception.unitree_lidar import decode_cloud, encode_cloud

    _, sc = _standing_scanner()
    pts, t_s, t0 = sc.pop_raw_frames()[0]
    msg = encode_cloud(pts, t_s * 1e9, t0, rng=np.random.default_rng(0))
    assert msg.point_step == 22 and msg.width > 1.8 * len(pts)  # ~55% placeholders
    p2, t2, stamp = decode_cloud(msg)
    keep = np.linalg.norm(pts, axis=1) > 0.1
    assert np.allclose(np.sort(p2[:, 0]), np.sort(pts[keep, 0].astype(np.float32)), atol=1e-6)
    assert abs(stamp - t0) < 1e-6 and t2.max() <= 0.1 + 1e-6


def test_mount_calibration_recovers_injected_error():
    """Sensor pitched 2 deg / rolled 1 deg / 2 cm low vs the nominal mount: the floor fit brings it back."""
    from g1deploy.perception.cloud_memory import BodyFilter
    from g1deploy.perception.unitree_lidar import CloudFromSensorFrame, Mount, calibrate_mount

    sim, sc = _standing_scanner({"pitch_deg": 2.0, "roll_deg": 1.0, "z_m": -0.02})
    frames = [f[0] for f in sc.pop_raw_frames()]
    q = sim.data.qpos.copy()
    bf = BodyFilter()
    m, rep = calibrate_mount(frames, q, Mount.nominal(), bf)
    assert rep[0]["tilt_deg"] > 1.5 and abs(rep[-1]["tilt_deg"]) < 0.1 and abs(rep[-1]["floor_z_at_sensor_m"]) < 0.005
    conv = CloudFromSensorFrame(bf, m)
    pw, _ = conv(frames[0], np.linspace(0, 0.1, len(frames[0])), 1.0, lambda t: q)
    floor = pw[np.abs(pw[:, 2]) < 0.15]
    assert abs(floor[:, 2].mean()) < 0.01


def test_rear_blind_sector_drops_returns_behind_the_torso():
    """`rear_blind_deg=180` (node --rear-blind-deg): the spotter, rope and gantry behind the robot are never fused;
    returns ahead and to the sides are unchanged (inverted lab-g1 mount)."""
    from pathlib import Path

    from g1deploy.perception.cloud_memory import BodyFilter
    from g1deploy.perception.unitree_lidar import CloudFromSensorFrame, Mount

    mount = Mount.load(Path(__file__).resolve().parents[1] / "robots/lab-g1/mount.json")
    p_t = np.array([[2.0, 0.0, -0.5], [-2.0, 0.0, -0.5], [0.3, 1.5, -0.5], [-0.3, 1.5, -0.5], [-0.2, 0.0, 0.1]])
    p_s = (p_t - mount.pos_torso) @ mount.rot_torso
    q = np.zeros(36)
    q[2], q[3] = 0.78, 1.0
    for deg, kept in ((0.0, 5), (180.0, 2)):
        conv = CloudFromSensorFrame(BodyFilter(margin_m=0.05), mount, rear_blind_deg=deg)
        pw, _ = conv(p_s, np.zeros(len(p_s)), 100.0, lambda t: q)
        assert len(pw) + conv.stats["self_removed"] == kept and conv.stats["rear_blind"] == 5 - kept


def test_near_rear_blind_keeps_far_rear_returns():
    """`rear_blind_range_m=1.0`: the rope/spotter close behind is dropped, a wall 2 m behind is kept."""
    from pathlib import Path

    from g1deploy.perception.cloud_memory import BodyFilter
    from g1deploy.perception.unitree_lidar import CloudFromSensorFrame, Mount

    mount = Mount.load(Path(__file__).resolve().parents[1] / "robots/lab-g1/mount.json")
    p_t = np.array([[2.0, 0.0, -0.5], [-0.5, 0.0, -0.1], [-0.6, 0.3, -0.2], [-2.0, 0.0, -0.5], [-1.5, 1.2, -0.5]])
    p_s = (p_t - mount.pos_torso) @ mount.rot_torso
    q = np.zeros(36)
    q[2], q[3] = 0.78, 1.0
    conv = CloudFromSensorFrame(BodyFilter(margin_m=0.05), mount, rear_blind_deg=180.0, rear_blind_range_m=1.0)
    pw, _ = conv(p_s, np.zeros(len(p_s)), 100.0, lambda t: q)
    assert conv.stats["rear_blind"] == 2 and len(pw) + conv.stats["self_removed"] == 3
