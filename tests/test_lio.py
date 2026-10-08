import numpy as np
from scipy.spatial.transform import Rotation

from g1deploy import contract as C
from g1deploy.adapters.lio import G1Kinematics, LioFrames, hom, inv, quat_to_rot


def _sensor_body_world(frames, pos, quat, q):
    """Ground truth: where the LIO body frame is in the world for a given pelvis pose + joints."""
    return hom(quat_to_rot(quat), pos) @ frames.t_pelvis_lidar(q) @ frames.t_lidar_body


def test_lio_roundtrip_recovers_pelvis_pose_through_the_waist():
    kin = G1Kinematics()
    f = LioFrames(kin)
    q0 = C.DEFAULT_MJ.copy()
    imu0 = Rotation.from_euler("xyz", [0.02, -0.03, 0.7]).as_quat()[[3, 0, 1, 2]]  # arbitrary start yaw
    t_wp0 = f.initialise(imu0, q0)
    assert abs(t_wp0[2, 3] - 0.79) < 0.05  # standing pelvis height from FK, feet on the floor
    world_true_caminit = _sensor_body_world(f, t_wp0[:3, 3], Rotation.from_matrix(t_wp0[:3, :3]).as_quat()[[3, 0, 1, 2]], q0)
    rng = np.random.default_rng(0)
    for _ in range(5):  # walk + duck: pelvis moves, waist bends
        pos = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.5, 0.8)])
        quat = Rotation.from_euler("xyz", rng.normal(0, 0.2, 3) + [0, 0, rng.uniform(-3, 3)]).as_quat()[[3, 0, 1, 2]]
        q = q0.copy()
        q[12:15] = rng.normal(0, 0.3, 3)  # waist yaw / roll / pitch
        t_caminit_body = inv(world_true_caminit) @ _sensor_body_world(f, pos, quat, q)  # what LIO would report
        p_est, q_est = f.pelvis_pose(t_caminit_body, q)
        assert np.allclose(p_est, pos, atol=1e-6)
        assert np.allclose(quat_to_rot(q_est), quat_to_rot(quat), atol=1e-6)


def test_registered_points_land_on_the_floor():
    f = LioFrames()
    f.initialise([1, 0, 0, 0], C.DEFAULT_MJ)
    # a floor point 1 m ahead, expressed in the LIO start frame, must map back to z = 0
    floor_world = np.array([[1.0, 0.0, 0.0]])
    p_caminit = (inv(f.t_world_caminit)[:3, :3] @ floor_world.T).T + inv(f.t_world_caminit)[:3, 3]
    assert np.allclose(f.points_world(p_caminit), floor_world, atol=1e-9)


def test_pose_filter_gives_smooth_50hz_velocity_from_10hz_fixes():
    from g1deploy.adapters.lio import PoseFilter

    f = PoseFilter()
    rng = np.random.default_rng(1)
    speed, err, vel = 1.2, [], []
    imu_yaw_bias = 0.3  # the pelvis IMU yaw is off by 0.3 rad; LIO yaw is right
    prev = None
    for k in range(250):  # 5 s at 50 Hz
        t = k * 0.02
        true_p = np.array([speed * t, 0.2 * np.sin(t), 0.75])
        true_yaw = 0.1 * t
        imu_q = Rotation.from_euler("z", true_yaw + imu_yaw_bias).as_quat()[[3, 0, 1, 2]]
        if k % 5 == 0:  # 10 Hz LIO fix with 1 cm noise
            f.fix(t, true_p + rng.normal(0, 0.01, 3), Rotation.from_euler("z", true_yaw).as_quat()[[3, 0, 1, 2]], imu_q)
        p, q, v = f.pose(t, imu_q)
        if k > 50:
            err.append(np.linalg.norm(p - true_p))
            yaw = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")[2]
            assert abs(np.arctan2(np.sin(yaw - true_yaw), np.cos(yaw - true_yaw))) < 0.02
            if prev is not None:
                vel.append(np.linalg.norm((p - prev)[:2]) / 0.02)
        prev = p
    assert np.mean(err) < 0.03
    assert np.std(vel) < 0.25 and abs(np.mean(vel) - speed) < 0.1  # no 10 Hz velocity spikes
