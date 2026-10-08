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
