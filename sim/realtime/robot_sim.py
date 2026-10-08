#!/usr/bin/env python3
"""Real-time G1 on the DDS bus for the deployment rehearsal (Path B).

lucid-g1-deploy's `gear_sonic_sim` (the same robot side the C++ runner was rehearsed against:
rt/lowcmd in; rt/lowstate, rt/secondary_imu, ground-truth rt/odostate out; tau = tau_ff +
kp (q* - q) + kd (dq* - dq) from the runner's LowCmd), plus:
- the task's obstacle boxes, injected when the simulator loads its scene;
- a MuJoCo ray-cast MID-360 (same mount + pattern as the student's training sensor) published at
  10 Hz on ZMQ topic `cloud`: points [N,3] f32 world frame, origin [3], stamp (sim s);
- a physics-rate log (root pose, obstacle contact force) for Isaac-rule scoring afterwards.

The robot starts held standing at the task's start xy/yaw until the runner's first LowCmd.

usage: robot_sim.py --task task.json --log out.npz [--iface lo] [--cloud-bind tcp://*:5560] [--duration S]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LUCID = Path(os.environ.get("LUCID_DEPLOY_ROOT", "/home/linjiw/lucid-sonic/lucid-g1-deploy"))
sys.path[:0] = [str(ROOT), str(LUCID / "sim"), str(LUCID / "sdk")]

from g1deploy.perception import mid360  # noqa: E402
from g1deploy.zmq_stream import pack  # noqa: E402
from sim.scene import SceneIndex, augment_spec, obstacle_contact_force, quat_matrix  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--cloud-bind", default="tcp://*:5560")
    ap.add_argument("--cloud-hz", type=float, default=10.0)
    ap.add_argument("--range-noise-m", type=float, default=0.02)
    ap.add_argument("--lidar", choices=("mid360", "grid"), default="mid360",
                    help="mid360: real pattern + per-point timing (unfiltered, deskewed, per-point times); "
                         "grid: the 2 deg grid the student was evaluated with")
    ap.add_argument("--lidar-cfg", default=None, help='JSON, e.g. {"mount_error": {"pitch_deg": 1.0}, "dropout": 0.1}')
    ap.add_argument("--lidar-io", choices=("zmq", "dds"), default="zmq",
                    help="zmq: deskewed world cloud (what a LIO bridge delivers); dds: the G1's own topic "
                         "rt/utlidar/cloud_livox_mid360, raw sensor frame, per-point time, placeholders")
    ap.add_argument("--lidar-clock-offset-s", type=float, default=-1234.5,
                    help="header stamps on a different clock (PC1 vs Orin without chrony); the node must not use them")
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--release-file", default=None,
                    help="hold the pelvis (gantry) after the first LowCmd until this file exists")
    args = ap.parse_args(argv)

    import mujoco
    import zmq
    from gear_sonic_sim.configs import SimLoopConfig
    from gear_sonic_sim.simulator_factory import SimulatorFactory
    import gear_sonic_sim.base_sim as base_sim

    task = json.loads(Path(args.task).read_text())
    boxes = task["scene_boxes_world"]
    z = np.load(task["initial_state_record"])
    f0 = int(task.get("initial_frame", 0))
    q0 = z["root_quaternion_wxyz"][f0]
    yaw0 = math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))
    xy0 = np.asarray(task.get("initial_root_xy", z["root_position_m"][f0][:2]), dtype=float)

    cfg = SimLoopConfig()
    wbc = cfg.load_wbc_yaml()
    wbc.update(ENV_NAME="default", INTERFACE=args.iface, ENABLE_ONSCREEN=bool(args.viewer), ENABLE_OFFSCREEN=False,
               ENABLE_ELASTIC_BAND=False, USE_JOYSTICK=0)

    original = mujoco.MjModel.from_xml_path

    def with_task(path, *a, **kw):
        original(path, *a, **kw)  # registers the "last XML" that gear_sonic_sim re-parses for joint classes
        return augment_spec(mujoco.MjSpec.from_file(str(path)), boxes).compile()

    mujoco.MjModel.from_xml_path = with_task

    def keep_fallen(self):  # never reset: the rehearsal scores falls
        self.fall = self.mj_data.qpos[2] < 0.2

    base_sim.DefaultEnv.check_fall = keep_fallen
    try:
        sim = SimulatorFactory.create_simulator(config=wbc, env_name="default", onscreen=bool(args.viewer),
                                                offscreen=False, enable_image_publish=False)
    finally:
        mujoco.MjModel.from_xml_path = original
    env, bridge = sim.sim_env, sim.unitree_bridge
    m, d = env.mj_model, env.mj_data
    d.qpos[0:2] = xy0
    d.qpos[3:7] = [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
    mujoco.mj_forward(m, d)
    qpos0 = d.qpos.copy()
    index = SceneIndex(m)

    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.bind(args.cloud_bind)
    rays = mid360.ray_grid()
    rng = np.random.default_rng(0)
    pelvis = index.pelvis
    scanner = None
    if args.lidar == "mid360":
        from g1deploy.perception.mid360_scan import Mid360Scanner

        scanner = Mid360Scanner(m, d, frame_hz=args.cloud_hz, range_noise_m=args.range_noise_m, rng=rng,
                                **(json.loads(args.lidar_cfg) if args.lidar_cfg else {}))
        if args.lidar_io == "dds":
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

            from g1deploy.perception.unitree_lidar import TOPIC_CLOUD, encode_cloud

            scanner.keep_raw = True
            cloud_pub = ChannelPublisher(TOPIC_CLOUD, PointCloud2_)
            cloud_pub.Init()
    # encode + serialize + write of a ~1 MB frame takes ~30 ms: off the 200 Hz physics loop, like a driver process
    import queue
    import threading

    out_q = queue.Queue(maxsize=4)

    def publisher():
        while True:
            item = out_q.get()
            if item is None:
                return
            kind, payload = item
            if kind == "dds":
                pts_s, t_s, t0_frame = payload
                cloud_pub.Write(encode_cloud(pts_s, t_s * 1e9, t0_frame + args.lidar_clock_offset_s, rng=rng))
            else:
                sock.send(payload)

    pub_thread = threading.Thread(target=publisher, daemon=True)
    pub_thread.start()

    log = {"t": [], "wall": [], "root_pos": [], "root_quat": [], "contact_n": [], "qpos": []}
    released = args.release_file is None
    print(f"[sim] task {task.get('task_id')} start xy={xy0.round(3).tolist()} yaw={yaw0:.3f} "
          f"boxes={len(boxes)} cloud={args.cloud_bind}", flush=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    seen_cmd, sim_t, next_cloud, step = False, 0.0, 0.0, 0
    t_wall0 = time.monotonic()
    try:
        while sim._running and not stop["flag"]:
            t_step = time.monotonic()
            if not seen_cmd:
                d.qpos[:] = qpos0
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                if bridge.cmd_received():
                    seen_cmd = True
                    print(f"[sim] EVENT first_lowcmd sim_t={sim_t:.3f}", flush=True)
            elif not released:
                # gantry: pelvis pinned at the standing pose, joints driven by the runner (INIT ramp, stand)
                d.qpos[0:7] = qpos0[0:7]
                d.qvel[0:6] = 0.0
                if Path(args.release_file).exists():
                    released = True
                    print(f"[sim] EVENT released sim_t={sim_t:.3f} wall={time.monotonic():.3f}", flush=True)
            env.sim_step()
            sim_t += sim.sim_dt
            if step % 2 == 0:  # 100 Hz log is plenty for scoring at the 50 Hz control rate
                log["t"].append(sim_t)
                log["wall"].append(time.monotonic())
                log["root_pos"].append(d.qpos[0:3].copy())
                log["root_quat"].append(d.qpos[3:7].copy())
                log["contact_n"].append(obstacle_contact_force(m, d, index))
                if step % 4 == 0:
                    log["qpos"].append(d.qpos.copy())
            if scanner is not None:
                scanner.substep(sim_t)
                for frame in scanner.pop_raw_frames():
                    if not out_q.full():
                        out_q.put(("dds", frame))
                for (pts, org, t_end), times in (scanner.pop_frames(with_times=True) if args.lidar_io == "zmq"
                                                 else scanner.pop_frames(with_times=True) and []):
                    if not out_q.full():
                        out_q.put(("zmq", pack([("points", pts.astype(np.float32)), ("origins", org.astype(np.float32)),
                                                ("dt", (times - t_end).astype(np.float32)),
                                                ("origin", org[-1] if len(org) else org.sum(0)),
                                                ("stamp", np.array([t_end])), ("wall", np.array([time.monotonic()]))],
                                               topic=b"cloud")))
            elif sim_t >= next_cloud:
                next_cloud += 1.0 / args.cloud_hz
                pts, origin, _ = mid360.mj_cast(m, d, d.xpos[pelvis].copy(), d.xmat[pelvis].reshape(3, 3).copy(), rays,
                                                range_noise_m=args.range_noise_m, rng=rng)
                sock.send(pack([("points", pts.astype(np.float32)), ("origin", origin.astype(np.float64)),
                                ("stamp", np.array([sim_t])), ("wall", np.array([time.monotonic()]))], topic=b"cloud"))
            if args.duration and sim_t >= args.duration:
                break
            step += 1
            if (rest := sim.sim_dt - (time.monotonic() - t_step)) > 0:
                time.sleep(rest)
    finally:
        np.savez_compressed(args.log, **{k: np.asarray(v) for k, v in log.items()}, sim_dt=sim.sim_dt,
                            wall_s=time.monotonic() - t_wall0, boxes=json.dumps(boxes))
        print(f"[sim] stopped sim_t={sim_t:.2f}s wall={time.monotonic() - t_wall0:.2f}s log={args.log}", flush=True)
        sock.close(0)
        sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
