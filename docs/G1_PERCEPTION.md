# G1 onboard perception: what the robot gives you, and how this stack uses it

Sources: Unitree G1 developer docs (raw markdown via `doc-cdn.unitree.com/11/<id>/en/11_<id>_en`: 733 lidar
service, 266 lidar instructions, 709 odometer, 734 SLAM, 223 about G1, 267 depth camera, 253 robot state,
1216 time sync), unitree_sdk2(_python), unitree_ros URDFs, livox_ros_driver2 / Livox-SDK2, FAST-LIO, Point-LIO,
and community G1 deployments (Click-and-Traverse, grove-g1, apex-humanoid, ggjbj255/g1-elevation-mapping,
Unitree_G1_Workspace, BeamDojo, GLAD, Gallant). [verify] = must be checked on our robot.

## 1. What the G1 publishes

| what | topic (DDS, domain 0, 192.168.123.x) | type | rate | notes |
|---|---|---|---|---|
| MID-360 cloud | `rt/utlidar/cloud_livox_mid360` | `sensor_msgs::msg::dds_::PointCloud2_` | 10 Hz | frame `livox_frame`; **raw sensor frame of the upside-down lidar, not deskewed**; ~20k points + ~55% (0,0,0) placeholders; point_step 22: x y z intensity f32, ring u16 @16, time f32 @18 = **ns from frame start**; header stamp = frame start on **PC1's clock** |
| MID-360 IMU | `rt/utlidar/imu_livox_mid360` | `sensor_msgs::msg::dds_::Imu_` | 200 Hz | accel in **g**, orientation all zeros; IDL absent from unitree_sdk2py (ours: `perception/unitree_lidar.imu_type()`) |
| joints + pelvis IMU | `rt/lowstate` | `unitree_hg LowState_` | 500 Hz | what SONIC and our node read |
| Unitree odometry | `rt/odommodestate`, `rt/lf/odommodestate` | `unitree_go SportModeState_` | 500 / 20 Hz | **not available in low-level (debug) mode** — unusable with SONIC |
| Unitree SLAM | `rt/unitree/slam_*` + RPC `slam_operate` | PointCloud2_/Odometry_ | — | needs its services and loco; not for debug mode |
| D435i | none on DDS; USB on the Orin (librealsense) | — | 60 Hz | head, 47.6° down; never sees overhead beams (flow F21) — the student does not use it |

- The lidar driver runs on **PC1** (closed), not on the Orin: you cannot restart it from PC2, and its stamps are
  on PC1's clock (one user saw > 20 min skew). Our node never uses the header stamp: it times points on arrival
  (`--lidar-transport-s`). For LIO, run chrony on the Orin against PC1 (`server 192.168.123.161 iburst prefer`).
- [verify] that `rt/utlidar/*` keeps publishing after `MotionSwitcherClient.ReleaseMode()` (one community stack
  runs Point-LIO on it in low-level mode; Unitree's lidar page has no warning, the odometer page does).
- Units shipped after April 2026 carry a MID-360**S**.
- Running your own livox_ros_driver2 makes the lidar send to your host (Livox-SDK2 rewrites the destination IP on
  connect) and **kills `rt/utlidar/*` and Unitree SLAM**. Only do it with Unitree's `lidar_driver` service off.

## 2. Mount (sensor -> torso_link)

URDF `g1_29dof_rev_1_0`: xyz (0.0002835, 0.00003, 0.428434), rpy (π, 0.0511, 0) — upside down, pitched 2.93°;
about 0.472 m above the pelvis with the waist at zero. Unitree's page says −2.3° and gives an ambiguous
(−0, 0, −0.47618) offset; older URDFs have it upright; community setups use 0.41618 m / 0° pitch; one unit was ~3°
off and heads get knocked. **Calibrate per robot** (`tools/calibrate_mount.py`): standing still on open floor, fit the
floor through FK + pelvis IMU, correct pitch/roll/height until the floor is level at z = 0. In sim it recovers an
injected 2° pitch + 1° roll + 2 cm error to < 0.1° / < 5 mm (`tests/test_perception.py`).

Geometry consequences (inverted, −7..+52° sensor band → +7..−52° in the world, sensor ~1.22 m up): floor closer than
~0.9 m to the feet is never in a single scan — it only exists in the accumulated memory. One unit measured a front
blind sector of ±45° azimuth from the head shell [verify on ours with `calibrate_mount.py --png`]; G4d measures what
it costs the student.

## 3. How this stack consumes it (no ROS required on the control path)

```
rt/utlidar/cloud_livox_mid360 ──► node: decode (fields by name) ─► drop placeholders/blind ─► time on arrival
rt/lowstate (joints, pelvis IMU) ─► 200 Hz state ring ─────────────┐
odometry (LIO bridge | leg odometry | sim ground truth) ───────────┤
                                                                   ▼
      per 5 ms slice: pose at that instant -> FK through the waist -> calibrated mount -> world points (deskew)
      -> self filter (mesh capsules of the robot's own model, Dex3 or rubber hands) -> evidence memory
```

Tested end to end in the real-time rehearsal (`sim/realtime/rehearse.py --lidar-io dds`): the simulator publishes
`rt/utlidar/cloud_livox_mid360` in the measured format (real MID-360 scan pattern, per-point time, placeholders,
a deliberately wrong header clock) and the node passes two beams with it.

## 4. Odometry options (the planner needs a smooth 50 Hz pelvis pose, floor at z = 0)

| option | how | status here |
|---|---|---|
| **Point-LIO on `rt/utlidar/*`** (recommended) | ROS 2 on the Orin sees the DDS topics as `/utlidar/cloud_livox_mid360`, `/utlidar/imu_livox_mid360` (rmw_cyclonedds, domain 0). Point-LIO: `lidar_type: 2`, `timestamp_unit: 3` (ns), `acc_norm: 1.0` (accel in g), `gravity_align: true`, `extrinsic_T: [-0.011, -0.02329, 0.04412]`, `extrinsic_R: I`, `extrinsic_est_en: false`, `pcd_save_en: false`; odometry on `/aft_mapped_to_init`. Then `adapters/ros2_lio_bridge.py --odom-topic /aft_mapped_to_init` → node `--odom zmq` | frame math + 50 Hz filter tested; bridge untested (no ROS here) |
| FAST-LIO2 with your own driver | livox_ros_driver2 `xfer_format: 1` (CustomMsg: per-point `offset_time`), lidar 192.168.123.120, host = the Orin; FAST-LIO `lidar_type: 1, scan_line: 4, blind: 0.5, timestamp_unit: 3`, same extrinsic as above, `extrinsic_est_en: false`. `/Odometry` is 10 Hz, stamped at scan end, `camera_init` is **not** gravity aligned (our `LioFrames.initialise` re-levels it; never take roll/pitch from FAST-LIO — our pose filter uses the pelvis IMU for roll/pitch). Orin NX ≈ 40 ms/scan | as above; requires Unitree's lidar_driver off |
| leg odometry (`--odom leg`) | joints + pelvis IMU from `rt/lowstate` only | under-counts travel: lab-g1 0.85–0.99 of FAST-LIO's, MuJoCo ~0.83 of truth (contact 1 cm); its drift smears the lidar memory (gaps close), so a fallback for short first tests |
| Unitree odometer / SLAM | — | unavailable in low-level mode |

Known LIO pitfalls on the G1: keep `extrinsic_est_en` false (grove-g1 #69 diverged to 10^5 m in a corridor);
the robot must be still for ~50 ms at LIO start (gravity init); never write the Livox `extrinsic_parameter`
(rotates points, not the IMU); expect degeneracy in long featureless corridors.

## 5. Robot-side setup checklist

1. Network: PC on the robot switch at 192.168.123.x (wired), DDS domain 0; Orin `ssh unitree@192.168.123.164`.
2. Time: chrony on the Orin/PC against PC1 (doc 1216).
3. Check topics with the robot in damping: `rt/lowstate` 500 Hz, `rt/utlidar/cloud_livox_mid360` 10 Hz.
4. Release Unitree's controller (the runner does it via MotionSwitcher); re-check that the lidar topic still runs.
5. `tools/calibrate_mount.py --iface <nic> --out mount.json --png coverage.png` standing on open floor: floor level,
   blocked sectors listed. Set `G1_BODY_MODEL=dex3` if the robot has Dex3 hands (self filter).
6. Odometry: Point-LIO (above) + bridge, or `--odom leg` for the first short walks.
7. `tools/check_robot.py --iface <nic> --perception --cloud dds`: no returns left on the robot, floor at 0.
