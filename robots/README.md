# Robot profiles

`./g1 robot --robot NAME ...` and `./g1 doctor --robot NAME` read `robots/NAME.json`: the robot's network port,
lidar path, hands, odometry and lidar mount calibration. Anything given on the command line wins.

| key | values |
|---|---|
| `iface` | the deploy PC's wired port on the robot switch (`eno1`) |
| `lidar` | `dds`: Unitree's lidar service publishes `rt/utlidar/cloud_livox_mid360`; `livox`: no such service on this firmware, so `livox_ros_driver2` on this PC + `g1deploy/adapters/livox_to_utlidar.py` publish it |
| `body` | `dex3` or `rubber` (the self filter's hands) |
| `odom` | `leg` (joints + IMU, first walks) or `zmq` (LIO bridge) |
| `mount_calib` | output of `./g1 calibrate` for this robot (repo-relative or absolute) |
| `facts` | what was measured on this robot, with dates (documentation only) |
