"""sensor_msgs::msg::dds_::Imu_ for rt/utlidar/imu_livox_mid360 (absent from unitree_sdk2py; ROS 2 sensor_msgs/Imu).

Defined at module level and without `from __future__ import annotations`: cyclonedds resolves field types from the
annotations, so the nested types must be real objects in this module (`unitree_lidar.imu_type()` builds the same
struct inside a function under postponed annotations, which cyclonedds cannot resolve).
"""

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Quaternion_, Vector3_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Imu_(idl.IdlStruct, typename="sensor_msgs.msg.dds_.Imu_"):
    header: Header_
    orientation: Quaternion_
    orientation_covariance: types.array[types.float64, 9]
    angular_velocity: Vector3_
    angular_velocity_covariance: types.array[types.float64, 9]
    linear_acceleration: Vector3_
    linear_acceleration_covariance: types.array[types.float64, 9]
