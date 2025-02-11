#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Pedro Roque, Jaeyoung Lim"
__contact__ = "padr@kth.se, jalim@ethz.ch"

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint
from px4_msgs.msg import ActuatorMotors
from px4_msgs.msg import VehicleTorqueSetpoint
from px4_msgs.msg import VehicleThrustSetpoint

from mpc_msgs.srv import SetPose


def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    # msg.header.stamp = Clock().now().nanoseconds / 1000
    pose_msg.header.frame_id = frame_id
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = float(position[0])
    pose_msg.pose.position.y = float(position[1])
    pose_msg.pose.position.z = float(position[2])
    return pose_msg


class SpacecraftMPC(Node):

    def __init__(self):
        super().__init__('spacecraft_mpc')

        # Get mode; rate, wrench, direct_allocation
        self.mode = self.declare_parameter('mode', 'wrench').value
        self.sitl = True

        # Get namespace
        self.namespace = self.declare_parameter('namespace', '').value
        self.namespace_prefix = f'/{self.namespace}' if self.namespace else ''

        # Get setpoint from rviz (true/false)
        self.setpoint_from_rviz = self.declare_parameter('setpoint_from_rviz', False).value

        # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        # Setup publishers and subscribers
        self.set_publishers_subscribers(qos_profile_pub, qos_profile_sub)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        # Create Spacecraft and controller objects
        if self.mode == 'rate':
            from px4_mpc.models.spacecraft_rate_model import SpacecraftRateModel
            from px4_mpc.controllers.spacecraft_rate_mpc import SpacecraftRateMPC
            self.model = SpacecraftRateModel()
            self.mpc = SpacecraftRateMPC(self.model)
        elif self.mode == 'wrench':
            from px4_mpc.models.spacecraft_wrench_model import SpacecraftWrenchModel
            from px4_mpc.controllers.spacecraft_wrench_mpc import SpacecraftWrenchMPC
            self.model = SpacecraftWrenchModel()
            self.mpc = SpacecraftWrenchMPC(self.model)
        elif self.mode == 'direct_allocation':
            from px4_mpc.models.spacecraft_direct_allocation_model import SpacecraftDirectAllocationModel
            from px4_mpc.controllers.spacecraft_direct_allocation_mpc import SpacecraftDirectAllocationMPC
            self.model = SpacecraftDirectAllocationModel()
            self.mpc = SpacecraftDirectAllocationMPC(self.model)

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([1.0, 0.0, 0.0])
        self.setpoint_attitude = np.array([1.0, 0.0, 0.0, 0.0])

    def set_publishers_subscribers(self, qos_profile_pub, qos_profile_sub):
        self.status_sub = self.create_subscription(
            VehicleStatus,
            f'{self.namespace_prefix}/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile_sub)

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            f'{self.namespace_prefix}/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_profile_sub)
        self.angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            f'{self.namespace_prefix}/fmu/out/vehicle_angular_velocity',
            self.vehicle_angular_velocity_callback,
            qos_profile_sub)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            f'{self.namespace_prefix}/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile_sub)

        if self.setpoint_from_rviz:
            self.set_pose_srv = self.create_service(
                SetPose,
                f'{self.namespace_prefix}/set_pose',
                self.add_set_pos_callback
            )
        else:
            self.setpoint_pose_sub = self.create_subscription(
                PoseStamped,
                f'{self.namespace_prefix}/px4_mpc/setpoint_pose',
                self.get_setpoint_pose_callback,
                0
            )

        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode,
            f'{self.namespace_prefix}/fmu/in/offboard_control_mode',
            qos_profile_pub)
        self.publisher_rates_setpoint = self.create_publisher(
            VehicleRatesSetpoint,
            f'{self.namespace_prefix}/fmu/in/vehicle_rates_setpoint',
            qos_profile_pub)
        self.publisher_direct_actuator = self.create_publisher(
            ActuatorMotors,
            f'{self.namespace_prefix}/fmu/in/actuator_motors',
            qos_profile_pub)
        self.publisher_thrust_setpoint = self.create_publisher(
            VehicleThrustSetpoint,
            f'{self.namespace_prefix}/fmu/in/vehicle_thrust_setpoint',
            qos_profile_pub)
        self.publisher_torque_setpoint = self.create_publisher(
            VehicleTorqueSetpoint,
            f'{self.namespace_prefix}/fmu/in/vehicle_torque_setpoint',
            qos_profile_pub)
        self.predicted_path_pub = self.create_publisher(
            Path,
            f'{self.namespace_prefix}/px4_mpc/predicted_path',
            10)
        self.reference_pub = self.create_publisher(
            Marker,
            f"{self.namespace_prefix}/px4_mpc/reference",
            10)

        if self.sitl:
            self.odom_pub = self.create_publisher(
                Odometry,
                f'{self.namespace_prefix}/odom',
                qos_profile_pub)
        return

    def vehicle_attitude_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_attitude[0] = msg.q[0]
        self.vehicle_attitude[1] = msg.q[1]
        self.vehicle_attitude[2] = -msg.q[2]
        self.vehicle_attitude[3] = -msg.q[3]

    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz

    def vehicle_angular_velocity_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_angular_velocity[0] = msg.xyz[0]
        self.vehicle_angular_velocity[1] = -msg.xyz[1]
        self.vehicle_angular_velocity[2] = -msg.xyz[2]

    def vehicle_status_callback(self, msg):
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        pub.publish(msg)

    def publish_rate_setpoint(self, u_pred):
        thrust_rates = u_pred[0, :]
        thrust_command = thrust_rates[0:3]
        rates_setpoint_msg = VehicleRatesSetpoint()
        rates_setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        rates_setpoint_msg.roll  = float(thrust_rates[3])
        rates_setpoint_msg.pitch = -float(thrust_rates[4])
        rates_setpoint_msg.yaw   = -float(thrust_rates[5])
        rates_setpoint_msg.thrust_body[0] = float(thrust_command[0])
        rates_setpoint_msg.thrust_body[1] = -float(thrust_command[1])
        rates_setpoint_msg.thrust_body[2] = -float(thrust_command[2])
        self.publisher_rates_setpoint.publish(rates_setpoint_msg)

    def publish_wrench_setpoint(self, u_pred):
        thrust_outputs_msg = VehicleThrustSetpoint()
        thrust_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        torque_outputs_msg = VehicleTorqueSetpoint()
        torque_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        thrust_outputs_msg.xyz = [u_pred[0, 0], -u_pred[0, 1], -0.0]
        torque_outputs_msg.xyz = [0.0, -0.0, -u_pred[0, 2]]

        self.publisher_thrust_setpoint.publish(thrust_outputs_msg)
        self.publisher_torque_setpoint.publish(torque_outputs_msg)

    def publish_direct_actuator_setpoint(self, u_pred):
        actuator_outputs_msg = ActuatorMotors()
        actuator_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        # NOTE:
        # Output is float[16]
        # u1 needs to be divided between 1 and 2
        # u2 needs to be divided between 3 and 4
        # u3 needs to be divided between 5 and 6
        # u4 needs to be divided between 7 and 8
        # positve component goes for the first, the negative for the second
        thrust = u_pred[0, :] / self.model.max_thrust  # normalizes w.r.t. max thrust
        # print("Thrust rates: ", thrust[0:4])

        thrust_command = np.zeros(12, dtype=np.float32)
        thrust_command[0] = 0.0 if thrust[0] <= 0.0 else thrust[0]
        thrust_command[1] = 0.0 if thrust[0] >= 0.0 else -thrust[0]

        thrust_command[2] = 0.0 if thrust[1] <= 0.0 else thrust[1]
        thrust_command[3] = 0.0 if thrust[1] >= 0.0 else -thrust[1]

        thrust_command[4] = 0.0 if thrust[2] <= 0.0 else thrust[2]
        thrust_command[5] = 0.0 if thrust[2] >= 0.0 else -thrust[2]

        thrust_command[6] = 0.0 if thrust[3] <= 0.0 else thrust[3]
        thrust_command[7] = 0.0 if thrust[3] >= 0.0 else -thrust[3]

        actuator_outputs_msg.control = thrust_command.flatten()
        self.publisher_direct_actuator.publish(actuator_outputs_msg)

    def publish_sitl_odometry(self):
        msg = Odometry()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.header.stamp = Clock().now().to_msg()
        msg.pose.pose.position.x = self.vehicle_local_position[0]
        msg.pose.pose.position.y = self.vehicle_local_position[1]
        msg.pose.pose.position.z = self.vehicle_local_position[2]
        msg.pose.pose.orientation.w = self.vehicle_attitude[0]
        msg.pose.pose.orientation.x = self.vehicle_attitude[1]
        msg.pose.pose.orientation.y = self.vehicle_attitude[2]
        msg.pose.pose.orientation.z = self.vehicle_attitude[3]
        msg.twist.twist.linear.x = self.vehicle_local_velocity[0]
        msg.twist.twist.linear.y = self.vehicle_local_velocity[1]
        msg.twist.twist.linear.z = self.vehicle_local_velocity[2]
        msg.twist.twist.angular.x = self.vehicle_angular_velocity[0]
        msg.twist.twist.angular.y = self.vehicle_angular_velocity[1]
        msg.twist.twist.angular.z = self.vehicle_angular_velocity[2]
        self.odom_pub.publish(msg)
        return

    def cmdloop_callback(self):

        # Publish odometry for SITL
        if self.sitl:
            self.publish_sitl_odometry()

        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = False
        if self.mode == 'rate':
            offboard_msg.body_rate = True
        elif self.mode == 'direct_allocation':
            offboard_msg.direct_actuator = True
        elif self.mode == 'wrench':
            offboard_msg.thrust_and_torque = True
        self.publisher_offboard_mode.publish(offboard_msg)

        # Set state and references for each MPC
        if self.mode == 'rate':
            x0 = np.array([self.vehicle_local_position[0],
                           self.vehicle_local_position[1],
                           self.vehicle_local_position[2],
                           self.vehicle_local_velocity[0],
                           self.vehicle_local_velocity[1],
                           self.vehicle_local_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3]]).reshape(10, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude,       # attitude
                                  np.zeros(6)), axis=0)         # inputs reference (F, w)
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        elif self.mode == 'wrench':
            x0 = np.array([self.vehicle_local_position[0],
                           self.vehicle_local_position[1],
                           self.vehicle_local_position[2],
                           self.vehicle_local_velocity[0],
                           self.vehicle_local_velocity[1],
                           self.vehicle_local_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude,       # attitude
                                  np.zeros(3),                  # angular velocity
                                  np.zeros(3)), axis=0)         # inputs reference (F, torque)
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        elif self.mode == 'direct_allocation':
            x0 = np.array([self.vehicle_local_position[0],
                           self.vehicle_local_position[1],
                           self.vehicle_local_position[2],
                           self.vehicle_local_velocity[0],
                           self.vehicle_local_velocity[1],
                           self.vehicle_local_velocity[2],
                           self.vehicle_attitude[0],
                           self.vehicle_attitude[1],
                           self.vehicle_attitude[2],
                           self.vehicle_attitude[3],
                           self.vehicle_angular_velocity[0],
                           self.vehicle_angular_velocity[1],
                           self.vehicle_angular_velocity[2]]).reshape(13, 1)
            ref = np.concatenate((self.setpoint_position,       # position
                                  np.zeros(3),                  # velocity
                                  self.setpoint_attitude,       # attitude
                                  np.zeros(3),                  # angular velocity
                                  np.zeros(4)), axis=0)         # inputs reference (u1, ..., u4) for 2D platform
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        # Solve MPC
        u_pred, x_pred = self.mpc.solve(x0, ref=ref)

        # Colect data
        idx = 0
        predicted_path_msg = Path()
        for predicted_state in x_pred:
            idx = idx + 1
            # Publish time history of the vehicle path
            predicted_pose_msg = vector2PoseMsg('map', predicted_state[0:3], self.setpoint_attitude)
            predicted_path_msg.header = predicted_pose_msg.header
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)
        self.publish_reference(self.reference_pub, self.setpoint_position)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if self.mode == 'rate':
                self.publish_rate_setpoint(u_pred)
            elif self.mode == 'direct_allocation' or self.mode == 'direct_allocation_trajectory':
                self.publish_direct_actuator_setpoint(u_pred)
            elif self.mode == 'wrench':
                self.publish_wrench_setpoint(u_pred)

    def add_set_pos_callback(self, request, response):
        self.setpoint_position[0] = request.pose.position.x
        self.setpoint_position[1] = request.pose.position.y
        self.setpoint_position[2] = request.pose.position.z
        self.setpoint_attitude[0] = request.pose.orientation.w
        self.setpoint_attitude[1] = request.pose.orientation.x
        self.setpoint_attitude[2] = request.pose.orientation.y
        self.setpoint_attitude[3] = request.pose.orientation.z
        return response

    def get_setpoint_pose_callback(self, msg):
        self.setpoint_position[0] = msg.pose.position.x
        self.setpoint_position[1] = msg.pose.position.y
        self.setpoint_position[2] = msg.pose.position.z
        self.setpoint_attitude[0] = msg.pose.orientation.w
        self.setpoint_attitude[1] = msg.pose.orientation.x
        self.setpoint_attitude[2] = msg.pose.orientation.y
        self.setpoint_attitude[3] = msg.pose.orientation.z


def main(args=None):
    rclpy.init(args=args)

    spacecraft_mpc = SpacecraftMPC()

    rclpy.spin(spacecraft_mpc)

    spacecraft_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
