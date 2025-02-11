#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2023 PX4 Development Team. All rights reserved.
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

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from mavros_msgs.msg import State, AttitudeTarget, PositionTarget
from mavros_msgs.srv import SetMode, CommandBool
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry

from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
from px4_mpc.controllers.multirotor_rate_mpc import MultirotorRateMPC

class QuadrotorMPC(Node):

    def __init__(self):
        super().__init__('quadrotor_mpc')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            qos_profile
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self.odom_callback,
            qos_profile
        )
        
        # NEW: Subscribe to 2D Nav Goal from RViz (PoseStamped)
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.nav_goal_callback,
            qos_profile
        )

        # Publishers
        self.attitude_pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            qos_profile
        )

        # Clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')

        timer_period = 0.01  # 10 ms
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        # State variables
        self.current_state = State()
        self.vehicle_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])

        # Default setpoint (will be updated by the 2D nav goal)
        # Z is set to 5 meters
        self.setpoint_position = np.array([0.0, 0.0, 5.0])

        # Create Quadrotor Model and MPC Controller
        self.model = MultirotorRateModel()
        self.mpc = MultirotorRateMPC(self.model)

    def state_callback(self, msg):
        self.current_state = msg

    def odom_callback(self, msg):
        self.vehicle_position[0] = msg.pose.pose.position.x
        self.vehicle_position[1] = msg.pose.pose.position.y
        self.vehicle_position[2] = msg.pose.pose.position.z
        self.vehicle_attitude[0] = msg.pose.pose.orientation.w
        self.vehicle_attitude[1] = msg.pose.pose.orientation.x
        self.vehicle_attitude[2] = msg.pose.pose.orientation.y
        self.vehicle_attitude[3] = msg.pose.pose.orientation.z
        self.vehicle_velocity[0] = msg.twist.twist.linear.x
        self.vehicle_velocity[1] = msg.twist.twist.linear.y
        self.vehicle_velocity[2] = msg.twist.twist.linear.z

    # NEW: Nav Goal callback to update setpoint_position (Z is kept at 5 m)
    def nav_goal_callback(self, msg: PoseStamped):
        # Use the XY from the 2D goal, but keep Z=5
        self.setpoint_position[0] = msg.pose.position.x
        self.setpoint_position[1] = msg.pose.position.y
        self.setpoint_position[2] = 5.0  # fixed height

        self.get_logger().info(
            f"Received 2D Nav Goal -> New setpoint: ["
            f"{self.setpoint_position[0]}, "
            f"{self.setpoint_position[1]}, "
            f"{self.setpoint_position[2]}]"
        )

    def cmdloop_callback(self):
        # Compute position error
        error_position = self.vehicle_position - self.setpoint_position
        self.get_logger().info(f"Position Error: {error_position}")

        # Current state vector
        x0 = np.array([
            error_position[0],
            error_position[1],
            error_position[2],
            self.vehicle_velocity[0],
            self.vehicle_velocity[1],
            self.vehicle_velocity[2],
            self.vehicle_attitude[0],
            self.vehicle_attitude[1],
            self.vehicle_attitude[2],
            self.vehicle_attitude[3]
        ]).reshape(10, 1)

        # Solve MPC
        u_pred, _ = self.mpc.solve(x0)
        thrust_rates = u_pred[0, :]

        # Example scaling for thrust
        thrust_command = thrust_rates[0] * 0.07

        # Create and publish AttitudeTarget
        attitude_target = AttitudeTarget()
        attitude_target.header.stamp = self.get_clock().now().to_msg()
        # For illustration, ignoring orientation and applying body rates + thrust:
        attitude_target.orientation.w = 1.0
        attitude_target.orientation.x = 0.0
        attitude_target.orientation.y = 0.0
        attitude_target.orientation.z = 0.0
        attitude_target.body_rate.x = thrust_rates[1]
        attitude_target.body_rate.y = thrust_rates[2]
        attitude_target.body_rate.z = -thrust_rates[3]
        attitude_target.thrust = thrust_command
        # type_mask = 8 means ignore orientation, use body rates & thrust
        attitude_target.type_mask = 8

        self.attitude_pub.publish(attitude_target)

    def arm_and_set_mode(self):
        # Example: Arm & set OFFBOARD mode
        if not self.current_state.armed:
            arm_cmd = CommandBool.Request()
            arm_cmd.value = True
            self.arm_client.call_async(arm_cmd)
        
        if self.current_state.mode != "OFFBOARD":
            mode_cmd = SetMode.Request()
            mode_cmd.custom_mode = "OFFBOARD"
            self.set_mode_client.call_async(mode_cmd)


def main(args=None):
    rclpy.init(args=args)
    quadrotor_mpc = QuadrotorMPC()
    rclpy.spin(quadrotor_mpc)
    quadrotor_mpc.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
