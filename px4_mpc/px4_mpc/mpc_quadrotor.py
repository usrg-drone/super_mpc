#!/usr/bin/env python
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import SetMode, CommandBool
from nav_msgs.msg import Odometry

from mars_quadrotor_msgs.msg import PositionCommand  # Correct message type

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
        
        self.goal_sub = self.create_subscription(
            PositionCommand,  # Updated message type
            '/planning/pos_cmd',
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
        self.setpoint_position = np.array([0.0, 0.0, 5.0])

        # Create Quadrotor Model and MPC Controller
        self.model = MultirotorRateModel()
        self.mpc = MultirotorRateMPC(self.model)

    def state_callback(self, msg):
        self.current_state = msg

    def odom_callback(self, msg):
        self.vehicle_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.vehicle_attitude = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ])
        self.vehicle_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

    def nav_goal_callback(self, msg: PositionCommand):
        """ Updates the setpoint from PositionCommand, keeping z fixed at 5m """
        self.setpoint_position = np.array([
            msg.position.x,
            msg.position.y,
            5.0  # Fixed altitude
        ])

        self.get_logger().info(
            f"Received PositionCommand -> New setpoint: ["
            f"{self.setpoint_position[0]}, "
            f"{self.setpoint_position[1]}, "
            f"{self.setpoint_position[2]}]"
        )

    def cmdloop_callback(self):
        error_position = self.vehicle_position - self.setpoint_position
        self.get_logger().info(f"Position Error: {error_position}")

        # Current state vector
        x0 = np.array([
            error_position[0], error_position[1], error_position[2],
            self.vehicle_velocity[0], self.vehicle_velocity[1], self.vehicle_velocity[2],
            self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3]
        ]).reshape(10, 1)

        # Solve MPC
        u_pred, _ = self.mpc.solve(x0)
        thrust_rates = u_pred[0, :]

        # Example scaling for thrust
        thrust_command = thrust_rates[0] * 0.07

        # Create and publish AttitudeTarget
        attitude_target = AttitudeTarget()
        attitude_target.header.stamp = self.get_clock().now().to_msg()
        attitude_target.orientation.w = 1.0
        attitude_target.orientation.x = 0.0
        attitude_target.orientation.y = 0.0
        attitude_target.orientation.z = 0.0
        attitude_target.body_rate.x = thrust_rates[1]
        attitude_target.body_rate.y = thrust_rates[2]
        attitude_target.body_rate.z = -thrust_rates[3]
        attitude_target.thrust = thrust_command
        attitude_target.type_mask = 8

        self.attitude_pub.publish(attitude_target)

    def arm_and_set_mode(self):
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
