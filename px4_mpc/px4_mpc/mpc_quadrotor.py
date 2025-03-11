#!/usr/bin/env python
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import SetMode, CommandBool
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Float64MultiArray

from mars_quadrotor_msgs.msg import PositionCommand  # Correct message type

from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
from px4_mpc.controllers.multirotor_rate_mpc import MultirotorRateMPC

from mars_quadrotor_msgs.msg import PolynomialTrajectory

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
        self.vel_sub = self.create_subscription(
            Odometry,
            '/mavros/local_position/velocity_local',
            self.vel_callback,
            qos_profile
        )
        self.goal_sub = self.create_subscription(
            PositionCommand,  # Updated message type
            '/planning/pos_cmd',
            self.nav_goal_callback,
            qos_profile
        )

        self.ref_traj_sub = self.create_subscription(
            PolynomialTrajectory,  # Updated message type
            '/planning_cmd/poly_traj',
            self.ref_traj_callback,
            qos_profile
        )

        # Publishers
        self.attitude_pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            qos_profile
        )
        self.ref_traj_pub = self.create_publisher(
            Path,
            '/planning_cmd/poly_traj_debug',
            qos_profile
        )
        self.solution_traj_pub = self.create_publisher(
            Path,
            '/mpc/solution_traj',
            qos_profile
        )
        self.solution_control_pub = self.create_publisher(
            Float64MultiArray,
            '/mpc/solution_control',
            qos_profile
        )

        # Clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')

        timer_period = 0.01  # 10 ms
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        timer_period2 = 0.1  # 100 ms
        self.timer2 = self.create_timer(timer_period2, self.debug_callback)
        self.dt = 0.1
        self.time = 0.0
        # State variables
        self.current_state = State()
        self.vehicle_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.ref_traj = None
        self.yref_array = []
        self.x_pred = None
        self.u_pred = None
        # Default setpoint (will be updated by the 2D nav goal)
        self.setpoint_position = np.array([0.0, 0.0, 5.0])

        # Create Quadrotor Model and MPC Controller
        self.model = MultirotorRateModel()
        self.mpc = MultirotorRateMPC(self.model)

    def state_callback(self, msg):
        self.current_state = msg

    def odom_callback(self, msg):
        self.time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
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
 
    def vel_callback(self, msg):
        self.vehicle_velocity = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])

    def nav_goal_callback(self, msg: PositionCommand):
        """ Updates the setpoint from PositionCommand, keeping z fixed at 5m """
        self.setpoint_position = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z,
        ])
        self.setpoint_velocity = np.array([
            msg.velocity.x,
            msg.velocity.y,
            msg.velocity.z,
        ])
        self.get_logger().info(
            f"Received PositionCommand -> New setpoint: ["
            f"{self.setpoint_position[0]}, "
            f"{self.setpoint_position[1]}, "
            f"{self.setpoint_position[2]}]"
        )

    def ref_traj_callback(self, msg: PolynomialTrajectory):
        position_traj = Path()
        position_traj.header = msg.header
        # position_traj.header.frame_id = 'map'
        stamp = self.time
        for i in range(msg.piece_num_pos):
            t = np.arange(0, msg.time_pos[i], self.dt)
            x = np.polyval(msg.coef_pos_x[8*i:8*(i+1)],t)
            y = np.polyval(msg.coef_pos_y[8*i:8*(i+1)],t)
            z = np.polyval(msg.coef_pos_z[8*i:8*(i+1)],t)
            for j in range(len(t)):
                pose = PoseStamped()
                pose.header = msg.header
                pose.header.stamp.sec = int(stamp)
                pose.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
                pose.pose.position.x = x[j]
                pose.pose.position.y = y[j]
                pose.pose.position.z = z[j]
                position_traj.poses.append(pose)
                stamp += self.dt
        idx = 0
        for i in range(msg.piece_num_yaw):
            t = np.arange(0, msg.time_yaw[i], self.dt)
            yaw = np.polyval(msg.coef_yaw[8*i:8*(i+1)],t)
            for j in range(len(t)):
                if idx < len(position_traj.poses):
                    position_traj.poses[idx].pose.orientation.w = np.cos(yaw[j]/2 * np.pi / 180.0)
                    position_traj.poses[idx].pose.orientation.z = np.sin(yaw[j]/2 * np.pi / 180.0)
                    idx += 1
        if len(position_traj.poses) > 0:
            self.ref_traj_pub.publish(position_traj)
            self.ref_traj = position_traj
            
        if self.ref_traj is None:
            return
        if self.vehicle_position is None:
            return
        closest_idx = 0
        closest_dist = 0
        self.yref_array = []
        for i in range(len(self.ref_traj.poses)):
            ref = self.ref_traj.poses[i]
            ref_pos = np.array([ref.pose.position.x, ref.pose.position.y, ref.pose.position.z])
            dist = np.linalg.norm(self.vehicle_position - ref_pos)
            if i == 0 or dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        for i in range(closest_idx, len(self.ref_traj.poses)):
            if i >= self.mpc.ocp_solver.N:
                break
            ref = self.ref_traj.poses[i]
            ref_last = self.ref_traj.poses[i-1] if i==0 else ref
            vx = (ref.pose.position.x - ref_last.pose.position.x) / self.dt
            vy = (ref.pose.position.y - ref_last.pose.position.y) / self.dt
            vz = (ref.pose.position.z - ref_last.pose.position.z) / self.dt
            yref = np.array([
                ref.pose.position.x, ref.pose.position.y, ref.pose.position.z,
                vx, vy, vz,
                ref.pose.orientation.w, ref.pose.orientation.x, ref.pose.orientation.y, ref.pose.orientation.z,
                0.0, 0.0, 0.0, 0.0
            ]).reshape(14, 1)
            self.yref_array.append(yref)


    def cmdloop_callback(self):
        error_position = self.vehicle_position - self.setpoint_position
        self.get_logger().info(f"Position Error: {error_position}")

        # error_velocity = self.vehicle_velocity - self.setpoint_velocity
        # Current state vector
        x0 = np.array([
            self.vehicle_position[0], self.vehicle_position[1], self.vehicle_position[2],
            self.vehicle_velocity[0], self.vehicle_velocity[1], self.vehicle_velocity[2],
            self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3]
        ]).reshape(10, 1)

        i = 0
        if self.ref_traj is None:
            return
        for i in range(len(self.yref_array)):
            if i >= self.mpc.ocp_solver.N:
                break
            self.mpc.ocp_solver.set(i, "yref", self.yref_array[i])
            i += 1
        if i < self.mpc.ocp_solver.N and len(self.yref_array) > 0:
            for j in range(i, self.mpc.ocp_solver.N):
                self.mpc.ocp_solver.set(j, "yref", self.yref_array[-1])
            self.mpc.ocp_solver.set(self.mpc.ocp_solver.N, "yref", self.yref_array[-1][:10])
            

        # Solve MPC
        u_pred, x_pred = self.mpc.solve(x0)
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
        attitude_target.type_mask = attitude_target.IGNORE_ATTITUDE

        self.attitude_pub.publish(attitude_target)

        self.x_pred = x_pred
        self.u_pred = u_pred

    def debug_callback(self):
        if self.x_pred is None or self.u_pred is None:
            return
        solution_path = Path()
        solution_path.header.stamp = self.get_clock().now().to_msg()
        solution_path.header.frame_id = 'world'
        for i in range(len(self.x_pred)):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = self.x_pred[i][0]
            pose.pose.position.y = self.x_pred[i][1]
            pose.pose.position.z = self.x_pred[i][2]
            pose.pose.orientation.w = self.x_pred[i][6]
            pose.pose.orientation.x = self.x_pred[i][7]
            pose.pose.orientation.y = self.x_pred[i][8]
            pose.pose.orientation.z = self.x_pred[i][9]
            solution_path.poses.append(pose)
        self.solution_traj_pub.publish(solution_path)

        # @TODO: Publish control trajectory for debugging
        # control = Float64MultiArray()
        # control.layout.dim.label = "control"
        # control.layout.dim.size = len(self.u_pred)
        # control.layout.dim.stride = len(self.u_pred[0])
        # for i in range(len(self.u_pred)):
        #     control.data.append(self.u_pred[i])
        # self.solution_control_pub.publish(control)

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
