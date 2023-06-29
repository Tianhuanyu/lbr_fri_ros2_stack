
import time
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from lbr_fri_msgs.msg import LBRCommand, LBRState

from .admittance_controller import AdmittanceController


class AdmittanceControlNode(Node):
    def __init__(self, node_name="admittance_control_node") -> None:
        super().__init__(node_name=node_name)

        # parameters
        self.declare_parameter("robot_description", "")
        self.declare_parameter("base_link", "lbr_link_0")
        self.declare_parameter("end_effector_link", "lbr_link_ee")

        self.init_ = False
        self.lbr_state_ = LBRState()

        self.controller_ = AdmittanceController(
            robot_description=str(self.get_parameter("robot_description").value),
            base_link=str(self.get_parameter("base_link").value),
            end_effector_link=str(self.get_parameter("end_effector_link").value),
        )

        # publishers and subscribers
        self.lbr_state_sub_ = self.create_subscription(
            LBRState,
            "/lbr_state",
            self.on_lbr_state_,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                deadline=Duration(nanoseconds=10 * 1e6),  # 10 milliseconds
            ),
        )
        self.lbr_command_pub_ = self.create_publisher(
            LBRCommand,
            "/lbr_command",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                deadline=Duration(nanoseconds=10 * 1e6),  # 10 milliseconds
            ),
        )

        self.count = 0
        self.time = 0

    def on_lbr_state_(self, lbr_state: LBRState) -> None:
        self.smooth_lbr_state_(lbr_state, 1.0)

        start = time.time()
        lbr_command = self.controller_(self.lbr_state_)
        # time.sleep(np.random.randint(0, 10)/1.e3)
        end = time.time()
        self.time += end - start
        self.count += 1
        self.get_logger().info(f"{self.time / self.count * 1.e3}")
        self.lbr_command_pub_.publish(lbr_command)

    def smooth_lbr_state_(self, lbr_state: LBRState, alpha: float):
        if not self.init_:
            self.lbr_state_ = lbr_state
            self.init_ = True
            return

        self.lbr_state_.measured_joint_position = (
            (1 - alpha) * np.array(self.lbr_state_.measured_joint_position.tolist())
            + alpha * np.array(lbr_state.measured_joint_position.tolist())
        ).data

        self.lbr_state_.external_torque = (
            (1 - alpha) * np.array(self.lbr_state_.external_torque.tolist())
            + alpha * np.array(lbr_state.external_torque.tolist())
        ).data


def main(args=None):
    rclpy.init(args=args)
    admittance_control_node = AdmittanceControlNode()
    rclpy.spin(admittance_control_node)
    rclpy.shutdown()
