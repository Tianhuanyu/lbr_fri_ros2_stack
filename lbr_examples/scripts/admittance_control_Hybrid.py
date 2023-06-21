#!/usr/bin/python3
import os

import kinpy
import numpy as np
import rclpy
import xacro
from ament_index_python import get_package_share_directory
from rclpy import qos
from rclpy.node import Node

from lbr_fri_msgs.msg import LBRCommand, LBRState
from std_msgs.msg import Float64MultiArray

import pathlib

# OpTaS
import optas
from optas.spatialmath import *

from scipy.spatial.transform import Rotation as Rot
from geometry_msgs.msg import Pose

# ros2 launch lbr_bringup lbr_bringup.launch.py model:=med7 sim:=false controller_configurations_package:=lbr_velocity_controllers controller_configurations:=config/sample_config.yml controller:=lbr_velocity_controller 


class TrackingController:
    def __init__(self,
        urdf_string: str,
        end_link_name: str = "lbr_link_ee",
        root_link_name: str = "lbr_link_0",
        f_threshold: np.ndarray = np.array([6.0, 6.0, 6.0, 1.0, 1.0, 1.0]),
        dx_gain: np.ndarray = np.array([1.0, 1.0, 1.0, 20.0, 40.0, 60.0]),
        smooth: float = 0.1,)->None:
        dt = 0.01
        
        pi = optas.np.pi  # 3.141...
        T = 1  # no. time steps in trajectory
        self.f_threshold_ = f_threshold
        self.dx_gain_ = np.diag(dx_gain)
        self.smooth_ = smooth
        # Setup robot
        kuka = optas.RobotModel(
            xacro_filename=urdf_string,
            time_derivs=[1],  # i.e. joint velocity
        )
        self.name = kuka.get_name()

        urdf_string_ = xacro.process(urdf_string)
        self.chain_ = kinpy.build_serial_chain_from_urdf(
            data=urdf_string_, end_link_name=end_link_name, root_link_name=root_link_name
        )

        self.dof_ = len(self.chain_.get_joint_parameter_names())
        self.dq_ = np.zeros(self.dof_)
        

        # setup model
        T = 1
        builder = optas.OptimizationBuilder(T, robots=[kuka], derivs_align=True)
        # Setup parameters
        qc = builder.add_parameter("qc", kuka.ndof)  # current robot joint configuration
        pg = builder.add_parameter("pg", 7) # current robot joint configuration
        fh = builder.add_parameter("fh", 6)  

        # Get joint velocity
        dq = builder.get_model_state(self.name, t=0, time_deriv=1)

        # Get next joint state
        q = qc + dt * dq

        # Get jacobian
        J = kuka.get_global_link_geometric_jacobian(end_link_name, qc)

        # Get end-effector velocity
        dp = J @ dq

        # Get current end-effector position
        pc = kuka.get_global_link_position(end_link_name, qc)
        Rc = kuka.get_global_link_rotation(end_link_name, qc)
        
        # Get 
        Om = skew(dp[3:]) 

        #TODO
        p = Rc.T @ dt @ dp[:3]
        R = Rc.T @ (Om * dt + I3())
        
        # Cost: match end-effector position
        Rotq = Quaternion(pg[3],pg[4],pg[5],pg[6])
        Rg = Rotq.getrotm()
        
        # pg_ee = -Rc.T @ pc + Rc.T @ pg[:3]
        # pg_ee = optas.DM([0.0,0.0,0.0]).T
        # pg_ee = -Rc.T @ pc
        Rg_ee = Rc.T @ Rg

   
        # diffp = optas.diag([0.2, 0.2, 0.0]) @ (p - pg_ee[:3])
        diffp = optas.diag([0.2, 0.2, 0.2]) @ p
        
        diffR = Rg_ee.T @ R
        cvf = Rc.T @ fh[:3]
        diffFl =optas.diag([0.0, 0.0, 0.2]) @ Rc.T @ fh[:3]-  Rc.T @ dp[:3]
        


        rvf = Rc.T @ fh[3:]
        diffFR =  optas.diag([0.3, 0.3, 0.3])@ rvf - Rc.T @dp[3:]
        # diffFr = nf @ nf.T @ (fh[3:] - optas.diag([1e-3, 1e-3, 1e-3]) @ Rc.T @dq[3:])

        W_p = optas.diag([1e4, 1e4, 1e4])
        builder.add_cost_term("match_p", diffp.T @ W_p @ diffp)

        builder.add_leq_inequality_constraint(
            "p_con", diffp.T @ diffp, 0.0000001
        )

        # W_f = optas.diag([1e1, 1e1, 1e1])
        # builder.add_cost_term("match_f", diffFl.T @ W_f @ diffFl)

        W_fr = optas.diag([1e1, 1e1, 1e1])
        builder.add_cost_term("match_fr", diffFR.T @ W_fr @ diffFR)
        
        w_dq = 0.01
        builder.add_cost_term("min_dq", w_dq * optas.sumsqr(dq))

        # w_ori = 1e1
        # builder.add_cost_term("match_r", w_ori * optas.sumsqr(diffR - I3()))

 
        builder.add_leq_inequality_constraint(
            "dq1", dq[0] * dq[0], 4e1
        )
        builder.add_leq_inequality_constraint(
            "dq2", dq[1] * dq[1], 4e1
        )
        builder.add_leq_inequality_constraint(
            "dq3", dq[2] * dq[2], 4e1
        )
        builder.add_leq_inequality_constraint(
            "dq4", dq[3] * dq[3], 4e1
        )
        builder.add_leq_inequality_constraint(
            "dq5", dq[4] * dq[4], 4e1
        )
        builder.add_leq_inequality_constraint(
            "dq6", dq[5] * dq[5], 4e1
        )
        builder.add_leq_inequality_constraint(
            "dq7", dq[6] * dq[6], 4e1
        )
        # builder.add_leq_inequality_constraint(
        #     "eff_z", diffp[2] * diffp[2], 1e-8
        # )

        optimization = builder.build()
        self.solver = optas.CasADiSolver(optimization).setup("sqpmethod")
        # Setup variables required later
        

        self.solution = None

    def compute_target_velocity(self, qc, pg, fh):
        # solution = self.solver.solve()
        dq = np.zeros(7)

        if self.solution is not None:
            self.solver.reset_initial_seed(self.solution)
        else:
            self.solver.reset_initial_seed({f"{self.name}/q": optas.horzcat(qc, qc)})
        # self.solver.reset_parameters({"rcm": self._rcm, "qc": q, "dq_goal": self.dq_})
        self.solver.reset_parameters({"qc": optas.DM(qc), "pg": optas.DM(pg), "fh": optas.DM(fh)})

        self.solution = self.solver.solve()
        if self.solver.did_solve():
            dq = self.solution[f"{self.name}/dq"].toarray().flatten()
        else:
            dq = np.zeros(7)

        return dq
    
    def __call__(self, q: np.ndarray, pg: np.ndarray, tau_ext: np.ndarray) -> np.ndarray:
        
        jacobian = self.chain_.jacobian(q)
        # J^T fext = tau
        if tau_ext.size != self.dof_ or q.size != self.dof_:
            raise BufferError(
                f"Expected joint position and torque with {self.dof_} dof, got {q.size()} amd {tau_ext.size()} dof."
            )

        jacobian_inv = np.linalg.pinv(jacobian, rcond=0.05)

        f_ext = jacobian_inv.T @ tau_ext
        # print("f_ext1 = {0}".format(f_ext))
        f_ext = np.where(
            abs(f_ext) > self.f_threshold_,
            self.dx_gain_ @ np.sign(f_ext) * (abs(f_ext) - self.f_threshold_),
            0.0
        )
        


        # print("self.dx_gain_ = {0}".format(self.dx_gain_))
        # print("f_ext2 = {0}".format(f_ext))

        dq = self.compute_target_velocity(q, pg, f_ext)
        # print("v = {0}".format(jacobian @ dq))
        # dq = self.dq_gain_ @ jacobian_inv @ f_ext
        self.dq_ = (1.0 - self.smooth_) * self.dq_ + self.smooth_ * dq
        return self.dq_, f_ext




class AdmittanceControlNode(Node):
    def __init__(self, node_name="admittance_control_node") -> None:
        super().__init__(node_name=node_name)

        # parameters
        self.declare_parameter("model", "med7")
        self.declare_parameter("end_link_name", "lbr_link_ee")
        self.declare_parameter("root_link_name", "lbr_link_0")
        self.declare_parameter("command_rate", 100.0)
        self.declare_parameter("buffer_len", 20)

        self.model_ = str(self.get_parameter("model").value)
        self.end_link_name_ = str(self.get_parameter("end_link_name").value)
        self.root_link_name_ = str(self.get_parameter("root_link_name").value)
        self.dt_ = 1.0 / float(self.get_parameter("command_rate").value)

        # controller
        path = os.path.join(
            get_package_share_directory("lbr_description"),
            "urdf",
            self.model_,
            f"{self.model_}_dock.urdf.xacro",
        )
        self.urdf_string_ = xacro.process(path)

        self.chain_ = kinpy.build_serial_chain_from_urdf(
            data=self.urdf_string_, end_link_name=self.end_link_name_, root_link_name=self.root_link_name_
        )


        # The controller based on conventional controller
        # self.controller_ = Controller(
        #     urdf_string=path,
        #     end_link_name=self.end_link_name_,
        #     root_link_name=self.root_link_name_,
        # )

        # The controller based on optas
        self.controller_optas = TrackingController(
            urdf_string=path,
            end_link_name=self.end_link_name_,
            root_link_name=self.root_link_name_,
        )

        self.lbr_state_ = None

        # publishers and subscribers
        self.lbr_state_sub_ = self.create_subscription(
            LBRState, "/lbr_state", self.lbr_state_cb_, qos.qos_profile_system_default
        )

        self.lbr_state_sub_ = self.create_subscription(
            LBRState, "/lbr_state", self.lbr_state_cb_, qos.qos_profile_system_default
        )

        self.cam_state_sub_ = self.create_subscription(
            Pose, 'object_pose_topic',self.cam_state_cb_, qos.qos_profile_system_default)

        self.lbr_command_pub_ = self.create_publisher(
            Float64MultiArray, "/lbr_velocity_controller/command", qos.qos_profile_system_default
        )



        self.lbr_command_timer_ = self.create_timer(self.dt_, self.timer_cb_)

        self.joint_position_buffer_len_ = int(self.get_parameter("buffer_len").value)
        self.joint_position_buffer_ = []

        self.external_torque_buffer_len_ = int(self.get_parameter("buffer_len").value)
        self.external_torque_buffer_ = []
        self.init_ = True
        self.goal_pose_ = None
        

        # pg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def lbr_state_cb_(self, msg: LBRState) -> None:
        self.lbr_state_ = msg

    def cam_state_cb_(self, msg:Pose) ->None:
        self.goal_pose_ = msg

    def timer_cb_(self) -> None:
        if not self.lbr_state_:
            return
        # compute control
        q = np.array(self.lbr_state_.measured_joint_position.tolist())

        if(self.init_):
            self.pg = self.chain_.forward_kinematics(q)
            print(self.pg)
            self.init_ = False
            return
        else:
            pass
        pg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pg[:3] = self.pg.pos
        pg[3:] = self.pg.rot

        current_pose = self.chain_.forward_kinematics(q)
        print("The error = {0}".format(pg[:3]-current_pose.pos))

        if len(self.joint_position_buffer_) > self.joint_position_buffer_len_:
            self.joint_position_buffer_.pop(0)
        self.joint_position_buffer_.append(q)

        q = np.zeros_like(q)
        for qi in self.joint_position_buffer_:
            q += qi / len(self.joint_position_buffer_)

        tau_ext = np.array(self.lbr_state_.external_torque.tolist())

        if len(self.external_torque_buffer_) > self.external_torque_buffer_len_:
            self.external_torque_buffer_.pop(0)
        self.external_torque_buffer_.append(tau_ext)

        tau_ext = np.zeros_like(tau_ext)
        for tau_ext_i in self.external_torque_buffer_:
            tau_ext += tau_ext_i / len(self.external_torque_buffer_)

        # dq, f_ext = self.controller_(q, tau_ext)


        

        # TODO
        # Target position should be given according to vision. A static one can be given firstly
        # Param: pg ->
        dq, f_ext = self.controller_optas(q, pg ,tau_ext)
        # TrackingController()

        # command
        command = Float64MultiArray(data = dq.data)
        self.lbr_command_pub_.publish(command)


def main(args=None):
    rclpy.init(args=args)
    admittance_control_node = AdmittanceControlNode()
    rclpy.spin(admittance_control_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
