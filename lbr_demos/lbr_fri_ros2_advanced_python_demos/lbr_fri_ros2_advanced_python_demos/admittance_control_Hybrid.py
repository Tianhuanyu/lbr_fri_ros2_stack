#!/usr/bin/python3
import os
import time

import kinpy
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=1000)
import rclpy
import xacro
from ament_index_python import get_package_share_directory
from rclpy import qos
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from lbr_fri_msgs.msg import LBRCommand, LBRState
from std_msgs.msg import Float64MultiArray

import pathlib
import csv
import copy
# OpTaS
import optas
from optas.spatialmath import *

from scipy.spatial.transform import Rotation as Rot
from geometry_msgs.msg import Pose


# tau = {0} [-1.1552312   3.70274581 -1.09817682 -3.02117871  0.23070508  0.51952975
#  -0.39371812]
def ExtractFromParamsCsv(path):
        # path_pos = os.path.join(
        #     get_package_share_directory("gravity_compensation"),
        #     "test",
        #     "measurements_with_ext_tau.csv",
        # )
        params = []
        with open(path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # print("111 = {0}".format(row.values()))
                params = [float(x) for x in list(row.values())]

        return params

def RNEA_function(Nb,Nk,rpys,xyzs,axes):
    # 2. RNEA
    Nf = Nb+Nk
    """
    input: q, qdot, qddot, model
    output: tau
    """
    
    om0 = cs.DM([0.0,0.0,0.0])
    om0D = cs.DM([0.0,0.0,0.0])
    gravity_para = cs.DM([0.0, 0.0, -9.81])
    # gravity_para = cs.DM([4.905, 0.0, -8.496])

    """
    The definination of joint position from joint0 to joint(Nb-1)
    """

    q = cs.SX.sym('q', Nb, 1)
    qd = cs.SX.sym('qd', Nb, 1)
    qdd = cs.SX.sym('qdd', Nb, 1)

    """
    The definination of mass for link1 to linkee
    The definination of center of Mass for link1 to linkee
    The definination of Inertial tensor for link1 to linkee
    """
    m = cs.SX.sym('m', 1, Nb+1)
    cm = cs.SX.sym('cm',3,Nb+1)
    Icm = cs.SX.sym('Icm',3,3*Nb+3)

    """
    external force given by link0
    The list will be appended via RNEA
    """
    fs = [cs.DM([0.0,0.0,0.0])]
    ns = [cs.DM([0.0,0.0,0.0])]


    """
    $$ Forward part of RNEA $$
    oms,omDs,vDs given by link0. The list will be appended via RNEA.
    Notes: the gravity_para represents a base acceration to subsitute the gravity.

    """
    oms = [om0]
    omDs = [om0D]
    vDs = [-gravity_para]
    
    
    # 2.1 forward part of RNEA
    """
    link0->1, link1->2 ...., link7->end-effector (for example)
    joint0,   joint1 ....
    
    """
    
    for i in range(Nf):
        print("link p({0}) to i{1}".format(i,i+1))
        if(i!=Nf-1):
            # print(joints_list_r1)
            iRp = (rpy2r(rpys[i]) @ angvec2r(q[i], axes[i])).T
            iaxisi = iRp @ axes[i]
            omi = iRp @ oms[i] + iaxisi* qd[i]
            omDi = iRp @ omDs[i] +  skew(iRp @oms[i]) @ (iaxisi*qd[i]) + iaxisi*qdd[i]
        else:
            iRp = rpy2r(rpys[i]).T
            omi = iRp @ oms[i]
            omDi = iRp @ omDs[i]

        vDi = iRp @ (vDs[i] 
                        + skew(omDs[i]) @ xyzs[i]
                    + skew(oms[i]) @ (skew(oms[i])@ xyzs[i]))
        
        fi = m[i] * (vDi + skew(omDi)@ cm[:,i]+ skew(omi)@(skew(omi)@cm[:,i]))
        ni = Icm[:,i*3:i*3+3] @ omDi + skew(omi) @ Icm[:,i*3:i*3+3] @ omi #+ skew(cm[:,i]) @ fi
        

        oms.append(omi)
        omDs.append(omDi)
        vDs.append(vDi)
        fs.append(fi)
        ns.append(ni)


    """
    $$ Backward part of RNEA $$
    """

    # pRi = rpy2r(rpys[-1])
    ifi = fs[-1]#cs.DM([0.0,0.0,0.0])
    ini = ns[-1] + skew(cm[:,-1]) @ fs[-1]#cs.DM([0.0,0.0,0.0])
    # ifi = cs.DM([0.0,0.0,0.0])
    # ini = cs.DM([0.0,0.0,0.0])
    taus = []

    # print("Backward: fs[i+1] {0}".format(len(fs)))
    for i in range(Nf-1,0,-1):

        print("Backward: link i({0}) to p{1}".format(i+1,i))
        # print("Backward: fs[i+1]".format(fs[i+1]))
        if(i < Nf-1):
            pRi = rpy2r(rpys[i]) @ angvec2r(q[i], axes[i])
        elif(i == Nf-1):
            pRi = rpy2r(rpys[i])
        else:
            pRi = rpy2r(rpys[i])
        

        ini = ns[i] + pRi @ ini +skew(cm[:,i-1]) @ fs[i] +skew(xyzs[i]) @ pRi @ifi
        ifi= pRi @ ifi + fs[i]
        pRi = rpy2r(rpys[i-1]) @ angvec2r(q[i-1], axes[i-1])
        _tau = ini.T @pRi.T @ axes[i-1]
        taus.append(_tau)


        
    tau_=cs.vertcat(*[taus[k] for k in range(len(taus)-1,-1,-1)])
    dynamics_ = optas.Function('dynamics', [q,qd,qdd,m,cm,Icm], [tau_])
    return dynamics_



def DynamicLinearlization(dynamics_,Nb):
    """
    The definination of joint position from joint0 to joint(Nb-1)
    """

    q = cs.SX.sym('q', Nb, 1)
    qd = cs.SX.sym('qd', Nb, 1)
    qdd = cs.SX.sym('qdd', Nb, 1)

    """
    The definination of mass for link1 to linkee
    The definination of center of Mass for link1 to linkee
    The definination of Inertial tensor for link1 to linkee
    """
    m = cs.SX.sym('m', 1, Nb+1)
    cm = cs.SX.sym('cm',3,Nb+1)
    Icm = cs.SX.sym('Icm',3,3*Nb+3)

    
    Y = []
        
    for i in range(Nb):
        # for every link
        # Y_line = []
        Y_line = []
        # PI_a = []
        for j in range(m.shape[1]):
            # for every parameters
            ## 1. get mass
            m_indu = np.zeros([m.shape[1],m.shape[0]])
            cm_indu = np.zeros([3,Nb+1])#np.zeros([cm.shape[1],cm.shape[0]])
            Icm_indu = np.zeros([3,3*Nb+3])#np.zeros([Icm.shape[1],Icm.shape[0]])
            # print(*m.shape)
            m_indu[j] = 1.0
            # print(m_indu)

            output = dynamics_(q,qd,qdd,m_indu,cm_indu,Icm_indu)[i]
            Y_line.append(output)


            ## 2. get cmx
            output1 = dynamics_(q,qd,qdd,m_indu,cm,Icm_indu)[i]-output
            for k in range(3):
                output_cm = optas.jacobian(output1,cm[k,j])
                output_cm1 = optas.substitute(output_cm,cm,cm_indu)
                Y_line.append(output_cm1)

            ## 3.get Icm
            output2 = dynamics_(q,qd,qdd,m_indu,cm_indu,Icm)[i]-output
            for k in range(3):
                for l in range(k,3,1):
                    output_Icm = optas.jacobian(output2,Icm[k,l+3*j])
                    Y_line.append(output_Icm)

            # sx_lst = optas.horzcat(*Y_seg)
            # Y_line
        sx_lst = optas.horzcat(*Y_line)
        Y.append(sx_lst)
        # print("Y_line shape = {0}, {1}".format(Y_line[0].shape[0],Y_line[0].shape[1]))
        # print("sx_lst shape = {0}, {1}".format(sx_lst.shape[0],sx_lst.shape[1]))

    Y_mat = optas.vertcat(*Y)
    # print(Y_mat)
    print("Y_mat shape = {0}, {1}".format(Y_mat.shape[0],Y_mat.shape[1]))
    Ymat = optas.Function('Dynamic_Ymat',[q,qd,qdd],[Y_mat])

    PI_a = []
    for j in range(m.shape[1]):
        # for every parameters
        pi_temp = [m[j],
                    m[j]*cm[0,j],
                    m[j]*cm[1,j],
                    m[j]*cm[2,j],
                    Icm[0,0+3*j] + m[j]*(cm[1,j]*cm[1,j]+cm[2,j]*cm[2,j]),  # XXi
                    Icm[0,1+3*j] - m[j]*(cm[0,j]*cm[1,j]),  # XYi
                    Icm[0,2+3*j] - m[j]*(cm[0,j]*cm[2,j]),  # XZi
                    Icm[1,1+3*j] + m[j]*(cm[0,j]*cm[0,j]+cm[2,j]*cm[2,j]),  # YYi
                    Icm[1,2+3*j] - m[j]*(cm[1,j]*cm[2,j]),  # YZi
                    Icm[2,2+3*j] + m[j]*(cm[0,j]*cm[0,j]+cm[1,j]*cm[1,j])] # ZZi
        PI_a.append(optas.vertcat(*pi_temp))

    PI_vecter = optas.vertcat(*PI_a)
    print("PI_vecter shape = {0}, {1}".format(PI_vecter.shape[0],PI_vecter.shape[1]))
    PIvector = optas.Function('Dynamic_PIvector',[m,cm,Icm],[PI_vecter])

    return Ymat, PIvector

def find_eigen_value(dof, parm_num, regressor_func,shape):
    '''
    Find dynamic parameter dependencies (i.e., regressor column dependencies).
    '''

    samples = 100
    round = 10

    pi = np.pi

    # Z = np.zeros((dof * samples, parm_num))
    A_mat = np.zeros(( shape,shape ))

    for i in range(samples):
        a = np.random.random([parm_num,dof])*2.0-1.0
        b = np.random.random([parm_num,dof])*2.0-1.0

        A_mat = A_mat+ regressor_func(a,b)

    print("U V finished")
    U, s, V = np.linalg.svd(A_mat)
        # Z[i * dof: i * dof + dof, :] = np.matrix(
        #     regressor_func(q, dq, ddq)).reshape(dof, parm_num)

    # R1_diag = np.linalg.qr(Z, mode='r').diagonal().round(round)
    

    return U,V

def getJointParametersfromURDF(robot, ee_link="lbr_link_ee"):
    robot_urdf = robot.urdf
    root = robot_urdf.get_root()
    # ee_link = "lbr_link_ee"
    xyzs, rpys, axes = [], [], []


    joints_list = robot_urdf.get_chain(root, ee_link, links=False)
    # print("joints_list = {0}"
    #         .format(joints_list)
    #         )
    # assumption: The first joint is fixed. The information in this joint is not recorded
    """
    xyzs starts from joint 0 to joint ee
    rpys starts from joint 0 to joint ee
    axes starts from joint 0 to joint ee
    """
    joints_list_r = joints_list[1:]
    for joint_name in joints_list_r:
        print(joint_name)
        joint = robot_urdf.joint_map[joint_name]
        xyz, rpy = robot.get_joint_origin(joint)
        axis = robot.get_joint_axis(joint)

        # record the kinematic parameters
        xyzs.append(xyz)
        rpys.append(rpy)
        axes.append(axis)
    print("xyz, rpy, axis = {0}, {1} ,{2}".format(xyzs, rpys, axes))

    Nb = len(joints_list_r)-1
    return Nb, xyzs, rpys, axes



def find_dyn_parm_deps(dof, parm_num, regressor_func):
    '''
    Find dynamic parameter dependencies (i.e., regressor column dependencies).
    '''

    samples = 10000
    round = 10

    pi = np.pi

    Z = np.zeros((dof * samples, parm_num))

    for i in range(samples):
        q = [float(np.random.random() * 2.0 * pi - pi) for j in range(dof)]
        dq = [float(np.random.random() * 2.0 * pi - pi) for j in range(dof)]
        ddq = [float(np.random.random() * 2.0 * pi - pi)
            for j in range(dof)]
        Z[i * dof: i * dof + dof, :] = np.matrix(
            regressor_func(q, dq, ddq)).reshape(dof, parm_num)

    R1_diag = np.linalg.qr(Z, mode='r').diagonal().round(round)
    dbi = []
    ddi = []
    for i, e in enumerate(R1_diag):
        if e != 0:
            dbi.append(i)
        else:
            ddi.append(i)
    dbn = len(dbi)

    P = np.mat(np.eye(parm_num))[:, dbi + ddi]
    Pb = P[:, :dbn]
    Pd = P[:, dbn:]

    Rbd1 = np.mat(np.linalg.qr(Z * P, mode='r'))
    Rb1 = Rbd1[:dbn, :dbn]
    Rd1 = Rbd1[:dbn, dbn:]

    Kd = np.mat((np.linalg.inv(Rb1) * Rd1).round(round))

    return Pb, Pd, Kd

# ros2 launch lbr_bringup lbr_bringup.launch.py model:=med7 sim:=false controller_configurations_package:=lbr_velocity_controllers controller_configurations:=config/sample_config.yml controller:=lbr_velocity_controller 
def csv_save(path, vector):
    with open(path,'a', newline='') as file:
        writer =csv.writer(file)
        writer.writerow(vector)
    return True

def state2Msg(p,q):
    pose = Pose()
    pose.position.x = p[0]
    pose.position.y = p[1]
    pose.position.z = p[2]

    pose.orientation.w = q[0]
    pose.orientation.x = q[1]
    pose.orientation.y = q[2]
    pose.orientation.z = q[3]
    return pose

def quaternion_to_rotation_matrix(q):

    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R

def Msg2state(msg):
    p = np.array([0.0]*3)
    q = np.array([0.0]*4)


    p[0] = msg.position.x
    p[1] = msg.position.y
    p[2] = msg.position.z

    q[0] = msg.orientation.w
    q[1] = msg.orientation.x
    q[2] = msg.orientation.y
    q[3] = msg.orientation.z
    return p,q


def from_msg2T44(geo_msg):
    p,q = Msg2state(geo_msg)
    T44 = from_pq2T44(p,q)
    return T44

def from_pq2T44(p,q):
    T44 = np.zeros((4,4))
    T44[3,3] = 1.0

    R = quaternion_to_rotation_matrix(q)
    T44[:3,:3] = R
    T44[0,3] = p[0]
    T44[1,3] = p[1]
    T44[2,3] = p[2]
    return T44
def rotation_matrix_to_quaternion(R):
    qw = np.sqrt(1 + np.trace(R)) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    
    return np.array([qw, qx, qy, qz])

def from_pose2T44(pose):
    p = pose[:3]
    q = pose[3:]
    T44 = from_pq2T44(p,q)
    return T44

def from_T442pq(T44):
    R = T44[:3,:3]

    q = rotation_matrix_to_quaternion(R)
    p = np.array([0.0]*3)
    p[0] = T44[0,3]
    p[1] = T44[1,3]
    p[2] = T44[2,3]
    
    return p,q
def from_T442pose(T44):
    p,q = from_T442pq(T44)
    pose = np.array([0.0]*7)
    pose[:3] = p
    pose[3:] = q/np.linalg.norm(q)
    return pose

class TrackingController:
    def __init__(self,
        urdf_string: str,
        end_link_name: str = "lbr_link_ee",
        root_link_name: str = "lbr_link_0",
        f_threshold: np.ndarray = np.array([2.0 , 2.0, 5.0, 0.5, 0.5, 0.5]),
        dx_gain: np.ndarray = np.array([0.5, 0.5, 0.2, 1.0, 1.0, 1.0]),
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
        p = dt @ dp[:3]  #p = Rc.T @ dt @ dp[:3]?
        R = Rc.T @ (Om * dt + I3()) @ Rc
        
        # Cost: match end-effector position
        Rotq = Quaternion(pg[4],pg[5],pg[6],pg[3])
        Rg = Rotq.getrotm()
        
        pg_ee = -Rc.T @ pc + Rc.T @ (pg[:3]- optas.DM([0.0,0.0,0.2]))
        # pg_ee = optas.DM([0.0,0.0,0.0]).T
        # pg_ee = -Rc.T @ pc
        Rg_ee = Rc.T @ Rg

        Bx = 0.05
        By = 0.05

        # diffp = optas.diag([2.0, 2.0, 0.0]) @ (p - 0.1*pg_ee[:3])
        # diffp = optas.diag([0.2, 0.2, 0.2]) @ p
        
        diffR = Rg_ee.T @ R
        cvf = Rc.T @ fh[:3]
        # diffFl =optas.diag([0.25, 0.25, 0.25]) @ Rc.T @ fh[:3] -  Rc.T @ dp[:3]
        # diffFl =optas.diag([0.00, 0.00, 0.25]) @ Rc.T @ fh[:3] -  Rc.T @ dp[:3]
        # diffFl = optas.diag([1.0, 1.0, 0.0]) @ (p- pg_ee- optas.diag([Bx, By, 0.0]) @ cvf ) +diffFl
        diffp = optas.diag([1.0, 1.0, 0.0]) @ (pg_ee[:3]-p) 
        diffFl = optas.diag([0.00, 0.00, 0.1]) @ (-Rc.T @ fh[:3] +  Rc.T @ dp[:3])
        


        rvf = Rc.T @ fh[3:]
        diffFR =  optas.diag([0.1, 0.1, 0.1])@ rvf - Rc.T @dp[3:]
        # diffFr = nf @ nf.T @ (fh[3:] - optas.diag([1e-3, 1e-3, 1e-3]) @ Rc.T @dq[3:])

        W_p = optas.diag([1e3, 1e3, 1e3])
        builder.add_cost_term("match_p", diffp.T @ W_p @ diffp)

        # builder.add_leq_inequality_constraint(
        #     "p_con", diffp.T @ diffp, 0.0000001
        # )

        W_f = optas.diag([1e3, 1e3, 1e3])
        builder.add_cost_term("match_f", diffFl.T @ W_f @ diffFl)

        # W_fr = optas.diag([1e1, 1e1, 1e1])
        # builder.add_cost_term("match_fr", diffFR.T @ W_fr @ diffFR)
        
        w_dq = 0.1
        builder.add_cost_term("min_dq", w_dq * optas.sumsqr(dq))

        # w_ori = 1e1
        # builder.add_cost_term("match_r", w_ori * optas.sumsqr(diffR - I3()))

 
        builder.add_leq_inequality_constraint(
            "dq1", dq[0] * dq[0], 1e2
        )
        builder.add_leq_inequality_constraint(
            "dq2", dq[1] * dq[1], 1e2
        )
        builder.add_leq_inequality_constraint(
            "dq3", dq[2] * dq[2], 1e2
        )
        builder.add_leq_inequality_constraint(
            "dq4", dq[3] * dq[3], 1e2
        )
        builder.add_leq_inequality_constraint(
            "dq5", dq[4] * dq[4], 1e2
        )
        builder.add_leq_inequality_constraint(
            "dq6", dq[5] * dq[5], 1e2
        )
        builder.add_leq_inequality_constraint(
            "dq7", dq[6] * dq[6], 1e2
        )
        # builder.add_leq_inequality_constraint(
        #     "eff_z", diffp[2] * diffp[2], 1e-8

        
        # )

        Om = skew(dp[3:]) 
        
        # print("eff_goal size = {0}".format(eff_goal.size()))
        # RF = (Om * dt + I3()) @ Rc  
        RF = (Om * dt + I3()) @ Rc
        zF = RF[:3,2]
        #TODO
        axis_align_update = Rg[:,2]
        builder.add_cost_term("eff_axis_align", 1e2 * optas.sumsqr(zF - axis_align_update))

        self.distance = optas.Function('distance', [qc, pg], [pg_ee])
        self.rg = optas.Function('rg', [qc, pg], [axis_align_update])
        # self.zf = optas.Function('zf', [qc, pg], [zF])

        optimization = builder.build()
        self.solver = optas.CasADiSolver(optimization).setup("sqpmethod")
        # self.solver = optas.ScipyMinimizeSolver(optimization).setup('SLSQP')
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
        f_ext_raw = copy.deepcopy(f_ext)
        # print("f_ext1 = {0}".format(f_ext))
        # TODO 是否可以考虑在这里增加一个静态偏执。 用于数据分析。
        f_ext = np.where(
            abs(f_ext) > self.f_threshold_,
            self.dx_gain_ @ np.sign(f_ext) * (abs(f_ext) - self.f_threshold_),
            0.0
        )

        # print("self.distance = {0}".format(self.distance(q,pg)))
        print("self.rg = {0}".format(self.rg(q,pg)))
        # print("self.zf = {0}".format(self.zf(q,pg)))
        


        # print("self.dx_gain_ = {0}".format(self.dx_gain_))
        # print("f_ext2 = {0}".format(f_ext))

        dq = self.compute_target_velocity(q, pg, f_ext)
        # print("v = {0}".format(jacobian @ dq))
        # dq = self.dq_gain_ @ jacobian_inv @ f_ext
        self.dq_ = (1.0 - self.smooth_) * self.dq_ + self.smooth_ * dq
        # self.dq_ = dq
        return self.dq_, f_ext_raw

# class DockingController(TrackingController):
#     def __init__(self,
#     urdf_string: str,
#     end_link_name: str = "lbr_link_ee",
#     root_link_name: str = "lbr_link_0",
#     f_threshold: np.ndarray = np.array([2.0, 2.0, 2.0, 0.5, 0.5, 0.5]),
#     dx_gain: np.ndarray = np.array([1.0, 1.0, 1.0, 20.0, 40.0, 60.0]),
#     smooth: float = 0.01,)->None:


class AdmittanceControlNode(Node):
    def __init__(self, node_name="admittance_control_node") -> None:
        super().__init__(node_name=node_name)

        # parameters
        self.declare_parameter("model", "med7dock")
        self.declare_parameter("end_link_name", "link_shaft")
        self.declare_parameter("root_link_name", "lbr_link_0")
        self.declare_parameter("buffer_len", 40)

        self.model_ = str(self.get_parameter("model").value)
        self.end_link_name_ = str(self.get_parameter("end_link_name").value)
        self.root_link_name_ = str(self.get_parameter("root_link_name").value)

        # controller
        path = os.path.join(
            get_package_share_directory("med7_dock_description"),
            "urdf",
            f"{self.model_}.urdf.xacro",
        )
        self.urdf_string_ = xacro.process(path)

        self.chain_ = kinpy.build_serial_chain_from_urdf(
            data=self.urdf_string_, end_link_name=self.end_link_name_, root_link_name=self.root_link_name_
        )


        self.robot = optas.RobotModel(
            xacro_filename=path,
            time_derivs=[1],  # i.e. joint velocity
        )
        Nb, xyzs, rpys, axes = getJointParametersfromURDF(self.robot)
        self.dynamics_ = RNEA_function(Nb,1,rpys,xyzs,axes)
        self.Ymat, self.PIvector = DynamicLinearlization(self.dynamics_,Nb)

        Pb, Pd, Kd =find_dyn_parm_deps(7,80,self.Ymat)
        K = Pb.T +Kd @Pd.T

        self.Pb = Pb
        self.K = K

        path_pos = os.path.join(
            get_package_share_directory("gravity_compensation"),
            "test",
            "DynamicParameters.csv",
        )
        self.params = ExtractFromParamsCsv(path_pos)

        print("self.params = {0}".format(self.params))

        # The controller based on optas
        self.controller_optas = TrackingController(
            urdf_string=path,
            end_link_name=self.end_link_name_,
            root_link_name=self.root_link_name_,
        )

        self.lbr_command_ = LBRCommand()
        self.lbr_state_ = None

        # publishers and subscribers
        self.lbr_state_sub_ = self.create_subscription(
            LBRState, "/lbr_state", self.lbr_state_cb_,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                deadline=Duration(nanoseconds=10 * 1e6),  # 10 milliseconds
            )
        )

        self.cam_state_sub_ = self.create_subscription(
            Pose, 'object_pose_topic',self.cam_state_cb_, qos.qos_profile_system_default)

        self.lbr_command_pub_ = self.create_publisher(
            LBRCommand, "/lbr_command",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                deadline=Duration(nanoseconds=10 * 1e6),  # 10 milliseconds
            ),
        )

        self.timer_ = self.create_timer(
            0.01, self.on_timer_
        )

        self.init_ = True
        self.command_init_ = False
        self.goal_pose_ = None
        self.qd = np.array([0.0]*7)
        self.qd_last = np.array([0.0]*7)
        self.pa_size = Pb.shape[1]

        self.T_bt = np.zeros([4,4])
        self.T_bt[3,3] = 1.0
        self.tau_ext_ground = None

        # pg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def on_timer_(self):
        if not self.lbr_state_:
            return
        q = np.array(self.lbr_state_.measured_joint_position.tolist())
        tau = np.array(self.lbr_state_.measured_torque.tolist())

        qdd = (self.qd-self.qd_last)/0.01
        current_pose = self.chain_.forward_kinematics(q)
        T_be = from_pq2T44(current_pose.pos,current_pose.rot)

        if(self.init_):
            self.pg = self.chain_.forward_kinematics(q)
            print(self.pg)
            self.init_ = False
            self.q_last = copy.deepcopy(q)

            # jacobian = self.chain_.jacobian(q)
            # jacobian_inv = np.linalg.pinv(jacobian, rcond=0.05)
            # f_ext_base = jacobian_inv.T @ tau_ext
            self.tau_ext_ground = tau - (self.Ymat(q.tolist(),
                                   self.qd.tolist(),
                                   qdd.tolist())@self.Pb @  self.params[:self.pa_size] + 
                np.diag(np.sign(self.qd)) @ self.params[self.pa_size:self.pa_size+7]+ 
                np.diag(qdd) @ self.params[self.pa_size+7:])

            return
        else:
            pass
        pg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pg[:3] = self.pg.pos
        pg[3:] = self.pg.rot

        # pose_bt = pg

        # print("self.goal_pose_ = {0}".format(self.goal_pose_))

        
        # print("The error = {0}".format(pg[:3]-current_pose.pos))

        # ee 贴反了
        if(self.goal_pose_ is not None):
            T_et = from_msg2T44(self.goal_pose_)
            self.T_bt = T_be @ T_et
        
            print("self.T_be z = {0}".format(T_be[:,2]))
            print("self.T_et z = {0}".format(T_et[:,2]))
        
        print("self.T_bt z = {0}".format(self.T_bt[:,2]))

        pose_bt = from_T442pose(self.T_bt)
        csv_save("/home/thy/ros2_ws/pose_bt.csv", pose_bt)
        csv_save("/home/thy/ros2_ws/pose_br.csv", current_pose.pos)
        
        # dq, f_ext = self.controller_(q, tau_ext)
        
        # print("pose_bt",pose_bt)
        R_be = T_be[:3,:3]
        tau_ext = tau - (self.Ymat(q.tolist(),
                                   self.qd.tolist(),
                                   qdd.tolist())@self.Pb @  self.params[:self.pa_size] + 
                np.diag(np.sign(self.qd)) @ self.params[self.pa_size:self.pa_size+7]+ 
                np.diag(qdd) @ self.params[self.pa_size+7:]) - self.tau_ext_ground

        # TODO
        # Target position should be given according to vision. A static one can be given firstly
        # Param: pg ->
        # pose_bt[:3] = 0.05*pose_bt[:3] + 0.95 *current_pose.pos
        # print("tau = {0}".format(tau))
        # print("self.qd = {0}".format(self.qd))
        # print("tau_ext = {0}".format(tau_ext.toarray().flatten().shape))
        # print("tau = {0}".format(tau.shape))
        dq, f_ext = self.controller_optas(q, pose_bt ,tau_ext.toarray().flatten())

        self.jacobian_ = np.array(self.chain_.jacobian(q))

        self.jacobian_inv_ = np.linalg.pinv(self.jacobian_, rcond=0.05)

        # print(self.jacobian_)
        temp_r = np.array([0.0]*6)
        temp_r[:3] = R_be.T @ (pose_bt[:3]- current_pose.pos - np.array([0.0, 0.0, 0.2]))
        print("f_ext = {0}".format(f_ext))
        print("pose_bt = {0}".format(pose_bt))
        print("current_pose.pos = {0}".format(current_pose.pos))

        # temp_r[:3] = np.array([0.0, 0.1, 0.0])

        ## clasic controller 
        # dq = 10.03 * self.jacobian_inv_@ (temp_r)

        
        # TrackingController()
        self.qd_last = copy.deepcopy(self.qd)
        self.qd = q-self.q_last
        self.q_last = copy.deepcopy(q)
        # command
        self.lbr_command_.joint_position = (
            np.array(self.lbr_state_.measured_joint_position.tolist())
            + self.lbr_state_.sample_time * dq * 1.0
        ).data
        # print("self.lbr_state_.sample_time * dq * 1.0 = {0}",format(self.lbr_state_.sample_time * dq * 1.0))
        # print("goal = {0}",format(self.goal_pose_))

        self.command_init_ = True

    def lbr_state_cb_(self, msg: LBRState) -> None:
        self.lbr_state_ = msg

        if self.command_init_:
            self.lbr_command_pub_.publish(self.lbr_command_)

    def cam_state_cb_(self, msg:Pose) ->None:
        self.goal_pose_ = msg
        # print()

def main(args=None):
    rclpy.init(args=args)
    admittance_control_node = AdmittanceControlNode()
    rclpy.spin(admittance_control_node)
    # executor = MultiThreadedExecutor()
    # executor.add_node(admittance_control_node)
    # executor.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
