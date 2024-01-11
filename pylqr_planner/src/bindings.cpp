// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <eigen3/Eigen/Dense>
#include <memory>

#include <ilqr_planner/sim/2DRobot.h>
#include <ilqr_planner/sim/KDLRobot.h>
#include <ilqr_planner/sim/SimulationInterface.h>
#include <ilqr_planner/sim/TransformedSimulationInterface.h>

#include <ilqr_planner/system/JointSpacePlannerSys.h>
#include <ilqr_planner/system/JointSpaceTimePlannerSys.h>
#include <ilqr_planner/system/PosOrnPlannerSys.h>
#include <ilqr_planner/system/PosOrnTimePlannerSys.h>
#include <ilqr_planner/system/SequentialSystem.h>
#include <ilqr_planner/system/System.h>

#include <ilqr_planner/system/AngularKeypoint.h>
#include <ilqr_planner/system/AngularTimeKeypoint.h>
#include <ilqr_planner/system/Keypoint.h>
#include <ilqr_planner/system/PosOrnKeypoint.h>
#include <ilqr_planner/system/PosOrnKeypointDistFunct.h>
#include <ilqr_planner/system/SpacetimeKeypoint.h>

#include <ilqr_planner/solver/AL-ILQR.h>
#include <ilqr_planner/solver/BatchILQR.h>
#include <ilqr_planner/solver/BatchILQRCP.h>
#include <ilqr_planner/solver/ILQRRecursive.h>
#include <ilqr_planner/solver/lqt.h>

#include <ilqr_planner/utils/CallbackMessage.h>
#include <ilqr_planner/utils/primitives.h>
#include <ilqr_planner/utils/sd.h>

#include "PythonCallbackMessage.h"

namespace py = pybind11;
using namespace ilqr_planner;

PYBIND11_MODULE(PyLQR, m) {
    m.doc() = R"moddoc(
        PyLQR module documentation
        --------------------------

        .. currentmodule:: PyLQR

        .. autosummary::
            :toctree: _generate

            sim
            system
            solver
            utils

    )moddoc";

    py::module m_sim = m.def_submodule("sim");

    m_sim.doc() = R"moddoc(
        sim module documentation
        ------------------------

        .. current module:: sim

        .. autoclass:: SimulationInterface
            :members:

        .. autoclass:: Robot2D
            :members:

        .. autoclass:: KDLRobot
            :members:

    )moddoc";

    // Wrapping simulator part
    py::class_<sim::SimulationInterface, std::shared_ptr<sim::SimulationInterface>>(m_sim, "SimulationInterface")
        .def("update_kinematics", &sim::SimulationInterface::updateKinematics, R"methdoc(
            Forward kinematics function, from the current configuration q,dq compute x,dx,w,ornQuat,J,Jp
        )methdoc")

        // Jacobian related functions
        .def("Jt", &sim::SimulationInterface::Jt, "Return the translational Jacobian")
        .def("Jr", &sim::SimulationInterface::Jr, "Return the rotational Jacobian")
        .def("Jtp", &sim::SimulationInterface::Jtp, "Return the time-derivative of the translational Jacobian")
        .def("Jrp", &sim::SimulationInterface::Jrp, "Return the time-derivative of the rotational Jacobian")
        .def("J", &sim::SimulationInterface::J, "Return the full Jacobian")
        .def("Jp", &sim::SimulationInterface::Jp, "Return the time-derivative of the full Jacobian")

        // Robot state
        .def("get_ee_pos", &sim::SimulationInterface::getEEPosition, "Return the end-effector position")
        .def("get_ee_orn", &sim::SimulationInterface::getEEOrnQuat, "Return the end-effector orientation in quaternion (w,x,y,z)")
        .def("get_ee_vel", &sim::SimulationInterface::getEEVelocity, "Return the end-effector cartesian velocity")
        .def("get_ee_ang_vel", &sim::SimulationInterface::getEEAngVel, "Return the end-effector angular velocity")
        .def("get_ee_ang_vel_quat", &sim::SimulationInterface::getEEAngVelQuat, "Return the end-effector angular velocity in quaternion")
        .def("get_q", &sim::SimulationInterface::getJointsPos, "Return the joint positions")
        .def("get_dq", &sim::SimulationInterface::getJointsVel, "Return the joint velocities")
        .def("get_time", &sim::SimulationInterface::getTime, "Return the current time of the system")
        .def("set_time", &sim::SimulationInterface::setTime, py::arg("time"), "Set the current time of the system")
        // Helpers
        .def("dquat_to_w_jac", &sim::SimulationInterface::dQuatToDxJac, py::arg("quat"), R"methdoc(
            Build the quaternion to angular velocity jacobian.

            :param quat: Quaternion to build the jacobian on
            :type quat: numpy array

        )methdoc")
        .def("set_conf", &sim::SimulationInterface::setConfiguration, py::arg("q"), py::arg("dq"), py::arg("reset_time"), R"methdoc(
            Update current joint state configuration of a robot with a new one

            :param q: Joint positions
            :type q: numpy array

            :param dq: Joint velocities
            :type dq: numpy array

            :param reset_time: If true, time will be set to zero
            :type reset_time: bool

        )methdoc")

        // Robot control
        .def("send_acc", &sim::SimulationInterface::sendAcc, py::arg("dt"), py::arg("ddq"), py::arg("updateKin"), R"methdoc(
            Send a joint acceleration command to the robot

            :param dt: simulation dt
            :type dt: double

            :param ddq: Acceleration command
            :type ddq: numpy array

            :param updateKin: If true update_kinematics function is called
            :type updateKin: boolean
            )methdoc")
        .def("send_vel", &sim::SimulationInterface::sendVel, py::arg("dt"), py::arg("dq"), py::arg("updateKin"), R"methdoc(
            Send a joint velocity command to the robot

            :param dt: simulation dt
            :type dt: double

            :param dq: Velocity command
            :type dq: numpy array

            :param updateKin: If true update_kinematics function is called
            :type updateKin: boolean
            )methdoc")
        .doc() = R"classdoc(
            Abstraction of a robot, this class is an abstract class
            )classdoc";

    py::class_<sim::KDLRobot, sim::SimulationInterface, std::shared_ptr<sim::KDLRobot>>(m_sim, "KDLRobot")
        .def(py::init<const std::string&, const std::string&, const std::string&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const bool&>(),
             py::arg("urdf"), py::arg("base_frame"), py::arg("tip_frame"), py::arg("q"), py::arg("dq"), py::arg("transform_rpy"), py::arg("transform_xyz"), py::arg("is_path"))
        .def(py::init<const std::string&, const std::string&, const std::string&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&>(), py::arg("urdf"),
             py::arg("base_frame"), py::arg("tip_frame"), py::arg("q"), py::arg("dq"), py::arg("transform_rpy"), py::arg("transform_xyz"))
        .def(py::init<const std::string&, const std::string&, const std::string&, const Eigen::VectorXd&, const Eigen::VectorXd&>(), py::arg("urdf"), py::arg("base_frame"), py::arg("tip_frame"),
             py::arg("q"), py::arg("dq"))
        .doc() = R"classdoc(
            A class abstracting a robot with the orocos-kdl library.

            :param urdf: Path to the urdf file
            :type urdf: string

            :param baseFrame: Name of the base frame in the urdf
            :type baseFrame: string

            :param tipFrame: Name of the tip frame in the urdf
            :type tipFrame: string

            :param q: Initial joint positions
            :type q: numpy array

            :param dq: Initial joint velocities
            :type dq: numpy array

            :param transform_rpy: Custom end-effector transform (Optional)(roll,pitch,yaw)
            :type transform_rpy: numpy array

            :param transform_xyz: Custom end-effector transform (Optional)(x,y,z)
            :type transform_xyz: numpy array

            )classdoc";

    py::class_<sim::Robot2D, sim::SimulationInterface, std::shared_ptr<sim::Robot2D>>(m_sim, "Robot2D")
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&>(), py::arg("lengths"), py::arg("default_q"))
        .def("fkine", static_cast<Eigen::VectorXd (sim::Robot2D::*)()>(&sim::Robot2D::fkine), "compute the forward kinematics for the current configuration")
        .def("fkine", static_cast<Eigen::VectorXd (sim::Robot2D::*)(const Eigen::VectorXd&)>(&sim::Robot2D::fkine), py::arg("q"), "compute the forward kinematics for a given joint configuration")
        .doc() = R"classdoc(
                A class abstracting a 2D planar robot.

                :param lengths: link lengths
                :type lengths: numpy array

                :param default_q: initial joint positions
                :type default_q: numpy array
            )classdoc";

    py::class_<sim::TransformedSimulationInterface, sim::SimulationInterface, std::shared_ptr<sim::TransformedSimulationInterface>>(m_sim, "TransformedSimulationInterface")
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const Eigen::MatrixXd&>(), py::arg("r"), py::arg("T"))
        .doc() = R"classdoc(
                A class Applying a special transformation to a SimulationInterface object.

                :param r: Object to apply the transform on
                :type r: SimulationInterface

                :param T: Custom transform
                :type T: numpy 2D-array
            )classdoc";

    // Wrapping system part
    py::module m_sys = m.def_submodule("system");

    m_sys.doc() = R"moddoc(
        system module documentation
        ---------------------------

        .. current module:: system

        .. autoclass:: System
            :members:

        .. autoclass:: JointSpacePlannerSys
            :members:

        .. autoclass:: JointSpaceTimePlannerSys
            :members:

        .. autoclass:: PosOrnPlannerSys
            :members:

        .. autoclass:: PosOrnTimePlannerSys
            :members:

        .. autoclass:: Keypoint
            :members:

        .. autoclass:: PosOrnKeypoint
            :members:

        .. autoclass:: PosOrnKeypointDistFunct
            :members:

        .. autoclass:: SpacetimeKeypoint
            :members:

        .. autoclass:: AngularKeypoint
            :members:

        .. autoclass:: AngularTimeKeypoint
            :members:

    )moddoc";

    py::class_<sys::Keypoint, std::shared_ptr<sys::Keypoint>>(m_sys, "Keypoint")
        .def("diff", &sys::Keypoint::diff, py::arg("state"), R"methdoc(
            Compute the difference between keypoint's state and given state.

            :param state: State to compute the difference with.
            :type state: numpy array
        )methdoc")
        .def("get_state", &sys::Keypoint::getState, "Return the vector representation of the keypoint")
        .def("get_precision", &sys::Keypoint::getPrecision, "Return the keypoint's precision matrix")
        .def("get_timestep", &sys::Keypoint::getTimestep, "Return the occurence timestep of the keypoint");

    py::class_<sys::PosOrnKeypoint, sys::Keypoint, std::shared_ptr<sys::PosOrnKeypoint>>(m_sys, "PosOrnKeypoint")
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::MatrixXd&, const int&>(), py::arg("position"), py::arg("orientation"), py::arg("precision"), py::arg("timestep"))
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::MatrixXd&, const int&>(), py::arg("position"), py::arg("dposition"),
             py::arg("orientation"), py::arg("dorientation"), py::arg("precision"), py::arg("timestep"))
        .def("get_position", &sys::PosOrnKeypoint::getPosition, "Return keypoint's position")
        .def("get_orientation", &sys::PosOrnKeypoint::getOrientation, "Return keypoint's orientation")
        .doc() = R"classdoc(
            This class inherits from Keypoint and represent a keypoint expressend in position and in orientation

            :param position: Keypoint's position
            :type position: numpy array

            :param dposition: Keypoint's linear velocity, if specified  dorientation needs to be specified too.
            :type dposition: numpy array

            :param orientation: Keypoint'orientation in quaternion (w,x,y,z)
            :type orientation: numpy array

            :param dorientation: Keypoint's angular velocity expressed in quaternion velocity (dw,dx,dy,dz), if specified  dorientation needs to be specified too.
            :type dorientation: numpy array

            :param precision: Keypoint's precision matrix
            :type precision: numpy 2D array

            :param timestep: Keypoint's discrete time of occurence
            :type timestep:  int

        )classdoc";

    py::class_<sys::PosOrnKeypointDistFunct, sys::Keypoint, std::shared_ptr<sys::PosOrnKeypointDistFunct>>(m_sys, "PosOrnKeypointDistFunct")
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::MatrixXd&, const double&, const Eigen::Vector3d&, const int&>(), py::arg("position"), py::arg("orientation"),
             py::arg("precision"), py::arg("pos_thresh"), py::arg("orn_thresh"), py::arg("timestep"))
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::MatrixXd&, const double&, const Eigen::Vector3d&, const int&>(),
             py::arg("position"), py::arg("dposition"), py::arg("orientation"), py::arg("dorientation"), py::arg("precision"), py::arg("pos_thresh"), py::arg("orn_thresh"), py::arg("timestep"))
        .doc() = R"classdoc(
            This class inherits from Keypoint and represent a keypoint expressend in position and in orientation

            :param position: Keypoint's position
            :type position: numpy array

            :param dposition: Keypoint's linear velocity, if specified  dorientation needs to be specified too.
            :type dposition: numpy array

            :param orientation: Keypoint'orientation in quaternion (w,x,y,z)
            :type orientation: numpy array

            :param dorientation: Keypoint's angular velocity expressed in quaternion velocity (dw,dx,dy,dz), if specified  dorientation needs to be specified too.
            :type dorientation: numpy array

            :param precision: Keypoint's precision matrix
            :type precision: numpy 2D array

            :param pos_thresh: Position sphere compliance radius.
            :type pos_thresh: double

            :param orn_thresh: Orientation bonding box compliance expressed in the tangent space.
            :type orn_thresh: numpy array

            :param timestep: Keypoint's discrete time of occurence
            :type timestep:  int

        )classdoc";

    py::class_<sys::SpacetimeKeypoint, sys::PosOrnKeypoint, sys::Keypoint, std::shared_ptr<sys::SpacetimeKeypoint>>(m_sys, "SpacetimeKeypoint")
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::MatrixXd&, const double&, const int&>(), py::arg("position"), py::arg("orientation"), py::arg("precision"),
             py::arg("continuous_time"), py::arg("timestep"))
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::MatrixXd&, const double&, const int&>(), py::arg("position"),
             py::arg("dposition"), py::arg("orientation"), py::arg("dorientation"), py::arg("precision"), py::arg("continuous_time"), py::arg("timestep"))
        .def("get_continuous_time", &sys::SpacetimeKeypoint::getContinuousTime, "Return keypoint's continuous time of occurence")
        .doc() = R"classdoc(
            This class inherits from PosOrnKeypoint and represent a keypoint expressend in position,orientation and time

            :param position: Keypoint's position
            :type position: numpy array

            :param dposition: Keypoint's linear velocity, if specified  dorientation needs to be specified too.
            :type dposition: numpy array

            :param orientation: Keypoint'orientation in quaternion (w,x,y,z)
            :type orientation: numpy array

            :param dorientation: Keypoint's angular velocity expressed in quaternion velocity (dw,dx,dy,dz), if specified  dorientation needs to be specified too.
            :type dorientation: numpy array

            :param precision: Keypoint's precision matrix
            :type precision: numpy 2D array

            :param continuous_time: Keypoint's continuous time of occurence
            :type continuous_time: double

            :param timestep: Keypoint's discrete time of occurence
            :type timestep:  int

        )classdoc";

    py::class_<sys::AngularKeypoint, sys::Keypoint, std::shared_ptr<sys::AngularKeypoint>>(m_sys, "AngularKeypoint")
        .def(py::init<const Eigen::VectorXd&, const Eigen::MatrixXd&, const int&>(), py::arg("position"), py::arg("precision"), py::arg("timestep"))
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::MatrixXd&, const int&>(), py::arg("position"), py::arg("dposition"), py::arg("precision"), py::arg("timestep"))
        .def("get_position", &sys::AngularKeypoint::getPosition, "Return keypoint's position")
        .doc() = R"classdoc(
            This class inherits from Keypoint and represent a keypoint expressend in joint space
            :param position: Keypoint's position
            :type position: numpy array

            :param dposition: Keypoint's velocity,optional
            :type dposition: numpy array

            :param precision: Keypoint's precision matrix
            :type precision: numpy 2D array

            :param timestep: Keypoint's discrete time of occurence
            :type timestep:  int

        )classdoc";

    py::class_<sys::AngularTimeKeypoint, sys::AngularKeypoint, sys::Keypoint, std::shared_ptr<sys::AngularTimeKeypoint>>(m_sys, "AngularTimeKeypoint")
        .def(py::init<const Eigen::VectorXd&, const Eigen::MatrixXd&, const double&, const int&>(), py::arg("position"), py::arg("precision"), py::arg("continuous_time"), py::arg("timestep"))
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::MatrixXd&, const double&, const int&>(), py::arg("position"), py::arg("dposition"), py::arg("precision"),
             py::arg("continuous_time"), py::arg("timestep"))
        .def("get_continuous_time", &sys::AngularTimeKeypoint::getContinuousTime, "Return keypoint's continuous time of occurence")
        .doc() = R"classdoc(
            This class inherits from AngularKeypoint and represent a keypoint expressend in position and time

            :param position: Keypoint's position
            :type position: numpy array

            :param dposition: Keypoint's velocity, optional.
            :type dposition: numpy array

            :param precision: Keypoint's precision matrix
            :type precision: numpy 2D array

            :param continuous_time: Keypoint's continuous time of occurence
            :type continuous_time: double

            :param timestep: Keypoint's discrete time of occurence
            :type timestep:  int

        )classdoc";

    py::class_<sys::System, std::shared_ptr<sys::System>>(m_sys, "System")
        .def("forward_pass", &sys::System::forwardPass, py::arg("xk"), py::arg("uk"), py::arg("k"), R"methdoc(
            Given the current state of the SimulationInterface object, apply the control command uk.

            :param xk: Current state of the SimulationInterface object
            :type xk: nummpy array

            :param uk: Command to apply on the SimulationInterface object
            :type uk: numpy array

            :param k: Time step of the command
            :type k: int

            :return: A tuple containing xkp1 (next state) , fxkp1 (next f(x) state), A (dxkp1/dxk), B (dxkp1/duk), J (d(fxkp1)/dxkp1)
            :rtype: tuple of a numpy arrays
        )methdoc")
        .def("diff", &sys::System::diff, py::arg("actual_state"), py::arg("k"), R"methdoc(
            Compute the difference between two f_x states

            :param actual_state: Current state
            :type actual_state: numpy array

            :param k: time step of the difference
            :type k: int

        )methdoc")
        .def("diff_batch", &sys::System::diffBatch, py::arg("x"), "call diff for a batch formulation (basically call diff horizon times")
        .def("forward_pass_with_limits", &sys::System::forwardPassWithLimits, py::arg("xk"), py::arg("uk"), py::arg("k"), R"methdoc(
            Perform the forward pass with respect to the joint & command limits

            :return: A tuple containing xkp1 (next state) , fxkp1 (next f(x) state), q (joint state violation), A (dxkp1/dxk), B (dxkp1/duk), J (d(fxkp1)/dxkp1), L (joint state violation penalty matrix)
            :rtype: tuple of a numpy arrays
        )methdoc")
        .def("forward_pass_batch", &sys::System::fpBatch, py::arg("u"), "perform the forward pass in batch form")
        .def("cost_F", &sys::System::cost_F, py::arg("xk"), "Final cost for the state <fxk>")
        .def("cost_F_x", &sys::System::cost_F_x, py::arg("xk"), R"methdoc(
            Gradient of the final cost with respect to x

            :param fxk: f(xk)
            :type fxk: numpy array

            :param Jk: Jacobian between f(xk) and xk
            :type Jk: numpy array

            )methdoc")
        .def("cost_F_xx", &sys::System::cost_F_xx, py::arg("xk"), "Hessian of the final cost with respect to x")
        .def("cost", &sys::System::cost, py::arg("xk"), py::arg("uk"), py::arg("k"), R"methdoc(
            Cost to go at time step <k>

            :param fxk: f(xk)
            :type fxk: numpy array

            :param uk: control command at <k>
            :type uk: numpy array

            :param k: timestep
            :type k: int

            :return: Cost to go at time step k
        )methdoc")
        .def("cost_x", &sys::System::cost_x, py::arg("xk"), py::arg("uk"), py::arg("k"), "Gradient of cost to go with respect to x")
        .def("cost_xx", &sys::System::cost_xx, py::arg("xk"), py::arg("uk"), py::arg("k"), "Hessian of the cost to go with respect to x")
        .def("cost_u", &sys::System::cost_u, py::arg("xk"), py::arg("uk"), py::arg("k"), "Gradient of the cost to go with respect to u")
        .def("cost_uu", &sys::System::cost_uu, py::arg("xk"), py::arg("uk"), py::arg("k"), "Hessian of the cost to go with respect to u")
        .def("cost_xu", &sys::System::cost_xu, py::arg("xk"), py::arg("uk"), py::arg("k"), "will be zero in most use cases")
        .def("cost_ux", &sys::System::cost_ux, py::arg("xk"), py::arg("uk"), py::arg("k"), "will be zero in most use cases")

        .def("get_mu_vector", &sys::System::getMuVector, "Return target states vectorized of shape (nb_state_var * horizon,1)")
        .def("get_Q_matrix", &sys::System::getQMatrix, "Return the Q matrix of shape (nb_Q_var * horizon, nb_Q_var * horizon)")

        .def("get_nb_state_var", &sys::System::getNbStateVar, "Return the number of state signals")
        .def("get_nb_ctrl_var", &sys::System::getNbCtrlVar, "Return the number of control signals")
        .def("get_nb_target_var", &sys::System::getNbTargetVar, "Return the size of the target space")
        .def("get_horizon", &sys::System::getHorizon, "Return the horizon of the system")

        .def("get_state", &sys::System::getState, "return the current state at the joint level")
        .def("get_fx_jac", static_cast<std::tuple<Eigen::VectorXd, Eigen::MatrixXd> (sys::System::*)()>(&sys::System::getFxJac), R"methdoc(
            return the current f(x) with the Jacobian with respect to the current state
            )methdoc")
        .def("get_fx_jac", static_cast<std::tuple<Eigen::VectorXd, Eigen::MatrixXd> (sys::System::*)(Eigen::VectorXd)>(&sys::System::getFxJac), R"methdoc(
            return the current f(x) with the Jacobian with respect to the given state

            :param xk: Given state
            :type xk: numpy array

            )methdoc",
             py::arg("xk"))
        .def("get_init_state", &sys::System::getInitState, "return initial state at the joint level")
        .def("get_init_fx_state", &sys::System::getInitFoXState, "return initial f(x)")

        .def("reset", &sys::System::reset, "Reset the simulation interface to the initial state");

    py::class_<sys::SequentialSystem, sys::System, std::shared_ptr<sys::SequentialSystem>>(m_sys, "SequentialSystem")
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::System>>&, const Eigen::VectorXd&, int, int>(), py::arg("r"), py::arg("systems"),
             py::arg("RtDiag"), py::arg("horizon"), py::arg("nbDeriv"))
        .doc() = R"classdoc(
            This class inherits from the sys::System class and provide a way to combine different systems together to optimize different functions.

            :param r: Simulation interface object
            :type r: SimulationInterface

            :param systems: List of systems to use
            :type systems: List<Systems>

            :param RtDiag: control penalty for each control signal
            :type RtDiag: numpy array

            :param horizon: Horizon of the problem
            :type horizon: int

            :param nbDeriv: Number of derivative of the linear system
            :type nbDeriv: int (1 or 2)

        )classdoc";

    py::class_<sys::JointSpacePlannerSys, sys::System, std::shared_ptr<sys::JointSpacePlannerSys>>(m_sys, "JointSpacePlannerSys")
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&,
                      const Eigen::VectorXd&, const Eigen::VectorXd&, int, int, double>(),
             py::arg("r"), py::arg("keypoints"), py::arg("RtDiag"), py::arg("qMax"), py::arg("qMin"), py::arg("dqMax"), py::arg("dqMin"), py::arg("horizon"), py::arg("nbDeriv"), py::arg("dt"))
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, int,
                      int, double>(),
             py::arg("r"), py::arg("keypoints"), py::arg("RtDiag"), py::arg("qMax"), py::arg("qMin"), py::arg("horizon"), py::arg("nbDeriv"), py::arg("dt"))
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, int, int, double>(), py::arg("r"),
             py::arg("keypoints"), py::arg("RtDiag"), py::arg("horizon"), py::arg("nbDeriv"), py::arg("dt"))
        .doc() = R"classdoc(
            This class inherits from the System class and provide the base system for a Joint space planner system

            :param r: Simulation interface object
            :type r: SimulationInterface

            :param keypoints: Keypoints that our system will track
            :type keypoints: list of PosOrnKeypoint object

            :param RtDiag: control penalty for each control signal
            :type RtDiag: numpy array

            :param qMax: Maximum allowed joint position, optional
            :type qMax: numpy array

            :param qMin: Minimum allowed joint position, optional
            :type qMin: numpy array

            :param dqMax: Maximum allowed joint velocity, optional
            :type dqMax: numpy array

            :param dqMin: Minimum allowed joint velocity, optional
            :type dqMin: numpy array

            :param horizon: Horizon of the problem
            :type horizon: int

            :param nbDeriv: Number of derivative of the linear system
            :type nbDeriv: int (1 or 2)

            :param dt: time step duration
            :type dt: double
        )classdoc";

    py::class_<sys::JointSpaceTimePlannerSys, sys::System, std::shared_ptr<sys::JointSpaceTimePlannerSys>>(m_sys, "JointSpaceTimePlannerSys")
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&,
                      const Eigen::VectorXd&, const Eigen::VectorXd&, int, int>(),
             py::arg("r"), py::arg("keypoints"), py::arg("RtDiag"), py::arg("qMax"), py::arg("qMin"), py::arg("dqMax"), py::arg("dqMin"), py::arg("horizon"), py::arg("nbDeriv"))
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, int,
                      int>(),
             py::arg("r"), py::arg("keypoints"), py::arg("RtDiag"), py::arg("qMax"), py::arg("qMin"), py::arg("horizon"), py::arg("nbDeriv"))
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, int, int>(), py::arg("r"), py::arg("keypoints"),
             py::arg("RtDiag"), py::arg("horizon"), py::arg("nbDeriv"))
        .doc() = R"classdoc(
            This class inherits from the System class and provide the base system for a joint space planner with time optimization system

            :param r: Simulation interface object
            :type r: SimulationInterface

            :param keypoints: Keypoints that the system will track
            :type keypoints: list of SpacetimeKeypoint

            :param RtDiag: control penalty for each control signal
            :type RtDiag: numpy array

            :param qMax: Maximum allowed joint position, optional
            :type qMax: numpy array

            :param qMin: Minimum allowed joint position, optional
            :type qMin: numpy array

            :param dqMax: Maximum allowed joint velocity, optional
            :type dqMax: numpy array

            :param dqMin: Minimum allowed joint velocity, optional
            :type dqMin: numpy array

            :param horizon: Horizon of the problem
            :type horizon: int

            :param nbDeriv: Number of derivative of the linear system
            :type nbDeriv: int (1 or 2)
        )classdoc";

    py::class_<sys::PosOrnPlannerSys, sys::System, std::shared_ptr<sys::PosOrnPlannerSys>>(m_sys, "PosOrnPlannerSys")
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&,
                      const Eigen::VectorXd&, const Eigen::VectorXd&, int, int, double>(),
             py::arg("r"), py::arg("keypoints"), py::arg("RtDiag"), py::arg("qMax"), py::arg("qMin"), py::arg("dqMax"), py::arg("dqMin"), py::arg("horizon"), py::arg("nbDeriv"), py::arg("dt"))
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, int,
                      int, double>(),
             py::arg("r"), py::arg("keypoints"), py::arg("RtDiag"), py::arg("qMax"), py::arg("qMin"), py::arg("horizon"), py::arg("nbDeriv"), py::arg("dt"))
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, int, int, double>(), py::arg("r"),
             py::arg("keypoints"), py::arg("RtDiag"), py::arg("horizon"), py::arg("nbDeriv"), py::arg("dt"))
        .doc() = R"classdoc(
            This class inherits from the System class and provide the base system for a Position and orientation planner system

            :param r: Simulation interface object
            :type r: SimulationInterface

            :param keypoints: Keypoints that our system will track
            :type keypoints: list of PosOrnKeypoint object

            :param RtDiag: control penalty for each control signal
            :type RtDiag: numpy array

            :param qMax: Maximum allowed joint position, optional
            :type qMax: numpy array

            :param qMin: Minimum allowed joint position, optional
            :type qMin: numpy array

            :param dqMax: Maximum allowed joint velocity, optional
            :type dqMax: numpy array

            :param dqMin: Minimum allowed joint velocity, optional
            :type dqMin: numpy array

            :param horizon: Horizon of the problem
            :type horizon: int

            :param nbDeriv: Number of derivative of the linear system
            :type nbDeriv: int (1 or 2)

            :param dt: time step duration
            :type dt: double
        )classdoc";

    py::class_<sys::PosOrnTimePlannerSys, sys::System, std::shared_ptr<sys::PosOrnTimePlannerSys>>(m_sys, "PosOrnTimePlannerSys")
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&,
                      const Eigen::VectorXd&, const Eigen::VectorXd&, int, int>(),
             py::arg("r"), py::arg("keypoints"), py::arg("RtDiag"), py::arg("qMax"), py::arg("qMin"), py::arg("dqMax"), py::arg("dqMin"), py::arg("horizon"), py::arg("nbDeriv"))
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, int,
                      int>(),
             py::arg("r"), py::arg("keypoints"), py::arg("RtDiag"), py::arg("qMax"), py::arg("qMin"), py::arg("horizon"), py::arg("nbDeriv"))
        .def(py::init<const std::shared_ptr<sim::SimulationInterface>&, const std::vector<std::shared_ptr<sys::Keypoint>>&, const Eigen::VectorXd&, int, int>(), py::arg("r"), py::arg("keypoints"),
             py::arg("RtDiag"), py::arg("horizon"), py::arg("nbDeriv"))
        .doc() = R"classdoc(
            This class inherits from the System class and provide the base system for a Position and orientation planner with time optimization system

            :param r: Simulation interface object
            :type r: SimulationInterface

            :param keypoints: Keypoints that the system will track
            :type keypoints: list of SpacetimeKeypoint

            :param RtDiag: control penalty for each control signal
            :type RtDiag: numpy array

            :param qMax: Maximum allowed joint position, optional
            :type qMax: numpy array

            :param qMin: Minimum allowed joint position, optional
            :type qMin: numpy array

            :param dqMax: Maximum allowed joint velocity, optional
            :type dqMax: numpy array

            :param dqMin: Minimum allowed joint velocity, optional
            :type dqMin: numpy array

            :param horizon: Horizon of the problem
            :type horizon: int

            :param nbDeriv: Number of derivative of the linear system
            :type nbDeriv: int (1 or 2)
        )classdoc";

    // Wrapping the solver part
    py::module m_sol = m.def_submodule("solver");

    m_sol.doc() = R"moddoc(
        solver module documentation
        ---------------------------

        .. current module:: solver

        .. autoclass:: Constraint
            :members:

        .. autoclass:: AL_ILQR
            :members:

        .. autoclass:: BatchILQRCP
            :members:

        .. autoclass:: BatchILQR
            :members:

        .. autoclass:: LQT
            :members:

        .. autoclass:: ILQRRecursive
            :members:

    )moddoc";

    py::class_<solver::Constraint>(m_sol, "Constraint").def(py::init<>()).def_readwrite("A", &solver::Constraint::A, "A matrix").def_readwrite("b", &solver::Constraint::b, "b vector").doc() =
        R"classdoc(
            This class is an helper to create a constraint under the form:

            .. math::
                \mathbf{A} \mathcal{S} \leq \mathbf{b}

        )classdoc";

    py::class_<solver::AL_ILQR>(m_sol, "AL_ILQR")
        .def(py::init<const std::shared_ptr<sys::System>&, const std::vector<solver::Constraint>&, const std::vector<Eigen::VectorXd>&>(), py::arg("s"), py::arg("inequality"), py::arg("initLambda"))
        .def("solve", &solver::AL_ILQR::solve, py::arg("U0"), py::arg("nb_iter"), py::arg("lag_update_step"), py::arg("penalty"), py::arg("scaling_factor"), py::arg("line_search"),
             py::arg("early_stop"), py::arg("cb"),
             R"methdoc(
            Solve the problem for the given number of iteration

            :param U0: List of initial control command for each timestep
            :type U0: list of vector

            :param nb_iter: Number of iteration to perform
            :type nb_iter: int

            :param lag_update_step: Lagrange multipliers update frequency.
            :type lag_update_step: int

            :param penalty: Weight of the constraints
            :type penalty: double

            :param scaling_factor: How the weight of constraints evolve (1->constant)
            :type scaling_factor: double

            :param line_search: If true, line search is performed at each iteration.
            :type line_search: bool

            :param early_stop: If true, stop optimization when cost is stagning
            :type early_stop: bool

            :param cb: Callback class to notify the user
            :type cb: CallBackMessage and its derivatives

        )methdoc")
        .doc() = R"classdoc(
            Iterative LQR with Augmented Lagrangian formulation

            :param s: Abstraction of the system that we want to optimize
            :type s: System

            :param inequality: List of inequality constraints for each timestep
            :type inequality: List of Constraint

            :param initLambda: list of the initial values for the multipliers
            :type initLambda: List of vector

        )classdoc";

    py::class_<solver::BatchILQR>(m_sol, "BatchILQR")
        .def(py::init<const std::shared_ptr<sys::System>&>(), py::arg("s"))
        .def(py::init<const std::shared_ptr<sys::System>&, const Eigen::MatrixXd&>(), py::arg("s"), py::arg("Q"))

        .def("solve", &solver::BatchILQR::solve, py::arg("nb_iter"), py::arg("u0"), py::arg("early_stop"), py::arg("cb"), R"metdoc(
            Solve the problem for <nb_iter> iterations, publish iteration callback inside <cb>

            :param nb_iter: Maximum number of iteration to perform.
            :type nb_iter: int

            :param u0: Intial control command
            :type u0: numpy array

            :param early_stop: If true, stop optimization when cost is stagning
            :type early_stop: bool

            :param cb: Callback class to notify the user
            :type cb: CallBackMessage and its derivatives
        )metdoc")
        .doc() = R"classdoc(
            Batch Iterative LQR

            :param s: Abstraction of the system that we want to optimize
            :type s: System

            :param Q: Precision matrix for the target states of shape (nbStateVar * horizon,nbStateVar*horizon), optional
            :type Q: numpy array

            )classdoc";

    py::class_<solver::BatchILQRCP>(m_sol, "BatchILQRCP")
        .def(py::init<const std::shared_ptr<sys::System>&, const Eigen::MatrixXd&, const Eigen::MatrixXd&>(), py::arg("s"), py::arg("Q"), py::arg("psi"))
        .def(py::init<const std::shared_ptr<sys::System>&, const Eigen::MatrixXd&>(), py::arg("s"), py::arg("psi"))
        .def("solve", &solver::BatchILQRCP::solve, py::arg("nb_iter"), py::arg("u0"), py::arg("early_stop"), py::arg("cb"), R"metdoc(
            Solve the problem for <nb_iter> iterations, publish iteration callback inside <cb>

            :param nb_iter: Maximum number of iteration to perform.
            :type nb_iter: int

            :param u0: Initical control command
            :type u0: numpy array

            :param early_stop: If true, stop optimization when cost is stagning
            :type early_stop: bool

            :param cb: Callback class to notify the user
            :type cb: CallBackMessage and its derivatives
        )metdoc")
        .doc() = R"classdoc(
            Batch Iterative LQR with the use of control primitives

            :param s: Abstraction of the system that we want to optimize
            :type s: System

            :param Q: Precision matrix for the target states of shape (nbStateVar * horizon,nbStateVar*horizon), optional
            :type Q: numpy array

            :param psi: Control primitives matrix, u = PSI * w
            :type psi: numpy array

            )classdoc";

    py::class_<solver::ILQRRecursive>(m_sol, "ILQRRecursive")
        .def(py::init<const std::shared_ptr<sys::System>&>(), py::arg("s"))
        .def("solve", &solver::ILQRRecursive::solve, py::arg("U0"), py::arg("nb_iter"), py::arg("line_search"), py::arg("early_stop"), py::arg("cb"), R"methdoc(
            Solve the iLQR problem, return X,f(X),U,K,k

            :param U0: Initial control command.
            :type U0: numpy array

            :param nb_iter: Number of iteration to perform
            :type nb_iter: int

            :param line_search: If true, line search is performed at each iteration.
            :type line_search: bool

            :param early_stop: If true, stop optimization when cost is stagning
            :type early_stop: bool

            :param cb: Callback class to notify the user
            :type cb: CallBackMessage and its derivatives
        )methdoc")
        .doc() = "Standard iterative LQR";

    py::class_<solver::LQT>(m_sol, "LQT")
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&, const std::vector<Eigen::MatrixXd>&, const Eigen::VectorXd&, float, int>())
        .def("solve_DP", &solver::LQT::solveDP)
        .def("solve_lin_al", &solver::LQT::solveLinAl)
        .def("get_nb_states", &solver::LQT::getNbStates)
        .def("get_predicted_states", &solver::LQT::getPredictedStates)
        .def("get_command", static_cast<Eigen::VectorXd (solver::LQT::*)(int)>(&solver::LQT::getCommand))
        .def("get_command", static_cast<Eigen::VectorXd (solver::LQT::*)(int, const Eigen::VectorXd&)>(&solver::LQT::getCommand));

    // Wrapping the utils part
    py::module m_ut = m.def_submodule("utils");

    m_ut.doc() = R"moddoc(
        utils module documentation
        --------------------------

        .. autosummary::
            :toctree: _generate

            Sd
            primitives
    )moddoc";

    py::class_<CallBackMessage>(m_ut, "CallBackMessage").doc() = "Abstract class";

    py::class_<PythonCallbackMessage, CallBackMessage>(m_ut, "PythonCallbackMessage").def(py::init<>()).doc() = R"classdoc(
            The purpose of this class is to notify Python code about the progression of the solver. This class will simply call a Python print at each iteration of the solver.
        )classdoc";

    py::module m_sd = m_ut.def_submodule("Sd");
    m_sd.doc() = "Contains all methods relative to the Sphere manifold";
    m_sd.def("logMap", &Sd::logMap, py::arg("base"), py::arg("y"), R"methdoc(
        Project a point on the Sd sphere <y> to the tangent space of <base>
        )methdoc");
    m_sd.def("expMap", &Sd::expMap, py::arg("base"), py::arg("u"), "Transform a point on the tangent space of <base> to the Sd sphere");
    m_sd.def("distance", &Sd::distance, py::arg("x"), py::arg("y"), "Distance of the geodesic between <x> and <y>");
    m_sd.def("transport", &Sd::transport, py::arg("v"), py::arg("base1"), py::arg("base2"), "Parallel transport of vector <v> relying on the tangent space of <base1> to the tangent space of <base2>");
    m_sd.def("dquat_to_w_jac", &Sd::dQuatToDxJac, py::arg("q"), "Quaternion to euler angles Jacobian");

    py::module m_prim = m_ut.def_submodule("primitives");
    m_prim.doc() = "Contains the functions to build the desired primitives";
    m_prim.def("build_psi_RBF", &buildPsiRBF, py::arg("dim"), py::arg("K"), "Radial Basis Function as primitives");
    m_prim.def("build_psi_bernstein", &buildPsiBernstein, py::arg("dim"), py::arg("K"), "Bernstein polynomial as primitives");
    m_prim.def("build_psi_unitstep", &buildPsiUnitstep, py::arg("dim"), py::arg("K"), "Unitstep functions as primitives");
    m_prim.def("build_psi_sawtooth", &buildPsiSawtooth, py::arg("dim"), py::arg("K"), "Sawtooth functions as primitives");
    m_prim.def("build_psi_linear", &buildPsiLinear, py::arg("dim"), py::arg("K"), "Linear functions as primitives");
}
