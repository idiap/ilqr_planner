/**
    This package provides a C++ iLQR library that comes with its python bindings.
    It allows you to solve iLQR optimization problem on any robot as long as you
    provide an [URDF file](http://wiki.ros.org/urdf/Tutorials) describing the
    kinematics chain of the robot. For debugging purposes it also provide a 2D
    planar robots class that you can use. You can also apply a spatial
    transformation to compute robot task space information in the base frame of
    your choice (e.g. object frame).

    Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
    Written by Jeremy Maceiras <jeremy.maceiras@idiap.ch>

    This file is part of ilqr_planner.

    ilqr_planner is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    ilqr_planner is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ilqr_planner. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <eigen3/Eigen/Dense>

namespace ilqr_planner {
namespace sim {
class SimulationInterface {
public:
    virtual ~SimulationInterface() {}
    /**
     * Forward kinematics function.
     * From the current configuration (q,dq), this method compute (x,dx,w,ornQuat,J,Jp)
     */
    virtual void updateKinematics() = 0;

    /**
     * Return the translational Jacobian
     */
    Eigen::MatrixXd Jt();

    /**
     * Return the rotational Jacobian
     */
    Eigen::MatrixXd Jr();

    /**
     * Return the time derivative of the translational Jacobian
     */
    Eigen::MatrixXd Jtp();

    /**
     * Return the time derivative of the rotational Jacobian
     */
    Eigen::MatrixXd Jrp();

    /**
     * Return the full Jacobian
     */
    virtual Eigen::MatrixXd J();

    /**
     * Return the time derivative of the full Jacobian
     */
    virtual Eigen::MatrixXd Jp();

    /**
     * Return the quaternion derivative to angular velocities Jacobian
     */
    Eigen::MatrixXd dQuatToDxJac(const Eigen::VectorXd& quat);

    /**
     * Send a joint acceleration command to the robot
     */
    virtual void sendAcc(double dt, const Eigen::VectorXd& ddq, bool updateKin = true);

    /**
     * Send a joint velocity command to the robot
     */
    virtual void sendVel(double dt, const Eigen::VectorXd& dq, bool updateKin = true);

    /**
     * Return the end-effector position
     */
    virtual Eigen::VectorXd getEEPosition();

    /**
     * Return the end-effector velocity
     */
    virtual Eigen::VectorXd getEEVelocity();

    /**
     * Return the end-effector angular velocity
     */
    virtual Eigen::VectorXd getEEAngVel();
    Eigen::VectorXd getEEAngVelQuat();

    /**
     * Return the end-effector orientation
     */
    virtual Eigen::VectorXd getEEOrnQuat();

    /**
     * Return the joint position of the robot
     */
    Eigen::VectorXd getJointsPos();

    /**
     * Return the joint velocities of the robot.
     */
    Eigen::VectorXd getJointsVel();

    /**
     * Return the degree of freedom of the robot
     */
    int getDOF();

    /**
     * Return the number of cartesian dimension of the robot
     */
    int getNbCarDim() { return this->nbCarDim; }

    double getTime() { return this->t; }
    virtual void setTime(double time) { this->t = time; }

    /**
     * Update the current configuration of the robot
     */
    virtual void setConfiguration(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, bool reset_time = true);

protected:
    Eigen::VectorXd q, dq, ddq;
    Eigen::VectorXd x, dx;
    Eigen::VectorXd ornQuat, w;
    Eigen::MatrixXd Jac, dJac;

    int dof, nbCarDim;
    double t;
};
}  // namespace sim
}  // namespace ilqr_planner