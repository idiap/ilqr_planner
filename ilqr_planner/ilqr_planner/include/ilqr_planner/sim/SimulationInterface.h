// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

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
