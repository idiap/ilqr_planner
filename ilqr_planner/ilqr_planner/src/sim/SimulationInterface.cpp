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

#include "ilqr_planner/sim/SimulationInterface.h"
#include <iostream>
using namespace Eigen;

namespace ilqr_planner {
namespace sim {
MatrixXd SimulationInterface::dQuatToDxJac(const VectorXd& quat) {
    MatrixXd J(3, 4);
    J << -quat(1), quat(0), -quat(3), quat(2), -quat(2), quat(3), quat(0), -quat(1), -quat(3), -quat(2), quat(1), quat(0);
    return J;
}

void SimulationInterface::sendAcc(double dt, const VectorXd& ddq, bool updateKin) {
    this->q += dt * this->dq + dt * dt / 2 * ddq;
    this->dq += dt * ddq;
    this->t += dt;
    if (updateKin)
        this->updateKinematics();
    this->ddq = ddq;
}

void SimulationInterface::sendVel(double dt, const VectorXd& dq, bool updateKin) {
    this->dq = dq;
    this->sendAcc(dt, VectorXd::Zero(dq.rows()), updateKin);
}

MatrixXd SimulationInterface::Jt() {
    return this->J().topRows(this->nbCarDim);
}

MatrixXd SimulationInterface::Jr() {
    return this->J().bottomRows(this->nbCarDim);
}

MatrixXd SimulationInterface::Jtp() {
    return this->Jp().topRows(this->nbCarDim);
}

MatrixXd SimulationInterface::Jrp() {
    return this->Jp().bottomRows(this->nbCarDim);
}

MatrixXd SimulationInterface::J() {
    return this->Jac;
}

MatrixXd SimulationInterface::Jp() {
    return this->dJac;
}

VectorXd SimulationInterface::getEEPosition() {
    return this->x;
}

VectorXd SimulationInterface::getEEVelocity() {
    return this->dx;
}

VectorXd SimulationInterface::getEEAngVel() {
    return this->w;
}

VectorXd SimulationInterface::getEEAngVelQuat() {
    auto w = this->getEEAngVel();
    auto et = this->getEEOrnQuat();
    return .5 * this->dQuatToDxJac(et).transpose() * w;
}

VectorXd SimulationInterface::getEEOrnQuat() {
    return this->ornQuat;
}

VectorXd SimulationInterface::getJointsPos() {
    return this->q;
}

VectorXd SimulationInterface::getJointsVel() {
    return this->dq;
}

int SimulationInterface::getDOF() {
    return this->dof;
}

void SimulationInterface::setConfiguration(const VectorXd& q, const VectorXd& dq, bool reset_time) {
    this->q = q;
    this->dq = dq;
    this->updateKinematics();
    if (reset_time) {
        this->t = 0;
    }
}
}  // namespace sim
}  // namespace ilqr_planner