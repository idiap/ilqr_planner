// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

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
