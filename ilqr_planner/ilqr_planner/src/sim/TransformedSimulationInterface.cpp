// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/sim/TransformedSimulationInterface.h"

namespace ilqr_planner {
namespace sim {

using namespace Eigen;

TransformedSimulationInterface::TransformedSimulationInterface(const std::shared_ptr<SimulationInterface>& r, const MatrixXd& T) : r(r), T(T), initialized(true) {
    this->nbCarDim = r->getNbCarDim();
    this->dof = r->getDOF();
    this->updateKinematics();
}

TransformedSimulationInterface::TransformedSimulationInterface(const MatrixXd& T) : T(T), initialized(false) {}

void TransformedSimulationInterface::subscribe(const std::shared_ptr<SimulationInterface>& r) {
    this->r = r;
    this->nbCarDim = r->getNbCarDim();
    this->dof = r->getDOF();

    this->initialized = true;
    this->updateKinematics();
}

void TransformedSimulationInterface::updateKinematics() {
    if (!this->initialized) {
        throw std::runtime_error("[TransformedSimulationInterface] Object is not initialized");
    }

    this->r->updateKinematics();
    this->q = this->r->getJointsPos();
    this->dq = this->r->getJointsVel();
    this->dJac = this->r->Jp();
    this->Jac = this->r->J();
    this->x = this->r->getEEPosition();
    this->ornQuat = this->r->getEEOrnQuat();
    this->dx = this->r->getEEVelocity();
    this->w = this->r->getEEAngVel();
    this->t = this->r->getTime();
}

void TransformedSimulationInterface::setTime(double time) {
    this->t = time;
    this->r->setTime(time);
}

MatrixXd TransformedSimulationInterface::J() {
    Matrix<double, 6, 6> Taug = Matrix<double, 6, 6>::Zero(6, 6);
    Taug.topLeftCorner(3, 3) = this->T.topLeftCorner(3, 3);
    Taug.bottomRightCorner(3, 3) = this->T.topLeftCorner(3, 3);
    return Taug.transpose() * this->Jac;
}

MatrixXd TransformedSimulationInterface::Jp() {
    Matrix<double, 6, 6> Taug = Matrix<double, 6, 6>::Zero(6, 6);
    Taug.topLeftCorner(3, 3) = this->T.topLeftCorner(3, 3);
    Taug.bottomRightCorner(3, 3) = this->T.topLeftCorner(3, 3);
    return Taug.transpose() * this->dJac;
}

VectorXd TransformedSimulationInterface::getEEPosition() {
    return this->T.topLeftCorner(3, 3).transpose() * (this->x - this->T.col(3).head(3));
}

VectorXd TransformedSimulationInterface::getEEVelocity() {
    return this->T.topLeftCorner(3, 3).transpose() * this->dx;
}

VectorXd TransformedSimulationInterface::getEEAngVel() {
    return this->T.topLeftCorner(3, 3).transpose() * this->w;
}

void TransformedSimulationInterface::sendAcc(double dt, const VectorXd& ddq, bool updateKin) {
    this->r->sendAcc(dt, ddq, updateKin);
    this->updateKinematics();
}

void TransformedSimulationInterface::setConfiguration(const VectorXd& q, const VectorXd& dq, bool reset_time) {
    this->r->setConfiguration(q, dq, reset_time);
    this->updateKinematics();
}

void TransformedSimulationInterface::sendVel(double dt, const VectorXd& dq, bool updateKin) {
    this->r->sendVel(dt, dq, updateKin);
    this->updateKinematics();
}

VectorXd TransformedSimulationInterface::getEEOrnQuat() {
    VectorXd orn = this->ornQuat;
    Quaterniond baseOrn(orn(0), orn(1), orn(2), orn(3));
    Matrix3d destOrnMat = this->T.topLeftCorner(3, 3).transpose() * baseOrn.toRotationMatrix();
    Quaterniond destOrn(destOrnMat);

    VectorXd destOrnQuat = VectorXd::Zero(4);
    destOrnQuat << destOrn.w(), destOrn.x(), destOrn.y(), destOrn.z();
    return destOrnQuat;
}

}  // namespace sim
}  // namespace ilqr_planner
