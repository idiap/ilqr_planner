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

#include "ilqr_planner/system/PosOrnTimePlannerSys.h"
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "ilqr_planner/utils/sd.h"
#include "ilqr_planner/utils/utils.h"
using namespace Eigen;

namespace ilqr_planner {
namespace sys {

PosOrnTimePlannerSys::PosOrnTimePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                           const VectorXd& RtDiag,
                                           const VectorXd& qMax,
                                           const VectorXd& qMin,
                                           const VectorXd& dqMax,
                                           const VectorXd& dqMin,
                                           int horizon,
                                           int nbDeriv)
    : System(r, keypoints, RtDiag, qMax, qMin, dqMax, dqMin, horizon, nbDeriv) {
    localInit();
}

PosOrnTimePlannerSys::PosOrnTimePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                           const VectorXd& RtDiag,
                                           const VectorXd& qMax,
                                           const VectorXd& qMin,
                                           int horizon,
                                           int nbDeriv)
    : System(r, keypoints, RtDiag, qMax, qMin, horizon, nbDeriv) {
    localInit();
}

PosOrnTimePlannerSys::PosOrnTimePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                           const VectorXd& RtDiag,
                                           int horizon,
                                           int nbDeriv)
    : System(r, keypoints, RtDiag, horizon, nbDeriv) {
    localInit();
}

void PosOrnTimePlannerSys::localInit() {
    this->checkKeypoints(EXPECTED_KP_TAG);

    this->q0 = r->getJointsPos();
    this->dq0 = r->getJointsVel();

    int nbCarDim = this->r->getEEPosition().rows();

    VectorXd f_x0(this->keypoints.at(0)->getState().rows());
    VectorXd x0(nbDeriv * this->r->getDOF() + 1);
    if (nbDeriv == 1) {
        f_x0 << this->r->getEEPosition(), this->r->getEEOrnQuat(), 0;
        x0 << q0, 0;
    } else {
        f_x0 << this->r->getEEPosition(), this->r->getEEOrnQuat(), this->r->getEEVelocity(), this->r->getEEAngVelQuat(), 0;
        x0 << q0, dq0, 0;
    }

    this->f_x0 = f_x0;
    this->x0 = x0;

    this->nbStateVar = this->x0.rows();
    this->nbCtrlVar = this->r->getDOF() + 1;
    this->nbTargetVar = this->f_x0.rows();
    this->nbQVar = this->nbTargetVar - this->nbDeriv;

    VectorXd state_max_augmented = VectorXd::Zero(this->state_max.rows() + 1);
    VectorXd state_min_augmented = VectorXd::Zero(this->state_min.rows() + 1);
    VectorXi joint_limits_weight_augmented = VectorXi::Zero(this->state_min.rows() + 1);

    state_max_augmented << this->state_max, 0;
    state_min_augmented << this->state_min, 0;
    joint_limits_weight_augmented << joint_limits_weight, 0;

    this->state_max = state_max_augmented;
    this->state_min = state_min_augmented;
    this->joint_limits_weight = joint_limits_weight_augmented;
}

std::tuple<VectorXd, MatrixXd> PosOrnTimePlannerSys::getFxJac() {
    VectorXd xk = this->r->getEEPosition();
    VectorXd et = this->r->getEEOrnQuat();

    VectorXd fx = VectorXd::Zero(this->nbTargetVar);

    MatrixXd J = this->r->J();
    MatrixXd Jk = MatrixXd::Zero(J.rows() + 1, J.cols() + 1);

    Jk.topLeftCorner(J.rows(), J.cols()) << J;
    Jk(Jk.rows() - 1, Jk.cols() - 1) = 1;

    if (this->nbDeriv == 1) {
        fx << xk, et, this->r->getTime();
        return std::make_tuple(fx, Jk);
    }

    VectorXd dxk = this->r->getEEVelocity();
    VectorXd det = this->r->getEEAngVelQuat();
    fx << xk, et, dxk, det, this->r->getTime();

    MatrixXd Js = MatrixXd ::Zero(2 * J.rows() + 1, 2 * J.cols() + 1);

    Js.topLeftCorner(J.rows(), J.cols()) << J;
    Js.block(J.rows(), J.cols(), J.rows(), J.cols()) << J;
    Js.bottomRightCorner(1, 1) << 1;
    return std::make_tuple(fx, Js);
}

std::tuple<VectorXd, MatrixXd> PosOrnTimePlannerSys::getFxJac(VectorXd xk) {
    VectorXd qk = xk.head(this->r->getDOF());
    VectorXd dqk = VectorXd::Zero(this->r->getDOF());

    double t = xk(xk.rows() - 1);

    if (this->nbDeriv == 2) {
        dqk = xk.segment(this->r->getDOF(), this->r->getDOF());
    }

    VectorXd old_q0 = this->r->getJointsPos();
    VectorXd old_dq0 = this->r->getJointsVel();
    double old_t = this->r->getTime();

    this->r->setConfiguration(qk, dqk);
    this->r->setTime(t);

    std::tuple<VectorXd, MatrixXd> fx_jac = this->getFxJac();

    this->r->setConfiguration(old_q0, old_dq0);
    this->r->setTime(old_t);

    return fx_jac;
}

VectorXd PosOrnTimePlannerSys::getState() {
    VectorXd xk(this->nbStateVar);
    if (this->nbDeriv == 1) {
        xk << this->r->getJointsPos(), this->r->getTime();
    } else {
        xk << this->r->getJointsPos(), this->r->getJointsVel(), this->r->getTime();
    }
    return xk;
}

std::tuple<VectorXd, VectorXd, MatrixXd, MatrixXd, MatrixXd> PosOrnTimePlannerSys::forwardPass(const VectorXd& xk, const VectorXd& uk, int k) {
    MatrixXd A = MatrixXd::Identity(this->nbStateVar, this->nbStateVar);
    MatrixXd B = MatrixXd::Zero(this->nbStateVar, this->nbCtrlVar);
    VectorXd x(this->nbStateVar);

    double dtSqrt = uk(this->nbCtrlVar - 1);
    double dt = dtSqrt * dtSqrt;

    if (this->nbDeriv == 1) {
        VectorXd dq = uk.head(this->nbCtrlVar - 1);
        this->r->sendVel(dt, dq);

        B.topRows(this->r->getDOF()) << MatrixXd::Identity(this->r->getDOF(), this->r->getDOF()) * dt, 2 * dtSqrt * dq;
        B.bottomRightCorner(1, 1) << 2 * dtSqrt;

        x << this->r->getJointsPos(), this->r->getTime();
    } else {
        VectorXd ddq = uk.head(this->nbCtrlVar - 1);

        this->r->sendAcc(dt, ddq);
        x << this->r->getJointsPos(), this->r->getJointsVel(), this->r->getTime();

        A.block(0, this->r->getDOF(), this->r->getDOF(), this->r->getDOF()) << dt * MatrixXd::Identity(this->r->getDOF(), this->r->getDOF());

        B.block(0, 0, this->r->getDOF(), this->r->getDOF()) << MatrixXd::Identity(this->r->getDOF(), this->r->getDOF()) * dt * dt / 2;
        B.block(this->r->getDOF(), 0, this->r->getDOF(), this->r->getDOF()) = MatrixXd::Identity(this->r->getDOF(), this->r->getDOF()) * dt;

        B.col(this->nbCtrlVar - 1) << 2 * dtSqrt * this->r->getJointsVel() + 2 * dtSqrt * dtSqrt * dtSqrt * ddq, 2 * dtSqrt * ddq, 2 * dtSqrt;
    }

    auto fxJ = this->getFxJac();
    auto fx = std::get<0>(fxJ);
    auto J = std::get<1>(fxJ);

    return std::make_tuple(x, fx, A, B, J);
}

void PosOrnTimePlannerSys::reset() {
    this->r->setConfiguration(this->q0, this->dq0);
}
}  // namespace sys
}  // namespace ilqr_planner