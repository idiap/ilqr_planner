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

#include "ilqr_planner/system/PosOrnPlannerSys.h"
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "ilqr_planner/utils/sd.h"
#include "ilqr_planner/utils/utils.h"

using namespace Eigen;

namespace ilqr_planner {
namespace sys {

PosOrnPlannerSys::PosOrnPlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                   const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                   const VectorXd& RtDiag,
                                   const VectorXd& qMax,
                                   const VectorXd& qMin,
                                   const VectorXd& dqMax,
                                   const VectorXd& dqMin,
                                   int horizon,
                                   int nbDeriv,
                                   double dt)
    : System(r, keypoints, RtDiag, qMax, qMin, dqMax, dqMin, horizon, nbDeriv) {
    this->localInit(dt);
}

PosOrnPlannerSys::PosOrnPlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                   const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                   const VectorXd& RtDiag,
                                   const VectorXd& qMax,
                                   const VectorXd& qMin,
                                   int horizon,
                                   int nbDeriv,
                                   double dt)
    : System(r, keypoints, RtDiag, qMax, qMin, horizon, nbDeriv) {
    this->localInit(dt);
}

PosOrnPlannerSys::PosOrnPlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                   const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                   const VectorXd& RtDiag,
                                   int horizon,
                                   int nbDeriv,
                                   double dt)
    : System(r, keypoints, RtDiag, horizon, nbDeriv) {
    this->localInit(dt);
}

void PosOrnPlannerSys::localInit(double dt) {
    this->checkKeypoints(EXPECTED_KP_TAG);

    this->dt = dt;

    this->q0 = this->r->getJointsPos();
    this->dq0 = this->r->getJointsVel();

    int nbCarDim = this->r->getEEPosition().rows();

    VectorXd f_x0(this->keypoints.at(0)->getState().rows());
    VectorXd x0(nbDeriv * this->r->getDOF());

    if (nbDeriv == 1) {
        f_x0 << this->r->getEEPosition(), this->r->getEEOrnQuat();
        x0 << q0;
    } else {
        f_x0 << this->r->getEEPosition(), this->r->getEEOrnQuat(), this->r->getEEVelocity(), this->r->getEEAngVelQuat();
        x0 << q0, dq0;
    }

    this->f_x0 = f_x0;
    this->x0 = x0;

    this->nbStateVar = this->x0.rows();
    this->nbCtrlVar = this->r->getDOF();
    this->nbTargetVar = this->f_x0.rows();
    this->nbQVar = this->nbTargetVar - this->nbDeriv;
}

std::tuple<VectorXd, MatrixXd> PosOrnPlannerSys::getFxJac() {
    VectorXd xk = this->r->getEEPosition();
    VectorXd et = this->r->getEEOrnQuat();

    VectorXd fx = VectorXd::Zero(this->nbTargetVar);
    fx.head(7) << xk, et;

    MatrixXd Jk = this->r->J();
    if (this->nbDeriv == 1) {
        return std::make_tuple(fx, Jk);
    }

    VectorXd dxk = this->r->getEEVelocity();
    VectorXd det = this->r->getEEAngVelQuat();
    fx.tail(7) << dxk, det;

    MatrixXd J = MatrixXd ::Zero(2 * Jk.rows(), 2 * Jk.cols());

    J.topLeftCorner(Jk.rows(), Jk.cols()) << Jk;
    J.bottomRightCorner(Jk.rows(), Jk.cols()) << Jk;

    return std::make_tuple(fx, J);
}

VectorXd PosOrnPlannerSys::getState() {
    VectorXd xk(this->nbStateVar);
    if (this->nbDeriv == 1) {
        xk << this->r->getJointsPos();
    } else {
        xk << this->r->getJointsPos(), this->r->getJointsVel();
    }
    return xk;
}

std::tuple<VectorXd, VectorXd, MatrixXd, MatrixXd, MatrixXd> PosOrnPlannerSys::forwardPass(const VectorXd& xk, const VectorXd& uk, int k) {
    MatrixXd A = MatrixXd::Identity(this->nbStateVar, this->nbStateVar);
    MatrixXd B = MatrixXd::Zero(this->nbStateVar, this->nbCtrlVar);
    VectorXd x(this->nbStateVar);

    if (this->nbDeriv == 1) {
        B = MatrixXd::Identity(this->nbStateVar, this->nbCtrlVar) * this->dt;
        this->r->sendVel(this->dt, uk);

        x << this->r->getJointsPos();
    } else {
        A.topRightCorner(this->r->getDOF(), this->r->getDOF()) << MatrixXd::Identity(this->r->getDOF(), this->r->getDOF()) * this->dt;

        B.topRows(this->r->getDOF()) = MatrixXd::Identity(this->r->getDOF(), this->r->getDOF()) * this->dt * this->dt / 2;
        B.bottomRows(this->r->getDOF()) = MatrixXd::Identity(this->r->getDOF(), this->r->getDOF()) * this->dt;
        this->r->sendAcc(this->dt, uk);
        x << this->r->getJointsPos(), this->r->getJointsVel();
    }

    auto fxJ = this->getFxJac();
    auto fx = std::get<0>(fxJ);
    auto J = std::get<1>(fxJ);

    return std::make_tuple(x, fx, A, B, J);
}

void PosOrnPlannerSys::reset() {
    this->r->setConfiguration(this->q0, this->dq0);
}
}  // namespace sys
}  // namespace ilqr_planner