// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

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
                                   int nb_deriv_,
                                   double dt)
    : System(r, keypoints, RtDiag, qMax, qMin, dqMax, dqMin, horizon, nb_deriv_, {"POS_ORN"}) {
    localInit(dt);
}

PosOrnPlannerSys::PosOrnPlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                   const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                   const VectorXd& RtDiag,
                                   const VectorXd& qMax,
                                   const VectorXd& qMin,
                                   int horizon,
                                   int nb_deriv,
                                   double dt)
    : System(r, keypoints, RtDiag, qMax, qMin, horizon, nb_deriv, {"POS_ORN"}) {
    localInit(dt);
}

PosOrnPlannerSys::PosOrnPlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                   const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                   const VectorXd& RtDiag,
                                   int horizon,
                                   int nb_deriv,
                                   double dt)
    : System(r, keypoints, RtDiag, horizon, nb_deriv, {"POS_ORN"}) {
    localInit(dt);
}

void PosOrnPlannerSys::localInit(double dt) {
    dt_ = dt;

    q0_ = r->getJointsPos();
    dq0_ = r->getJointsVel();

    VectorXd f_x0(keypoints.at(0)->getState().rows());
    VectorXd x0(nb_deriv_ * r->getDOF());

    if (nb_deriv_ == 1) {
        f_x0 << r->getEEPosition(), r->getEEOrnQuat();
        x0 << q0_;
    } else {
        f_x0 << r->getEEPosition(), r->getEEOrnQuat(), r->getEEVelocity(), r->getEEAngVelQuat();
        x0 << q0_, dq0_;
    }

    f_x0_ = f_x0;
    x0_ = x0;

    nb_state_var_ = x0_.rows();
    nb_ctrl_var_ = r->getDOF();
    nb_target_var_ = f_x0_.rows();
    nb_Q_var_ = nb_target_var_ - nb_deriv_;
}

std::tuple<VectorXd, MatrixXd> PosOrnPlannerSys::getFxJac() {
    VectorXd xk = r->getEEPosition();
    VectorXd et = r->getEEOrnQuat();

    VectorXd fx = VectorXd::Zero(nb_target_var_);
    fx.head(7) << xk, et;

    MatrixXd Jk = r->J();
    if (nb_deriv_ == 1) {
        return std::make_tuple(fx, Jk);
    }

    VectorXd dxk = r->getEEVelocity();
    VectorXd det = r->getEEAngVelQuat();
    fx.tail(7) << dxk, det;

    MatrixXd J = MatrixXd ::Zero(2 * Jk.rows(), 2 * Jk.cols());

    J.topLeftCorner(Jk.rows(), Jk.cols()) << Jk;
    J.bottomRightCorner(Jk.rows(), Jk.cols()) << Jk;

    return std::make_tuple(fx, J);
}

VectorXd PosOrnPlannerSys::getState() {
    VectorXd xk(nb_state_var_);
    if (nb_deriv_ == 1) {
        xk << r->getJointsPos();
    } else {
        xk << r->getJointsPos(), r->getJointsVel();
    }
    return xk;
}

std::tuple<VectorXd, VectorXd, MatrixXd, MatrixXd, MatrixXd> PosOrnPlannerSys::forwardPass(const VectorXd& xk, const VectorXd& uk, int k) {
    MatrixXd A = MatrixXd::Identity(nb_state_var_, nb_state_var_);
    MatrixXd B = MatrixXd::Zero(nb_state_var_, nb_ctrl_var_);
    VectorXd x(nb_state_var_);

    if (nb_deriv_ == 1) {
        B = MatrixXd::Identity(nb_state_var_, nb_ctrl_var_) * dt_;
        r->sendVel(dt_, uk);

        x << r->getJointsPos();
    } else {
        A.topRightCorner(r->getDOF(), r->getDOF()) << MatrixXd::Identity(r->getDOF(), r->getDOF()) * dt_;

        B.topRows(r->getDOF()) = MatrixXd::Identity(r->getDOF(), r->getDOF()) * dt_ * dt_ / 2;
        B.bottomRows(r->getDOF()) = MatrixXd::Identity(r->getDOF(), r->getDOF()) * dt_;
        r->sendAcc(dt_, uk);
        x << r->getJointsPos(), r->getJointsVel();
    }

    auto fxJ = getFxJac();
    auto fx = std::get<0>(fxJ);
    auto J = std::get<1>(fxJ);

    return std::make_tuple(x, fx, A, B, J);
}

void PosOrnPlannerSys::reset() {
    r->setConfiguration(q0_, dq0_);
}
}  // namespace sys
}  // namespace ilqr_planner
