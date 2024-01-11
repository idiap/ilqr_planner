// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

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
                                           int nb_deriv_)
    : System(r, keypoints, RtDiag, qMax, qMin, dqMax, dqMin, horizon, nb_deriv_, {"POS_ORN_TIME"}) {
    localInit();
}

PosOrnTimePlannerSys::PosOrnTimePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                           const VectorXd& RtDiag,
                                           const VectorXd& qMax,
                                           const VectorXd& qMin,
                                           int horizon,
                                           int nb_deriv_)
    : System(r, keypoints, RtDiag, qMax, qMin, horizon, nb_deriv_, {"POS_ORN_TIME"}) {
    localInit();
}

PosOrnTimePlannerSys::PosOrnTimePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                           const VectorXd& RtDiag,
                                           int horizon,
                                           int nb_deriv_)
    : System(r, keypoints, RtDiag, horizon, nb_deriv_, {"POS_ORN_TIME"}) {
    localInit();
}

void PosOrnTimePlannerSys::localInit() {
    q0_ = r->getJointsPos();
    dq0_ = r->getJointsVel();

    VectorXd f_x0(keypoints.at(0)->getState().rows());
    VectorXd x0(nb_deriv_ * r->getDOF() + 1);
    if (nb_deriv_ == 1) {
        f_x0 << r->getEEPosition(), r->getEEOrnQuat(), 0;
        x0 << q0_, 0;
    } else {
        f_x0 << r->getEEPosition(), r->getEEOrnQuat(), r->getEEVelocity(), r->getEEAngVelQuat(), 0;
        x0 << q0_, dq0_, 0;
    }

    f_x0_ = f_x0;
    x0_ = x0;

    nb_state_var_ = x0_.rows();
    nb_ctrl_var_ = r->getDOF() + 1;
    nb_target_var_ = f_x0_.rows();
    nb_Q_var_ = nb_target_var_ - nb_deriv_;

    VectorXd state_max_augmented = VectorXd::Zero(state_max_.rows() + 1);
    VectorXd state_min_augmented = VectorXd::Zero(state_min_.rows() + 1);
    VectorXi joint_limits_weight_augmented = VectorXi::Zero(state_min_.rows() + 1);

    state_max_augmented << state_max_, 0;
    state_min_augmented << state_min_, 0;
    joint_limits_weight_augmented << joint_limits_weight_, 0;

    state_max_ = state_max_augmented;
    state_min_ = state_min_augmented;
    joint_limits_weight_ = joint_limits_weight_augmented;
}

std::tuple<VectorXd, MatrixXd> PosOrnTimePlannerSys::getFxJac() {
    VectorXd xk = r->getEEPosition();
    VectorXd et = r->getEEOrnQuat();

    VectorXd fx = VectorXd::Zero(nb_target_var_);

    MatrixXd J = r->J();
    MatrixXd Jk = MatrixXd::Zero(J.rows() + 1, J.cols() + 1);

    Jk.topLeftCorner(J.rows(), J.cols()) << J;
    Jk(Jk.rows() - 1, Jk.cols() - 1) = 1;

    if (nb_deriv_ == 1) {
        fx << xk, et, r->getTime();
        return std::make_tuple(fx, Jk);
    }

    VectorXd dxk = r->getEEVelocity();
    VectorXd det = r->getEEAngVelQuat();
    fx << xk, et, dxk, det, r->getTime();

    MatrixXd Js = MatrixXd ::Zero(2 * J.rows() + 1, 2 * J.cols() + 1);

    Js.topLeftCorner(J.rows(), J.cols()) << J;
    Js.block(J.rows(), J.cols(), J.rows(), J.cols()) << J;
    Js.bottomRightCorner(1, 1) << 1;
    return std::make_tuple(fx, Js);
}

std::tuple<VectorXd, MatrixXd> PosOrnTimePlannerSys::getFxJac(VectorXd xk) {
    VectorXd qk = xk.head(r->getDOF());
    VectorXd dqk = VectorXd::Zero(r->getDOF());

    double t = xk(xk.rows() - 1);

    if (nb_deriv_ == 2) {
        dqk = xk.segment(r->getDOF(), r->getDOF());
    }

    VectorXd old_q0 = r->getJointsPos();
    VectorXd old_dq0 = r->getJointsVel();
    double old_t = r->getTime();

    r->setConfiguration(qk, dqk);
    r->setTime(t);

    std::tuple<VectorXd, MatrixXd> fx_jac = getFxJac();

    r->setConfiguration(old_q0, old_dq0);
    r->setTime(old_t);

    return fx_jac;
}

VectorXd PosOrnTimePlannerSys::getState() {
    VectorXd xk(nb_state_var_);
    if (nb_deriv_ == 1) {
        xk << r->getJointsPos(), r->getTime();
    } else {
        xk << r->getJointsPos(), r->getJointsVel(), r->getTime();
    }
    return xk;
}

std::tuple<VectorXd, VectorXd, MatrixXd, MatrixXd, MatrixXd> PosOrnTimePlannerSys::forwardPass(const VectorXd& xk, const VectorXd& uk, int k) {
    MatrixXd A = MatrixXd::Identity(nb_state_var_, nb_state_var_);
    MatrixXd B = MatrixXd::Zero(nb_state_var_, nb_ctrl_var_);
    VectorXd x(nb_state_var_);

    double dtSqrt = uk(nb_ctrl_var_ - 1);
    double dt = dtSqrt * dtSqrt;

    if (nb_deriv_ == 1) {
        VectorXd dq = uk.head(nb_ctrl_var_ - 1);
        r->sendVel(dt, dq);

        B.topRows(r->getDOF()) << MatrixXd::Identity(r->getDOF(), r->getDOF()) * dt, 2 * dtSqrt * dq;
        B.bottomRightCorner(1, 1) << 2 * dtSqrt;

        x << r->getJointsPos(), r->getTime();
    } else {
        VectorXd ddq = uk.head(nb_ctrl_var_ - 1);

        r->sendAcc(dt, ddq);
        x << r->getJointsPos(), r->getJointsVel(), r->getTime();

        A.block(0, r->getDOF(), r->getDOF(), r->getDOF()) << dt * MatrixXd::Identity(r->getDOF(), r->getDOF());

        B.block(0, 0, r->getDOF(), r->getDOF()) << MatrixXd::Identity(r->getDOF(), r->getDOF()) * dt * dt / 2;
        B.block(r->getDOF(), 0, r->getDOF(), r->getDOF()) = MatrixXd::Identity(r->getDOF(), r->getDOF()) * dt;

        B.col(nb_ctrl_var_ - 1) << 2 * dtSqrt * r->getJointsVel() + 2 * dtSqrt * dtSqrt * dtSqrt * ddq, 2 * dtSqrt * ddq, 2 * dtSqrt;
    }

    auto fxJ = getFxJac();
    auto fx = std::get<0>(fxJ);
    auto J = std::get<1>(fxJ);

    return std::make_tuple(x, fx, A, B, J);
}

void PosOrnTimePlannerSys::reset() {
    r->setConfiguration(q0_, dq0_);
}
}  // namespace sys
}  // namespace ilqr_planner
