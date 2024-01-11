// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/system/JointSpacePlannerSys.h"
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "ilqr_planner/utils/sd.h"
#include "ilqr_planner/utils/utils.h"

using namespace Eigen;

namespace ilqr_planner {
namespace sys {

JointSpacePlannerSys::JointSpacePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                           const VectorXd& RtDiag,
                                           const VectorXd& qMax,
                                           const VectorXd& qMin,
                                           const VectorXd& dqMax,
                                           const VectorXd& dqMin,
                                           int horizon,
                                           int nb_deriv_,
                                           double dt)
    : System(r, keypoints, RtDiag, qMax, qMin, dqMax, dqMin, horizon, nb_deriv_, {"JNT"}) {
    localInit(dt);
}

JointSpacePlannerSys::JointSpacePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                           const VectorXd& RtDiag,
                                           const VectorXd& qMax,
                                           const VectorXd& qMin,
                                           int horizon,
                                           int nb_deriv,
                                           double dt)
    : System(r, keypoints, RtDiag, qMax, qMin, horizon, nb_deriv, {"JNT"}) {
    localInit(dt);
}

JointSpacePlannerSys::JointSpacePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                                           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                                           const VectorXd& RtDiag,
                                           int horizon,
                                           int nb_deriv,
                                           double dt)
    : System(r, keypoints, RtDiag, horizon, nb_deriv, {"JNT"}) {
    localInit(dt);
}

void JointSpacePlannerSys::localInit(double dt) {
    dt_ = dt;

    q0_ = r->getJointsPos();
    dq0_ = r->getJointsVel();

    VectorXd x0(nb_deriv_ * r->getDOF());

    if (nb_deriv_ == 1) {
        x0 << q0_;
    } else {
        x0 << q0_, dq0_;
    }

    f_x0_ = x0;
    x0_ = x0;

    nb_state_var_ = x0_.rows();
    nb_ctrl_var_ = r->getDOF();
    nb_target_var_ = f_x0_.rows();
    nb_Q_var_ = nb_target_var_;
}

std::tuple<VectorXd, MatrixXd> JointSpacePlannerSys::getFxJac() {
    VectorXd fx = getState();
    MatrixXd jac = MatrixXd::Identity(nb_Q_var_, nb_ctrl_var_);
    return std::make_tuple(fx, jac);
}

VectorXd JointSpacePlannerSys::getState() {
    VectorXd xk(nb_state_var_);
    if (nb_deriv_ == 1) {
        xk << r->getJointsPos();
    } else {
        xk << r->getJointsPos(), r->getJointsVel();
    }
    return xk;
}

std::tuple<VectorXd, VectorXd, MatrixXd, MatrixXd, MatrixXd> JointSpacePlannerSys::forwardPass(const VectorXd& xk, const VectorXd& uk, int k) {
    MatrixXd A = MatrixXd::Identity(nb_state_var_, nb_state_var_);
    MatrixXd B = MatrixXd::Zero(nb_state_var_, nb_ctrl_var_);
    VectorXd x(nb_state_var_);

    if (nb_deriv_ == 1) {
        B = MatrixXd::Identity(nb_state_var_, nb_ctrl_var_) * dt_;
        r->sendVel(dt_, uk);
    } else {
        A.topRightCorner(r->getDOF(), r->getDOF()) << MatrixXd::Identity(r->getDOF(), r->getDOF()) * dt_;

        B.topRows(r->getDOF()) = MatrixXd::Identity(r->getDOF(), r->getDOF()) * dt_ * dt_ / 2;
        B.bottomRows(r->getDOF()) = MatrixXd::Identity(r->getDOF(), r->getDOF()) * dt_;
        r->sendAcc(dt_, uk);
    }

    x = getState();

    auto fxJ = getFxJac();
    auto fx = std::get<0>(fxJ);
    auto J = std::get<1>(fxJ);

    return std::make_tuple(x, fx, A, B, J);
}

void JointSpacePlannerSys::reset() {
    r->setConfiguration(q0_, dq0_);
}
}  // namespace sys
}  // namespace ilqr_planner
