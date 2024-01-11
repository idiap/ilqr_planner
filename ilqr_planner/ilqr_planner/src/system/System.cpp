// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/system/System.h"
#include "ilqr_planner/utils/sd.h"
#include "ilqr_planner/utils/utils.h"

#include <algorithm>

using namespace Eigen;

namespace ilqr_planner {
namespace sys {

System::System(const std::shared_ptr<sim::SimulationInterface>& r,
               const std::vector<std::shared_ptr<Keypoint>>& keypoints,
               const VectorXd& RtDiag,
               const VectorXd& qMax,
               const VectorXd& qMin,
               const int& horizon,
               const int& nb_deriv,
               const std::vector<std::string>& allowed_kp_tags)
    : System(r, keypoints, RtDiag, qMax, qMin, VectorXd::Zero(r->getDOF()), VectorXd::Zero(r->getDOF()), horizon, nb_deriv, allowed_kp_tags) {}

System::System(const std::shared_ptr<sim::SimulationInterface>& r,
               const std::vector<std::shared_ptr<Keypoint>>& keypoints,
               const VectorXd& RtDiag,
               const VectorXd& qMax,
               const VectorXd& qMin,
               const VectorXd& dqMax,
               const VectorXd& dqMin,
               const int& horizon,
               const int& nb_deriv,
               const std::vector<std::string>& allowed_kp_tags)
    : r(r), keypoints(keypoints), horizon_(horizon), nb_deriv_(nb_deriv), EXPECTED_KP_TAGS_(allowed_kp_tags) {
    limits_set_ = true;
    penalty_ = 1;
    R = (RtDiag.asDiagonal());
    init();

    int state_space_size = nb_deriv_ * r->getDOF();

    state_max_ = VectorXd::Zero(state_space_size);
    state_min_ = VectorXd::Zero(state_space_size);

    joint_limits_weight_ = VectorXi::Ones(state_space_size);

    if (nb_deriv_ == 1) {
        state_max_ = qMax;
        state_min_ = qMin;
    } else if (nb_deriv_ == 2) {
        state_max_ << qMax, dqMax;
        state_min_ << qMin, dqMin;

        if (dqMax.isApprox(dqMin)) {
            joint_limits_weight_.tail(r->getDOF()) = VectorXi::Zero(r->getDOF());
        }
    }
}

System::System(const std::shared_ptr<sim::SimulationInterface>& r,
               const std::vector<std::shared_ptr<Keypoint>>& keypoints,
               const VectorXd& RtDiag,
               const int& horizon,
               const int& nb_deriv,
               const std::vector<std::string>& allowed_kp_tags)
    : r(r), keypoints(keypoints), horizon_(horizon), nb_deriv_(nb_deriv), EXPECTED_KP_TAGS_(allowed_kp_tags) {
    limits_set_ = false;
    penalty_ = 0;
    R = (RtDiag.asDiagonal());
    init();
}

void System::init() {
    for (auto& kp : keypoints) {
        keypoints_map_[kp->getTimestep()] = kp;
    }

    std::sort(keypoints.begin(), keypoints.end(), [](const std::shared_ptr<Keypoint>& a, const std::shared_ptr<Keypoint>& b) -> bool { return a->getTimestep() < b->getTimestep(); });
    if (EXPECTED_KP_TAGS_.size() > 0) {
        checkKeypoints();
    }
}

std::vector<int> System::getKpIndexes() {
    std::vector<int> kpIndexes;
    for (auto const& kp : keypoints) {
        kpIndexes.push_back(kp->getTimestep());
    }
    return kpIndexes;
}

std::shared_ptr<Keypoint> System::getKeypoint(int k) {
    if (keypoints_map_.find(k) != keypoints_map_.end()) {
        return keypoints_map_[k];
    }
    return nullptr;
}

VectorXd System::diff(const VectorXd& actual_state, int k) {
    if (keypoints_map_.find(k) == keypoints_map_.end()) {
        return VectorXd::Zero(nb_Q_var_);
    }

    return keypoints_map_[k]->diff(actual_state);
}

VectorXd System::diffBatch(const VectorXd& x) {
    int horizon_ = keypoints.size();
    VectorXd residual = VectorXd::Zero(horizon_ * nb_Q_var_);
    for (int i = 0; i < horizon_; i++) {
        VectorXd xt = x.segment(i * nb_target_var_, nb_target_var_);
        residual.segment(i * nb_Q_var_, nb_Q_var_) = diff(xt, keypoints.at(i)->getTimestep());
    }
    return residual;
}

std::pair<MatrixXd, VectorXd> System::inspectJointLimit(VectorXd x_k) {
    MatrixXd L = MatrixXd::Zero(nb_state_var_, nb_state_var_);
    VectorXd q = VectorXd::Zero(nb_state_var_);

    if (limits_set_) {
        auto maxViolation = x_k.array() > state_max_.array();
        auto minViolation = x_k.array() < state_min_.array();

        for (int i = 0; i < nb_state_var_; i++) {
            if (joint_limits_weight_(i) != 0) {
                if (maxViolation(i)) {
                    q(i) = state_max_(i) - x_k(i);
                    L(i, i) = penalty_;
                } else if (minViolation(i)) {
                    q(i) = state_min_(i) - x_k(i);
                    L(i, i) = penalty_;
                }
            }
        }
    }
    return std::make_pair(L, q);
}

std::tuple<VectorXd, VectorXd, VectorXd, VectorXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd> System::forwardPassWithLimits(const VectorXd& xk, const VectorXd& uk, int k) {
    auto xFxABJ = forwardPass(xk, uk, k);

    auto x = std::get<0>(xFxABJ);
    auto fx = std::get<1>(xFxABJ);
    auto A = std::get<2>(xFxABJ);
    auto B = std::get<3>(xFxABJ);
    auto J = std::get<4>(xFxABJ);

    VectorXd u = VectorXd::Zero(nb_ctrl_var_);

    MatrixXd L;
    VectorXd q;

    std::tie(L, q) = inspectJointLimit(xk);

    return std::make_tuple(x, fx, q, u, A, B, J, L);
}

std::tuple<VectorXd, MatrixXd> System::getFxJac(VectorXd xk) {
    VectorXd qk = xk.head(r->getDOF());
    VectorXd dqk = VectorXd::Zero(r->getDOF());

    if (nb_deriv_ == 2) {
        dqk = xk.segment(r->getDOF(), r->getDOF());
    }

    VectorXd old_q0 = r->getJointsPos();
    VectorXd old_dq0 = r->getJointsVel();

    r->setConfiguration(qk, dqk);
    auto fx_jac = getFxJac();
    r->setConfiguration(old_q0, old_dq0);

    return fx_jac;
}

std::tuple<VectorXd, VectorXd, std::vector<std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>>> System::fpBatch(const VectorXd& u) {
    reset();
    int horizon_ = u.rows() / nb_ctrl_var_ + 1;

    VectorXd fX = VectorXd::Zero(horizon_ * nb_target_var_);
    VectorXd qL = VectorXd::Zero(horizon_ * nb_state_var_);
    VectorXd uL = VectorXd::Zero((horizon_ - 1) * nb_ctrl_var_);

    auto fJ0 = getFxJac();
    fX.head(nb_target_var_) << std::get<0>(fJ0);
    MatrixXd J0 = std::get<1>(fJ0);

    std::vector<std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>> ABJLs;

    MatrixXd A = MatrixXd::Identity(nb_state_var_, nb_state_var_);
    MatrixXd B = MatrixXd::Zero(nb_state_var_, nb_ctrl_var_);
    MatrixXd L = MatrixXd::Zero(nb_state_var_, nb_state_var_);

    ABJLs.push_back(std::make_tuple(A, B, J0, L));

    for (int i = 0; i < horizon_ - 1; i++) {
        VectorXd ut = u.segment(i * nb_ctrl_var_, nb_ctrl_var_);
        auto xFxquABJLLc = forwardPassWithLimits(getState(), ut, i + 1);

        fX.segment((i + 1) * nb_target_var_, nb_target_var_) << std::get<1>(xFxquABJLLc);
        qL.segment((i + 1) * nb_state_var_, nb_state_var_) << std::get<2>(xFxquABJLLc);
        uL.segment(i * nb_ctrl_var_, nb_ctrl_var_) << std::get<3>(xFxquABJLLc);
        ABJLs.push_back(std::make_tuple(std::get<4>(xFxquABJLLc), std::get<5>(xFxquABJLLc), std::get<6>(xFxquABJLLc), std::get<7>(xFxquABJLLc)));
    }
    return std::make_tuple(fX, qL, ABJLs);
}

VectorXd System::cost(const VectorXd& xk, const VectorXd& uk, int k) {
    VectorXd cost = VectorXd::Zero(1);

    // Task reaching part
    std::shared_ptr<Keypoint> kp = getKeypoint(k);
    if (kp != nullptr) {
        auto fxk = std::get<0>(getFxJac(xk));
        VectorXd diff = kp->diff(fxk);
        cost += diff.transpose() * kp->getPrecision() * diff + uk.transpose() * R * uk;
    }

    // Limit avoidance part
    if (limits_set_) {
        MatrixXd L;
        VectorXd q;

        std::tie(L, q) = inspectJointLimit(xk);
        cost += q.transpose() * L * q;
    }

    return cost;
}

VectorXd System::cost_F(const VectorXd& xk) {
    return cost(xk, VectorXd::Zero(nb_ctrl_var_), horizon_ - 1);
}

VectorXd System::cost_F_x(const VectorXd& xk) {
    return cost_x(xk, VectorXd::Zero(nb_ctrl_var_), horizon_ - 1);
}

MatrixXd System::cost_F_xx(const VectorXd& xk) {
    return cost_xx(xk, VectorXd::Zero(nb_ctrl_var_), horizon_ - 1);
}

VectorXd System::cost_x(const VectorXd& xk, const VectorXd& uk, int k) {
    VectorXd cost = VectorXd::Zero(nb_state_var_);

    // Task reaching part
    std::shared_ptr<Keypoint> kp = getKeypoint(k);
    if (kp != nullptr) {
        auto fxk_J = getFxJac(xk);
        auto fxk = std::get<0>(fxk_J);
        auto Jk = std::get<1>(fxk_J);

        VectorXd diff = kp->diff(fxk);
        cost += -1 * (Jk).transpose() * kp->getPrecision() * diff;
    }

    // Limit avoidance part
    if (limits_set_) {
        MatrixXd L;
        VectorXd q;

        std::tie(L, q) = inspectJointLimit(xk);
        cost += -L.transpose() * q;
    }

    return cost;
}

VectorXd System::cost_u(const VectorXd& xk, const VectorXd& uk, int k) {
    return R * uk;
}

MatrixXd System::cost_ux(const VectorXd& xk, const VectorXd& uk, int k) {
    return MatrixXd::Zero(uk.rows(), nb_state_var_);
}

MatrixXd System::cost_uu(const VectorXd& xk, const VectorXd& uk, int k) {
    return R;
}

MatrixXd System::cost_xx(const VectorXd& xk, const VectorXd& uk, int k) {
    MatrixXd cost = MatrixXd::Zero(nb_state_var_, nb_state_var_);

    // Task reaching part
    std::shared_ptr<Keypoint> kp = getKeypoint(k);
    if (kp != nullptr) {
        auto fxk_J = getFxJac(xk);
        auto Jk = std::get<1>(fxk_J);

        cost += Jk.transpose() * kp->getPrecision() * Jk;
    }

    // Limit avoidance part
    if (limits_set_) {
        MatrixXd L;
        VectorXd q;

        std::tie(L, q) = inspectJointLimit(xk);
        cost += L.transpose() * L;
    }

    return cost;
}

MatrixXd System::cost_xu(const VectorXd& xk, const VectorXd& uk, int k) {
    return MatrixXd::Zero(nb_state_var_, uk.rows());
}

VectorXd System::getInitState() {
    return x0_;
}
VectorXd System::getInitFoXState() {
    return f_x0_;
}

VectorXd System::getMuVector(bool sparse) {
    if (sparse) {
        VectorXd mu = VectorXd::Zero(nb_target_var_ * keypoints.size());
        for (int i = 0; i < keypoints.size(); i++) {
            mu.segment(i * nb_target_var_, nb_target_var_) = keypoints.at(i)->getState();
        }
        return mu;
    } else {
        VectorXd mu = VectorXd::Zero(horizon_ * nb_target_var_);

        int i = 0;
        for (auto const& kp : keypoints) {
            mu.segment(kp->getTimestep() * nb_target_var_, nb_target_var_) = kp->getState();
            i++;
        }

        return mu;
    }
}

MatrixXd System::getQMatrix(bool sparse) {
    if (sparse) {
        MatrixXd Q = MatrixXd::Zero(keypoints.size() * nb_Q_var_, keypoints.size() * nb_Q_var_);

        for (int i = 0; i < keypoints.size(); i++) {
            Q.block(i * nb_Q_var_, i * nb_Q_var_, nb_Q_var_, nb_Q_var_) = keypoints.at(i)->getPrecision();
        }

        return Q;
    } else {
        MatrixXd Q = MatrixXd::Zero(horizon_ * nb_Q_var_, horizon_ * nb_Q_var_);

        int i = 0;
        for (auto const& kp : keypoints) {
            Q.block(kp->getTimestep() * nb_Q_var_, kp->getTimestep() * nb_Q_var_, nb_Q_var_, nb_Q_var_) = kp->getPrecision();
            i++;
        }

        return Q;
    }
}

void System::checkKeypoints() {
    for (auto kp : keypoints) {
        if (std::find(EXPECTED_KP_TAGS_.begin(), EXPECTED_KP_TAGS_.end(), kp->getTAG()) == EXPECTED_KP_TAGS_.end()) {
            throw std::runtime_error("[PosOrnPlannerSys] Wrong keypoint type: got " + kp->getTAG());
        }
        if (kp->getType() != nb_deriv_) {
            throw std::runtime_error("[PosOrnPlannerSys] Wrong keypoint order (nb_deriv_): Expecting " + std::to_string(nb_deriv_) + " got " + std::to_string(kp->getType()));
        }
    }
}

}  // namespace sys
}  // namespace ilqr_planner
