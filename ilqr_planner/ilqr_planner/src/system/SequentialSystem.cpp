// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/system/SequentialSystem.h"
namespace ilqr_planner {
namespace sys {

using namespace Eigen;

SequentialSystem::SequentialSystem(const std::shared_ptr<sim::SimulationInterface>& r, const std::vector<std::shared_ptr<System>>& systems, const VectorXd& RtDiag, int horizon, int nbDeriv)
    : systems_(systems) {
    this->r = r;
    penalty_ = 0;
    limits_set_ = false;
    localInit(RtDiag);
}

void SequentialSystem::localInit(const VectorXd& RtDiag) {
    R = RtDiag.asDiagonal();

    int nb_state_var_0 = systems_.at(0)->getNbStateVar();
    int nb_ctrl_var_0 = systems_.at(0)->getNbCtrlVar();
    int horizon0 = systems_.at(0)->getHorizon();
    int nbDeriv0 = systems_.at(0)->getNbDeriv();

    VectorXd initState0 = systems_.at(0)->getInitState();

    int nb_target_var = systems_.at(0)->getNbTargetVar();
    int nb_Q_var = systems_.at(0)->getNbQVar();

    for (int i = 1; i < systems_.size(); i++) {
        auto system = systems_.at(i);
        nb_target_var += system->getNbTargetVar();
        nb_Q_var += system->getNbQVar();

        if (nb_state_var_0 != system->getNbStateVar()) {
            throw std::runtime_error(" All the systems does not have the same number of state variable ");
        }

        if (nb_ctrl_var_0 != system->getNbCtrlVar()) {
            throw std::runtime_error(" All the systems does not have the same number of control variable ");
        }

        if (horizon0 != system->getHorizon()) {
            throw std::runtime_error(" All the systems does not have the same horizon ");
        }

        if (nbDeriv0 != system->getNbDeriv()) {
            throw std::runtime_error(" All the systems does not have the same number of derivatives ");
        }

        if (initState0 != system->getInitState()) {
            throw std::runtime_error(" All the systems does not have the same initState ");
        }
    }

    nb_state_var_ = nb_state_var_0;
    nb_target_var_ = nb_target_var;
    nb_ctrl_var_ = nb_ctrl_var_0;
    nb_Q_var_ = nb_Q_var;
    horizon_ = horizon0;
    nb_deriv_ = nbDeriv0;
    x0_ = initState0;
    q0_ = r->getJointsPos();
    dq0_ = r->getJointsVel();
    f_x0_ = std::get<0>(getFxJac());

    for (auto sys : systems_) {
        keypoints.insert(keypoints.end(), sys->keypoints.begin(), sys->keypoints.end());
    }

    init();
}

std::tuple<VectorXd, VectorXd, MatrixXd, MatrixXd, MatrixXd> SequentialSystem::forwardPass(const VectorXd& xk, const VectorXd& uk, int k) {
    auto xFxABJ = systems_.at(0)->forwardPass(xk, uk, k);

    for (int i = 1; i < systems_.size(); i++) {
        systems_.at(i)->r->updateKinematics();
    }

    VectorXd x = std::get<0>(xFxABJ);
    MatrixXd A = std::get<2>(xFxABJ);
    MatrixXd B = std::get<3>(xFxABJ);
    auto fxJ = getFxJac();

    return std::make_tuple(x, std::get<0>(fxJ), A, B, std::get<1>(fxJ));
}

std::tuple<VectorXd, MatrixXd> SequentialSystem::getFxJac() {
    VectorXd fx = VectorXd::Zero(nb_target_var_);
    MatrixXd J = MatrixXd::Zero(nb_Q_var_, nb_state_var_);

    int fx_idx = 0;
    int J_idx = 0;

    for (auto sys : systems_) {
        auto fxJ_k = sys->getFxJac();
        VectorXd fxk = std::get<0>(fxJ_k);
        MatrixXd Jk = std::get<1>(fxJ_k);

        fx.segment(fx_idx, fxk.rows()) = fxk;
        J.block(J_idx, 0, Jk.rows(), Jk.cols()) = Jk;

        fx_idx += fxk.rows();
        J_idx += Jk.rows();
    }

    return std::make_tuple(fx, J);
}

VectorXd SequentialSystem::getState() {
    return systems_.at(0)->getState();
}

VectorXd SequentialSystem::cost_F(const VectorXd& xk) {
    VectorXd c = VectorXd::Zero(1);
    for (auto sys : systems_) {
        c += sys->cost_F(xk);
    }
    return c;
}

VectorXd SequentialSystem::cost_F_x(const VectorXd& xk) {
    VectorXd c = VectorXd::Zero(nb_state_var_);
    for (auto sys : systems_) {
        c += sys->cost_F_x(xk);
    }
    return c;
}

MatrixXd SequentialSystem::cost_F_xx(const VectorXd& xk) {
    MatrixXd c = MatrixXd::Zero(nb_state_var_, nb_state_var_);
    for (auto sys : systems_) {
        c += sys->cost_F_xx(xk);
    }
    return c;
}

VectorXd SequentialSystem::cost(const VectorXd& xk, const VectorXd& uk, int k) {
    VectorXd c = VectorXd::Zero(1);
    for (auto sys : systems_) {
        c += sys->cost(xk, uk, k);
    }
    return c;
}

VectorXd SequentialSystem::cost_x(const VectorXd& xk, const VectorXd& uk, int k) {
    VectorXd c = VectorXd::Zero(nb_state_var_);
    for (auto sys : systems_) {
        c += sys->cost_x(xk, uk, k);
    }
    return c;
}

MatrixXd SequentialSystem::cost_xx(const VectorXd& xk, const VectorXd& uk, int k) {
    MatrixXd c = MatrixXd::Zero(nb_state_var_, nb_state_var_);
    for (auto sys : systems_) {
        c += sys->cost_xx(xk, uk, k);
    }
    return c;
}

VectorXd SequentialSystem::diff(const VectorXd& state, int k) {
    VectorXd diff = VectorXd::Zero(nb_Q_var_);

    int Q_idx = 0;
    int target_idx = 0;

    for (auto sys : systems_) {
        int nb_target_var = sys->getNbTargetVar();
        int nb_Q_var = sys->getNbQVar();

        diff.segment(Q_idx, nb_Q_var) = sys->diff(state.segment(target_idx, nb_target_var), k);

        Q_idx += nb_Q_var;
        target_idx += nb_target_var;
    }

    return diff;
}

VectorXd SequentialSystem::getMuVector(bool sparse) {
    if (!sparse) {
        VectorXd mu = VectorXd::Zero(horizon_ * nb_target_var_);
        int idx = 0;
        for (auto sys : systems_) {
            VectorXd mu_k = sys->getMuVector();
            int nb_target_var_k = sys->getNbTargetVar();

            for (int j = 0; j < horizon_; j++) {
                VectorXd mu_k_t = mu_k.segment(j * nb_target_var_k, nb_target_var_k);
                mu.segment(j * nb_target_var_, nb_target_var_).segment(idx, nb_target_var_k) = mu_k_t;
            }

            idx += nb_target_var_k;
        }

        return mu;
    } else {
        VectorXd mu = VectorXd::Zero(nb_target_var_ * keypoints.size());

        for (int i = 0; i < keypoints.size(); i++) {
            VectorXd mu_t = VectorXd::Zero(nb_target_var_);
            int idx = 0;

            for (int j = 0; j < systems_.size(); j++) {
                auto kp = systems_.at(j)->getKeypoint(keypoints.at(i)->getTimestep());
                VectorXd mu_t_j = VectorXd::Zero(systems_.at(j)->getNbTargetVar());

                if (kp != nullptr) {
                    mu_t_j = kp->getState();
                }

                mu_t.segment(idx, mu_t_j.rows()) = mu_t_j;
                idx += mu_t_j.rows();
            }

            mu.segment(i * nb_target_var_, nb_target_var_) = mu_t;
        }

        return mu;
    }
}

MatrixXd SequentialSystem::getQMatrix(bool sparse) {
    // This solution is only concatening all the diagonal elements of each sub system matrices.
    // It means that at the moment you can not play with off-diagonal elements.
    // To do this, we need to change the algorithm and make it way more costly.

    if (!sparse) {
        MatrixXd Q = MatrixXd::Zero(horizon_ * nb_Q_var_, horizon_ * nb_Q_var_);

        int idx = 0;
        for (auto sys : systems_) {
            MatrixXd Q_k = sys->getQMatrix();
            int nb_Q_var_k = sys->getNbQVar();

            for (int j = 0; j < horizon_; j++) {
                MatrixXd Q_k_t = Q_k.block(j * nb_Q_var_k, j * nb_Q_var_k, nb_Q_var_k, nb_Q_var_k);
                Q.block(j * nb_Q_var_, j * nb_Q_var_, nb_Q_var_, nb_Q_var_).block(idx, idx, nb_Q_var_k, nb_Q_var_k) = Q_k_t;
            }
            idx += nb_Q_var_k;
        }

        return Q;
    } else {
        MatrixXd Q = MatrixXd::Zero(nb_Q_var_ * keypoints.size(), nb_Q_var_ * keypoints.size());

        for (int i = 0; i < keypoints.size(); i++) {
            MatrixXd Q_t = MatrixXd::Zero(nb_Q_var_, nb_Q_var_);
            int idx = 0;
            for (int j = 0; j < systems_.size(); j++) {
                auto kp = systems_.at(j)->getKeypoint(keypoints.at(i)->getTimestep());
                MatrixXd Q_t_j = MatrixXd::Zero(systems_.at(j)->getNbQVar(), systems_.at(j)->getNbQVar());

                if (kp != nullptr) {
                    Q_t_j = kp->getPrecision();
                }
                Q_t.block(idx, idx, systems_.at(j)->getNbQVar(), systems_.at(j)->getNbQVar()) = Q_t_j;
                idx += systems_.at(j)->getNbQVar();
            }

            Q.block(i * nb_Q_var_, i * nb_Q_var_, nb_Q_var_, nb_Q_var_) = Q_t;
        }

        return Q;
    }
}

void SequentialSystem::reset() {
    for (auto sys : systems_) {
        sys->reset();
    }
}

}  // namespace sys
}  // namespace ilqr_planner
