// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/solver/lqt.h"
#include <cmath>

#include <iostream>
using namespace Eigen;

namespace ilqr_planner {
namespace solver {

LQT::LQT(const MatrixXd& A, const MatrixXd& B, const std::vector<MatrixXd>& Qs, const VectorXd& states, float rfactor, int nb_deriv) : mu(states), A(A), B(B), Qs(Qs) {
    auto Rt = MatrixXd::Identity(B.cols(), B.cols()) * pow(rfactor, nb_deriv);
    this->Rt = Rt;

    this->rfactor = rfactor;
    this->nb_deriv = nb_deriv;
    this->nb_state_var = A.cols();
    this->nb_ctrl_var = B.cols();
    this->nb_states = states.size() / this->nb_state_var;
}

LQT::LQT() {}

void LQT::solveDP() {
    int nb_states = this->nb_states;
    int nb_state_var = this->nb_state_var;

    // Solve LQT with dynamic programming according to : Learning Control, S. Calinon, D. Lee
    // Initial condition to solve LQT in DP
    auto PT = this->Qs.back();
    auto dT = VectorXd::Zero(nb_state_var, 1);

    this->Ps.push_back(PT);
    this->ds.push_back(dT);

    for (int i = nb_states - 2; i >= 0; i--) {
        auto Qt = this->Qs.at(i);
        auto Ptp1 = this->Ps.back();
        auto dtp1 = this->ds.back();

        auto Pt = Qt - this->A.transpose() * (Ptp1 * this->B * (this->B.transpose() * Ptp1 * this->B + this->Rt).inverse() * this->B.transpose() * Ptp1 - Ptp1) * this->A;
        auto dt = (this->A.transpose() - this->A.transpose() * Ptp1 * this->B * (this->B.transpose() * Ptp1 * this->B + this->Rt).inverse() * this->B.transpose()) *
                  (Ptp1 * (A * this->mu.segment(i * nb_state_var, nb_state_var) - mu.segment((i + 1) * nb_state_var, nb_state_var)) + dtp1);

        this->Ps.push_back(Pt);
        this->ds.push_back(dt);
    }
}

void LQT::buildSystemMatrices() {
    MatrixXd Su = MatrixXd::Zero(this->nb_state_var * this->nb_states, this->nb_ctrl_var * (this->nb_states - 1));
    MatrixXd Sx = MatrixXd::Zero(this->nb_state_var * this->nb_states, this->nb_state_var);
    MatrixXd Q = MatrixXd::Zero(this->nb_state_var * this->nb_states, this->nb_state_var * this->nb_states);
    this->R = MatrixXd::Identity((this->nb_states - 1) * this->nb_ctrl_var, (this->nb_states - 1) * this->nb_ctrl_var) * pow(this->rfactor, this->nb_deriv);

    // Initialization for Sx,Su and Q
    auto M = this->B;
    Sx.block(0, 0, this->nb_state_var, this->nb_state_var) = MatrixXd::Identity(this->nb_state_var, this->nb_state_var);
    Q.block(0, 0, this->nb_state_var, this->nb_state_var) = this->Qs.at(0);

    for (int i = 1; i < this->nb_states; i++) {
        Sx.block(i * this->nb_state_var, 0, this->nb_state_var, this->nb_state_var) = Sx.block((i - 1) * this->nb_state_var, 0, this->nb_state_var, this->nb_state_var) * this->A;
        Su.block(i * this->nb_state_var, 0, M.rows(), M.cols()) = M;
        Q.block(i * this->nb_state_var, i * this->nb_state_var, this->nb_state_var, this->nb_state_var) = this->Qs.at(i);

        MatrixXd newM(M.rows(), M.cols() + B.cols());
        newM << (A * M), this->B;
        M = newM;
    }

    this->Sx = Sx;
    this->Su = Su;
    this->Q = Q;
}

void LQT::solveLinAl() {
    this->buildSystemMatrices();
    auto Su = this->Su;
    auto Sx = this->Sx;
    auto R = this->R;
    auto Q = this->Q;
    auto mu = this->mu;
    this->u = (Su.transpose() * Q * Su + R).inverse() * Su.transpose() * Q * (mu - Sx * mu.head(this->nb_state_var));
}

int LQT::getNbStates() {
    return this->nb_states;
}

VectorXd LQT::getCommand(int timestep) {
    if (this->u.rows() == 0) {
        throw std::runtime_error("solveLinal() or solveQP() first");
    }
    return this->u.segment(timestep * this->nb_ctrl_var, this->nb_ctrl_var);
}

VectorXd LQT::getCommand(int timestep, const VectorXd& curr_state) {
    timestep += 1;  // Because we want to reach state t+1 and not t (assume that we are already in state t)

    auto timestep2 = this->nb_states - timestep - 1;  // Ks & fs are on reverse order...

    if (this->Ps.size() == 0) {
        throw std::runtime_error("solveDP() first");
    }

    auto Pt = this->Ps.at(timestep2);
    auto dt = this->ds.at(timestep2);

    auto Kt = (this->B.transpose() * Pt * this->B + this->Rt).inverse() * this->B.transpose() * Pt * this->A;
    auto ft = -1 * (this->B.transpose() * Pt * this->B + this->Rt).inverse() * this->B.transpose() *
              (Pt * (this->A * this->mu.segment(timestep * nb_state_var, nb_state_var) - this->mu.segment(timestep * nb_state_var, nb_state_var)) + dt);
    VectorXd u = Kt * (this->mu.segment(timestep * nb_state_var, nb_state_var) - curr_state) + ft;

    return u;
}

VectorXd LQT::getPredictedStates() {
    if (this->u.rows() == 0) {
        throw std::runtime_error("solveLinal() or solveQP() first");
    }

    return this->Su * this->u + this->Sx * this->mu.head(this->nb_state_var);
}
}  // namespace solver
}  // namespace ilqr_planner
