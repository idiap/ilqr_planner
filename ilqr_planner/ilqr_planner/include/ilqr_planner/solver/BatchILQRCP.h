// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <eigen3/Eigen/Dense>
#include <functional>
#include <memory>
#include <tuple>
#include "ilqr_planner/system/System.h"
#include "ilqr_planner/utils/CallbackMessage.h"

namespace ilqr_planner {
namespace solver {

class BatchILQRCP {
    /**
     * Batch Iterative LQR with the use of control primitives
     */
public:
    /**
     * Constructor
     *  s <const std::shared_ptr<sys::System> &>: pointer to the abstraction of the object that we want to control
     *  Q <MatrixXd>: Precision matrix
     *  RtDiag <VectorXd>: Diagonal of Rt
     *  Psi <MatrixXd>: CP Matrix
     */
    BatchILQRCP(const std::shared_ptr<sys::System>& s, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& psi);
    BatchILQRCP(const std::shared_ptr<sys::System>& s, const Eigen::MatrixXd& psi);

    /**
     * Solve the problem for <nb_iter> iterations, publish iteration callback inside <cb>
     */
    Eigen::VectorXd solve(int nb_iter, const Eigen::VectorXd& u0, bool early_stop = false, CallBackMessage* cb = nullptr);

private:
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> buildSuJL(const std::vector<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>>& ABJLs);
    Eigen::VectorXd truncateStates(const Eigen::VectorXd& states, const int& state_size);
    Eigen::MatrixXd buildL(const std::vector<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>>& ABJLs);
    Eigen::MatrixXd buildLc(const std::vector<Eigen::MatrixXd>& Ls);

    Eigen::MatrixXd PSI, R;
    Eigen::MatrixXd Q;
    Eigen::VectorXd mu;

    std::shared_ptr<sys::System> s;
    std::vector<int> vp_indexes;
};
}  // namespace solver
}  // namespace ilqr_planner
