// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <eigen3/Eigen/Dense>
#include <memory>
#include <tuple>
#include <vector>
#include "ilqr_planner/system/System.h"
#include "ilqr_planner/utils/CallbackMessage.h"

namespace ilqr_planner {
namespace solver {
class ILQRRecursive {
    /**
     * Iterative LQR
     */
public:
    /**
     *  Constructor
     *      s <const std::shared_ptr<sys::System> &> pointer to the abstraction of the object that we want to control
     */
    explicit ILQRRecursive(const std::shared_ptr<sys::System>& s);

    /**
     * Solve the iLQR problem, return X,U,K,k
     *      U0: Initial control commands (0 a good choice)
     *      nb_iter: Number of iteration
     *      line_search: If true line search is performed @ each iteration
     *      cb: Callback object to notify the user
     */
    std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>, double>
    solve(const std::vector<Eigen::VectorXd>& U0, int nb_iter, bool line_search = true, bool early_stop = false, CallBackMessage* cb = nullptr);

private:
    std::shared_ptr<sys::System> s;
};
}  // namespace solver
}  // namespace ilqr_planner
