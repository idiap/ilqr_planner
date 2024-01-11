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

struct Constraint {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
};

class AL_ILQR {
    /**
     * Iterative LQR with Augmented Lagrangian
     */
public:
    /**
     *  Constructor
     *      s <const std::shared_ptr<sys::System> &> pointer to the abstraction of the object that we want to control
     *      inequality: Set of inequality constraints
     *      initLambda: Initial value of Lagrange multipliers
     */
    AL_ILQR(const std::shared_ptr<sys::System>& s, const std::vector<Constraint>& inequality, const std::vector<Eigen::VectorXd>& initLambda);

    /**
     * Solve the AL-ILQR problem, return X,U
     *      U0: Initial control commands (0 a good choice)
     *      nb_iter: Number of iteration
     *      horizon: T of the system
     *      lag_update_step: The Lagrange multipliers update frequency
     *      penalty: "weight" of the constraints
     *      scaling_factor: If you want evolve the weight of constraints
     *      line_search: If true line search is performed @ each iteration
     *      early_stop: If true stop optimization when cost is not moving anymore.
     */
    std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> solve(const std::vector<Eigen::VectorXd>& U0,
                                                                                                               int nb_iter,
                                                                                                               int lag_update_step,
                                                                                                               double penalty,
                                                                                                               double scaling_factor,
                                                                                                               bool line_search = true,
                                                                                                               bool early_stop = false,
                                                                                                               CallBackMessage* cb = nullptr);

private:
    /**
     * Compute the loss @ time k with respect of the constraitns
     */
    Eigen::VectorXd augmentedLossK(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k, const Eigen::VectorXd& lambdak, const Eigen::VectorXd& Ck, const Eigen::MatrixXd& Ik);

    /**
     * Compute the constraints at time k
     */
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd> constraints(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k);

    std::shared_ptr<sys::System> s;
    std::vector<Constraint> inequality;
    std::vector<Eigen::VectorXd> multipliers;
};
}  // namespace solver
}  // namespace ilqr_planner
