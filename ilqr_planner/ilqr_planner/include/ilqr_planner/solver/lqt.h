// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

namespace ilqr_planner {
namespace solver {

/**
 * The purpose of this class is to solve a Linear Quadratic Tracking problem.
 * The problem can be solve by relying on:
 * - Dynamic programming u_t = K_t (mu_t - x_t) + f_t
 * - Quadratic programming (equality and inequality constraints) minimize (x-mu)^T Q (x-mu) + u^T R u subject to C*x = b & E*x <= d
 * - Matrix form u = (Su^T Q Su + R) Su^T Q (mu - Sx X0)
 * author: jmaceiras
 */
class LQT {
public:
    /**
     * Constructor of the class.
     *      A (Eigen::MatrixXd), the system matrix.
     *      B (Eigen::MatrixXd), the input matrix.
     *      Qs (std::vector of Eigen::MatrixXd), The weights corresponding to each state.
     *      states (std::vector of Eigen::VectorXd), the desired states at each timestep.
     *      rfactor (float), nb_deriv(int), used to create the R matrix.
     */
    LQT(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const std::vector<Eigen::MatrixXd>& Qs, const Eigen::VectorXd& states, float rfactor = 0.1, int nb_deriv = 2);
    LQT();

    /**
     * Return the control command in function of your time step and current state.
     * You need to call solveDP() before using this method.
     *
     * Return a Eigen::VectorXd corresponding to your desired control command.
     */
    Eigen::VectorXd getCommand(int timestep, const Eigen::VectorXd& curr_state);

    /**
     * Return the control command in funciton of the time step.
     * You need to call solveQP() or solveLinAl() before using this method.
     *
     * Return a Eigen::VectorXd corresponding to your desired control command.
     */
    Eigen::VectorXd getCommand(int timestep);

    /**
     * Sole the LQT by relying on Dynamic Programming.
     */
    void solveDP();

    /**
     * Solve the LQT by relying on Linear Algebra.
     */
    void solveLinAl();

    /**
     * Helper function to know the total number of states in the LQT.
     */
    int getNbStates();

    /**
     * Function to get the predicted states given the system matrices:
     *      \hat{x} = S^u * u + S^x * x_0
     */
    Eigen::VectorXd getPredictedStates();

private:
    /**
     * Build the system matrices (Su, Sx, Q, R, mu) needed for solveQP() & solvelinAl()
     */
    void buildSystemMatrices();

    int nb_states, nb_state_var, nb_ctrl_var;  // Derived from the shape of A,B & states given by the constructor.
    int nb_deriv;
    float rfactor;
    Eigen::MatrixXd A, B, R, Rt, Sx, Su, Q;
    Eigen::VectorXd mu, u;
    std::vector<Eigen::MatrixXd> Qs, Ps;
    std::vector<Eigen::VectorXd> ds;
};
}  // namespace solver
}  // namespace ilqr_planner
