// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "ilqr_planner/sim/SimulationInterface.h"
#include "ilqr_planner/system/Keypoint.h"

#include <eigen3/Eigen/Dense>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace ilqr_planner {
namespace sys {

/**
 * Abstract class representing something that we want to control
 *
 * This abstract class contains all the methods needed by the AL-ILQR algorithm. To perform the algorithm, the "thing" you want to control
 * need to be a child of this class and implement all the methods.
 */
class System {
public:
    System() {}

    System(const std::shared_ptr<sim::SimulationInterface>& r,
           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
           const Eigen::VectorXd& RtDiag,
           const Eigen::VectorXd& qMax,
           const Eigen::VectorXd& qMin,
           const int& horizon,
           const int& nbDeriv,
           const std::vector<std::string>& allowed_kp_tags);

    System(const std::shared_ptr<sim::SimulationInterface>& r,
           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
           const Eigen::VectorXd& RtDiag,
           const Eigen::VectorXd& qMax,
           const Eigen::VectorXd& qMin,
           const Eigen::VectorXd& dqMax,
           const Eigen::VectorXd& dqMin,
           const int& horizon,
           const int& nbDeriv,
           const std::vector<std::string>& allowed_kp_tags);

    System(const std::shared_ptr<sim::SimulationInterface>& r,
           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
           const Eigen::VectorXd& RtDiag,
           const int& horizon,
           const int& nbDeriv,
           const std::vector<std::string>& allowed_kp_tags);

    virtual ~System() {}

    /**
     * Forward pass: xkp1 = A * xk + B * uk
     *
     * Return xkp1,f(xkp1),dA, dB, J (in most cases dA & dB == A & B)
     */
    virtual std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> forwardPass(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k) = 0;

    /**
     * Compute the difference between two states
     * In most cases, the difference will simply be the Euclidean distance
     */
    virtual Eigen::VectorXd diff(const Eigen::VectorXd& actual_state, int k);

    /**
     * Compute the difference between x and mu in batch form
     */
    Eigen::VectorXd diffBatch(const Eigen::VectorXd& actual_states);

    /**
     * Perform a forward pass and control the kinematics limits of the robot (q and dq)
     */
    virtual std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
    forwardPassWithLimits(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k);

    /**
     * Perform the forard pass in batch form
     */
    virtual std::tuple<Eigen::VectorXd, Eigen::VectorXd, std::vector<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>>> fpBatch(const Eigen::VectorXd& u);

    /**
     * Final state cost
     */
    virtual Eigen::VectorXd cost_F(const Eigen::VectorXd& xk);

    /**
     * d/dx (cost_F)
     */
    virtual Eigen::VectorXd cost_F_x(const Eigen::VectorXd& xk);

    /**
     * d/dx^2 (const_F)
     */
    virtual Eigen::MatrixXd cost_F_xx(const Eigen::VectorXd& xk);

    /**
     * Cost to go
     */
    virtual Eigen::VectorXd cost(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0);

    /**
     * Gradients and hessians of cost
     */
    virtual Eigen::VectorXd cost_x(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0);
    virtual Eigen::VectorXd cost_u(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0);
    virtual Eigen::MatrixXd cost_ux(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0);
    virtual Eigen::MatrixXd cost_uu(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0);
    virtual Eigen::MatrixXd cost_xu(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0);
    virtual Eigen::MatrixXd cost_xx(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0);

    /**
     * Return the current state of the system
     */
    virtual Eigen::VectorXd getState() = 0;

    /**
     * Return the task space position and the Jacobian of the current state
     */
    virtual std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getFxJac() = 0;

    /**
     * Return the task space position and the Jacobian of the desired state
     */
    virtual std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getFxJac(Eigen::VectorXd xk);

    virtual Eigen::VectorXd getMuVector(bool sparse = false);
    virtual Eigen::MatrixXd getQMatrix(bool sparse = false);
    Eigen::MatrixXd getRt() { return R; }

    int getNbStateVar() { return nb_state_var_; }
    int getNbCtrlVar() { return nb_ctrl_var_; }
    int getNbTargetVar() { return nb_target_var_; }
    int getNbQVar() { return nb_Q_var_; }
    int getHorizon() { return horizon_; }
    int getNbDeriv() { return nb_deriv_; }

    std::shared_ptr<Keypoint> getKeypoint(int k);

    void checkKeypoints();

    /**
     * Return the initial state of the system
     */
    virtual Eigen::VectorXd getInitState();
    virtual Eigen::VectorXd getInitFoXState();

    /**
     * Reset the system to the initial state
     */
    virtual void reset() = 0;

    std::shared_ptr<sim::SimulationInterface> r;
    std::vector<std::shared_ptr<Keypoint>> keypoints;

    std::vector<int> getKpIndexes();

protected:
    /**
     * Initialize the system
     */
    virtual void init();

    Eigen::VectorXd x0_;
    Eigen::VectorXd f_x0_;

    int nb_state_var_;
    int nb_target_var_;
    int nb_ctrl_var_;
    int nb_Q_var_;
    int horizon_;
    int nb_deriv_;

    bool limits_set_;
    Eigen::VectorXd state_max_, state_min_;
    Eigen::VectorXi joint_limits_weight_;

    double penalty_;  // Penalty in case of kinematics constraints violation

    Eigen::MatrixXd R;
    std::map<int, std::shared_ptr<Keypoint>> keypoints_map_;

private:
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> inspectJointLimit(Eigen::VectorXd xk);
    const std::vector<std::string> EXPECTED_KP_TAGS_;
};
}  // namespace sys
}  // namespace ilqr_planner
