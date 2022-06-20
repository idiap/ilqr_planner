/**
    This package provides a C++ iLQR library that comes with its python bindings.
    It allows you to solve iLQR optimization problem on any robot as long as you
    provide an [URDF file](http://wiki.ros.org/urdf/Tutorials) describing the
    kinematics chain of the robot. For debugging purposes it also provide a 2D
    planar robots class that you can use. You can also apply a spatial
    transformation to compute robot task space information in the base frame of
    your choice (e.g. object frame).

    Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
    Written by Jeremy Maceiras <jeremy.maceiras@idiap.ch>

    This file is part of ilqr_planner.

    ilqr_planner is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    ilqr_planner is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ilqr_planner. If not, see <http://www.gnu.org/licenses/>.
*/

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
           int horizon,
           int nbDeriv);

    System(const std::shared_ptr<sim::SimulationInterface>& r,
           const std::vector<std::shared_ptr<Keypoint>>& keypoints,
           const Eigen::VectorXd& RtDiag,
           const Eigen::VectorXd& qMax,
           const Eigen::VectorXd& qMin,
           const Eigen::VectorXd& dqMax,
           const Eigen::VectorXd& dqMin,
           int horizon,
           int nbDeriv);

    System(const std::shared_ptr<sim::SimulationInterface>& r, const std::vector<std::shared_ptr<Keypoint>>& keypoints, const Eigen::VectorXd& RtDiag, int horizon, int nbDeriv);

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
    Eigen::MatrixXd getRt() { return this->R; }

    int getNbStateVar() { return this->nbStateVar; }
    int getNbCtrlVar() { return this->nbCtrlVar; }
    int getNbTargetVar() { return this->nbTargetVar; }
    int getNbQVar() { return this->nbQVar; }
    int getHorizon() { return this->horizon; }
    int getNbDeriv() { return this->nbDeriv; }

    std::shared_ptr<Keypoint> getKeypoint(int k);

    void checkKeypoints(const std::string& expected_tag);

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

    Eigen::VectorXd x0;
    Eigen::VectorXd f_x0;

    int nbStateVar;
    int nbTargetVar;
    int nbCtrlVar;
    int nbQVar;
    int horizon;
    int nbDeriv;

    bool limitsSet;
    Eigen::VectorXd state_max, state_min;
    Eigen::VectorXi joint_limits_weight;

    double penalty;  // Penalty in case of kinematics constraints violation

    Eigen::MatrixXd R;
    std::map<int, std::shared_ptr<Keypoint>> keypoints_map;

private:
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> inspectJointLimit(Eigen::VectorXd xk);
};
}  // namespace sys
}  // namespace ilqr_planner