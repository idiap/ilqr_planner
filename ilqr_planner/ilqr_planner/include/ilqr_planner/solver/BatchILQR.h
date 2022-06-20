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

#include <eigen3/Eigen/Dense>
#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include "ilqr_planner/system/System.h"
#include "ilqr_planner/utils/CallbackMessage.h"

namespace ilqr_planner {
namespace solver {

class BatchILQR {
    /**
     * Batch Iterative LQR
     */
public:
    /**
     * Constructor
     *  s <const std::shared_ptr<sys::System> &>: pointer to the abstraction of the object that we want to control
     *  mu <VectorXd>: Target vector
     *  Q <MatrixXd>: Precision matrix
     */
    BatchILQR(const std::shared_ptr<sys::System>& s);
    BatchILQR(const std::shared_ptr<sys::System>& s, const Eigen::MatrixXd& Q);

    /**
     * Solve the problem for <nb_iter> iterations, publish iteration callback inside <cb>
     */
    Eigen::VectorXd solve(int nb_iter, const Eigen::VectorXd& u0, bool early_stop = false, CallBackMessage* cb = nullptr);

private:
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> buildSuJL(const std::vector<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>>& ABJLs);
    Eigen::VectorXd truncateStates(const Eigen::VectorXd& states, const int& state_size);
    Eigen::MatrixXd buildL(const std::vector<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>>& ABJLs);
    Eigen::MatrixXd buildLc(const std::vector<Eigen::MatrixXd>& Ls);

    Eigen::MatrixXd R;
    Eigen::MatrixXd Q;
    Eigen::VectorXd mu;

    std::shared_ptr<sys::System> s;

    std::vector<int> vp_indexes;
};
}  // namespace solver
}  // namespace ilqr_planner
