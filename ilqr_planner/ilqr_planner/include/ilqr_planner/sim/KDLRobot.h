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
#include <string>

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>

#include <memory>

#include "ilqr_planner/sim/SimulationInterface.h"

namespace ilqr_planner {
namespace sim {
class KDLRobot : public SimulationInterface {
public:
    KDLRobot(const std::string& urdf,
             const std::string& baseFrame,
             const std::string& tipFrame,
             const Eigen::VectorXd& q,
             const Eigen::VectorXd& dq,
             const Eigen::VectorXd& transform_rpy = Eigen::VectorXd::Zero(3),
             const Eigen::VectorXd& transform_xyz = Eigen::VectorXd::Zero(3));
    void updateKinematics() override;

private:
    bool initialize(const KDL::Chain& chain);

    KDL::Chain chain_;
    std::shared_ptr<KDL::ChainJntToJacSolver> jacobianSolver_;
    std::shared_ptr<KDL::ChainFkSolverPos_recursive> fkSolverPos_;

    KDL::JntArray positions_;
    KDL::Jacobian jacobian_;
    KDL::Frame pose_;
};
}  // namespace sim
}  // namespace ilqr_planner