// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

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
             const Eigen::VectorXd& transform_rpy,
             const Eigen::VectorXd& transform_xyz,
             const bool& is_path);
    KDLRobot(const std::string& urdf, const std::string& baseFrame, const std::string& tipFrame, const Eigen::VectorXd& q, const Eigen::VectorXd& dq)
        : KDLRobot(urdf, baseFrame, tipFrame, q, dq, Eigen::VectorXd::Zero(3), Eigen::VectorXd::Zero(3), true) {}
    KDLRobot(const std::string& urdf,
             const std::string& baseFrame,
             const std::string& tipFrame,
             const Eigen::VectorXd& q,
             const Eigen::VectorXd& dq,
             const Eigen::VectorXd& transform_rpy,
             const Eigen::VectorXd& transform_xyz)
        : KDLRobot(urdf, baseFrame, tipFrame, q, dq, transform_rpy, transform_xyz, true) {}

    void updateKinematics() override;

protected:
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
