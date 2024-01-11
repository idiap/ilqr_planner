// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

#include "ilqr_planner/sim/SimulationInterface.h"

namespace ilqr_planner {
namespace sim {

class Robot2D : public SimulationInterface {
public:
    Robot2D(const Eigen::VectorXd& lengths, const Eigen::VectorXd& default_q);
    Eigen::VectorXd fkine(const Eigen::VectorXd& q);
    Eigen::VectorXd fkine();
    void updateKinematics() override;

private:
    Eigen::VectorXd lengths;
};
}  // namespace sim
}  // namespace ilqr_planner
