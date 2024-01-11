// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "ilqr_planner/sim/SimulationInterface.h"
#include "ilqr_planner/system/System.h"

#include <eigen3/Eigen/Dense>
#include <vector>

namespace ilqr_planner {
namespace sys {

class SequentialSystem : public System {
public:
    SequentialSystem(const std::shared_ptr<sim::SimulationInterface>& r, const std::vector<std::shared_ptr<System>>& systems, const Eigen::VectorXd& RtDiag, int horizon, int nbDeriv);

    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> forwardPass(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int) override;
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getFxJac() override;

    Eigen::VectorXd getState() override;
    void reset() override;

    Eigen::VectorXd cost_F(const Eigen::VectorXd& xk) override;
    Eigen::VectorXd cost_F_x(const Eigen::VectorXd& xk) override;
    Eigen::MatrixXd cost_F_xx(const Eigen::VectorXd& xk) override;

    Eigen::VectorXd cost(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0) override;

    Eigen::VectorXd cost_x(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0) override;
    Eigen::MatrixXd cost_xx(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k = 0) override;

    Eigen::VectorXd diff(const Eigen::VectorXd& state, int k) override;

    Eigen::VectorXd getMuVector(bool sparse = false) override;
    Eigen::MatrixXd getQMatrix(bool sparse = false) override;

private:
    void localInit(const Eigen::VectorXd& RtDiag);

    std::vector<std::shared_ptr<System>> systems_;
    Eigen::VectorXd q0_, dq0_;
};
}  // namespace sys
}  // namespace ilqr_planner
