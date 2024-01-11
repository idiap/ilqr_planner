// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "ilqr_planner/sim/SimulationInterface.h"
#include "ilqr_planner/system/System.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

namespace ilqr_planner {
namespace sys {
class PosOrnTimePlannerSys : public System {
public:
    PosOrnTimePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                         const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                         const Eigen::VectorXd& RtDiag,
                         const Eigen::VectorXd& qMax,
                         const Eigen::VectorXd& qMin,
                         const Eigen::VectorXd& dqMax,
                         const Eigen::VectorXd& dqMin,
                         int horizon,
                         int nbDeriv);

    PosOrnTimePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                         const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                         const Eigen::VectorXd& RtDiag,
                         const Eigen::VectorXd& qMax,
                         const Eigen::VectorXd& qMin,
                         int horizon,
                         int nbDeriv);

    PosOrnTimePlannerSys(const std::shared_ptr<sim::SimulationInterface>& r, const std::vector<std::shared_ptr<Keypoint>>& keypoints, const Eigen::VectorXd& RtDiag, int horizon, int nbDeriv);

    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> forwardPass(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k) override;
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getFxJac() override;
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getFxJac(Eigen::VectorXd xk) override;

    Eigen::VectorXd getState() override;

    void reset() override;

private:
    void localInit();
    Eigen::VectorXd q0_, dq0_;
};
}  // namespace sys
}  // namespace ilqr_planner
