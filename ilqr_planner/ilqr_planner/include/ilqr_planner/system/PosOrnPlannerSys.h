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
#include <eigen3/Eigen/Sparse>
#include <string>
#include "ilqr_planner/sim/SimulationInterface.h"
#include "ilqr_planner/system/System.h"

namespace ilqr_planner {
namespace sys {
class PosOrnPlannerSys : public System {
public:
    PosOrnPlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                     const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                     const Eigen::VectorXd& RtDiag,
                     const Eigen::VectorXd& qMax,
                     const Eigen::VectorXd& qMin,
                     const Eigen::VectorXd& dqMax,
                     const Eigen::VectorXd& dqMin,
                     int horizon,
                     int nbDeriv,
                     double dt);

    PosOrnPlannerSys(const std::shared_ptr<sim::SimulationInterface>& r,
                     const std::vector<std::shared_ptr<Keypoint>>& keypoints,
                     const Eigen::VectorXd& RtDiag,
                     const Eigen::VectorXd& qMax,
                     const Eigen::VectorXd& qMin,
                     int horizon,
                     int nbDeriv,
                     double dt);

    PosOrnPlannerSys(const std::shared_ptr<sim::SimulationInterface>& r, const std::vector<std::shared_ptr<Keypoint>>& keypoints, const Eigen::VectorXd& RtDiag, int horizon, int nbDeriv, double dt);

    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> forwardPass(const Eigen::VectorXd& xk, const Eigen::VectorXd& uk, int k) override;
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> getFxJac() override;

    Eigen::VectorXd getState() override;
    void reset() override;

private:
    void localInit(double dt);

    double dt;
    Eigen::VectorXd q0, dq0;

    const std::string EXPECTED_KP_TAG = "POS_ORN";
};
}  // namespace sys
}  // namespace ilqr_planner