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

    std::vector<std::shared_ptr<System>> systems;
    Eigen::VectorXd q0, dq0;
};
}  // namespace sys
}  // namespace ilqr_planner