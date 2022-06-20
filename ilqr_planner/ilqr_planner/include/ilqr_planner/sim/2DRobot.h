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