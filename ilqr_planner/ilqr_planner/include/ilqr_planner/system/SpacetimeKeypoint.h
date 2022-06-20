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

#include "ilqr_planner/system/PosOrnKeypoint.h"

namespace ilqr_planner {
namespace sys {
class SpacetimeKeypoint : public PosOrnKeypoint {
public:
    SpacetimeKeypoint(const Eigen::VectorXd& position, const Eigen::VectorXd& orientation, const Eigen::MatrixXd& precision, const double& continuous_time, const int& timestep)
        : PosOrnKeypoint(position, orientation, precision, timestep), continuous_time_(continuous_time) {
        TAG_ = "POS_ORN_TIME";
    }

    SpacetimeKeypoint(const Eigen::VectorXd& position,
                      const Eigen::VectorXd& dposition,
                      const Eigen::VectorXd& orientation,
                      const Eigen::VectorXd& dorientation,
                      const Eigen::MatrixXd& precision,
                      const double& continuous_time,
                      const int& timestep)
        : PosOrnKeypoint(position, dposition, orientation, dorientation, precision, timestep), continuous_time_(continuous_time) {
        TAG_ = "POS_ORN_TIME";
    }

    double getContinuousTime() { return continuous_time_; }

    virtual Eigen::VectorXd diff(const Eigen::VectorXd& state) const override;
    virtual Eigen::VectorXd getState() const override;

protected:
    double continuous_time_;
};
}  // namespace sys
}  // namespace ilqr_planner