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

#include "ilqr_planner/system/SpacetimeKeypoint.h"

namespace ilqr_planner {
namespace sys {

Eigen::VectorXd SpacetimeKeypoint::getState() const {
    Eigen::VectorXd state = PosOrnKeypoint::getState();
    Eigen::VectorXd state_augmented = Eigen::VectorXd::Zero(state.rows() + 1);
    state_augmented << state, continuous_time_;
    return state_augmented;
}

Eigen::VectorXd SpacetimeKeypoint::diff(const Eigen::VectorXd& state) const {
    Eigen::VectorXd residual = PosOrnKeypoint::diff(state.head(state.rows() - 1));
    Eigen::VectorXd residual_augmented = Eigen::VectorXd::Zero(residual.rows() + 1);
    residual_augmented << residual, continuous_time_ - state(state.rows() - 1);
    return residual_augmented;
}

}  // namespace sys
}  // namespace ilqr_planner
