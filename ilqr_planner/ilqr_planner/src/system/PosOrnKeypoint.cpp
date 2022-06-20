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

#include "ilqr_planner/system/PosOrnKeypoint.h"
#include "ilqr_planner/utils/sd.h"

namespace ilqr_planner {
namespace sys {

Eigen::VectorXd PosOrnKeypoint::getState() const {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(state_size_);

    if (type_ == KpType::FIRST_ORDER)
        state << position_, orientation_;
    else if (type_ == KpType::SECOND_ORDER)
        state << position_, dposition_, orientation_, dorientation_;

    return state;
}

Eigen::VectorXd PosOrnKeypoint::diff(const Eigen::VectorXd& state) const {
    int residual_space_size = state_size_ - type_;

    Eigen::VectorXd residual = Eigen::VectorXd::Zero(residual_space_size);

    if (!state.isZero()) {
        residual.head(position_.rows()) = position_ - state.head(position_.rows());  // Position residual
        residual.segment(position_.rows(), orientation_.rows() - 1) =
            -2 * Sd::dQuatToDxJac(orientation_) * Sd::logMap(orientation_, state.segment(position_.rows(), orientation_.rows()));  // Orientation residual

        if (type_ == KpType::SECOND_ORDER) {
            residual.segment(position_.rows() + orientation_.rows() - 1, position_.rows()) =
                dposition_ - state.segment(position_.rows() + orientation_.rows(), position_.rows());  // dposition residual
            residual.segment(position_.rows() + orientation_.rows() - 1 + position_.rows(), orientation_.rows() - 1) =
                -2 * Sd::dQuatToDxJac(orientation_) *
                (dorientation_ - Sd::transport(state.segment(position_.rows() + orientation_.rows() + position_.rows(), orientation_.rows()), state.segment(position_.rows(), orientation_.rows()),
                                               orientation_));  // dorientation residual
        }
    }

    return residual;
}

}  // namespace sys
}  // namespace ilqr_planner