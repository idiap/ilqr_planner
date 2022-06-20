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
#include "ilqr_planner/system/Keypoint.h"

namespace ilqr_planner {
namespace sys {

class PosOrnKeypoint : public Keypoint {
public:
    PosOrnKeypoint(const Eigen::VectorXd& position, const Eigen::VectorXd& orientation, const Eigen::MatrixXd& precision, const int& timestep)
        : Keypoint(timestep, KpType::FIRST_ORDER, "POS_ORN"), position_(position), orientation_(orientation), precision_(precision), state_size_(7) {}

    PosOrnKeypoint(const Eigen::VectorXd& position,
                   const Eigen::VectorXd& dposition,
                   const Eigen::VectorXd& orientation,
                   const Eigen::VectorXd& dorientation,
                   const Eigen::MatrixXd& precision,
                   const int& timestep)
        : Keypoint(timestep, KpType::SECOND_ORDER, "POS_ORN"),
          position_(position),
          dposition_(dposition),
          orientation_(orientation),
          dorientation_(dorientation),
          precision_(precision),
          state_size_(14) {}

    Eigen::VectorXd getPosition() const { return position_; }

    Eigen::VectorXd getOrientation() const { return orientation_; }

    virtual Eigen::MatrixXd getPrecision() const override { return precision_; }

    virtual Eigen::VectorXd diff(const Eigen::VectorXd& state) const override;

    virtual Eigen::VectorXd getState() const override;

protected:
    Eigen::VectorXd position_;
    Eigen::VectorXd orientation_;

    Eigen::VectorXd dposition_;
    Eigen::VectorXd dorientation_;

    Eigen::MatrixXd precision_;
    int state_size_;
};

}  // namespace sys
}  // namespace ilqr_planner