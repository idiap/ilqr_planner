// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

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

    virtual bool isPartOfEuclideanSpace() const { return false; };

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
