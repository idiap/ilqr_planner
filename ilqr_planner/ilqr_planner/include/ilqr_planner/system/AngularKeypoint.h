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

class AngularKeypoint : public Keypoint {
public:
    AngularKeypoint(const Eigen::VectorXd& position, const Eigen::MatrixXd& precision, const int& timestep)
        : Keypoint(timestep, KpType::FIRST_ORDER, "JNT"), position_(position), precision_(precision), state_size_(position.size()) {}

    AngularKeypoint(const Eigen::VectorXd& position, const Eigen::VectorXd& dposition, const Eigen::MatrixXd& precision, const int& timestep)
        : Keypoint(timestep, KpType::SECOND_ORDER, "JNT"), position_(position), dposition_(dposition), precision_(precision), state_size_(position.size() * 2) {}

    Eigen::VectorXd getPosition() const { return position_; }

    virtual Eigen::MatrixXd getPrecision() const override { return precision_; }

    virtual Eigen::VectorXd diff(const Eigen::VectorXd& state) const override;

    virtual Eigen::VectorXd getState() const override;

    virtual bool isPartOfEuclideanSpace() const { return true; };

protected:
    Eigen::VectorXd position_;

    Eigen::VectorXd dposition_;

    Eigen::MatrixXd precision_;
    int state_size_;
};

}  // namespace sys
}  // namespace ilqr_planner
