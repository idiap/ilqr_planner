// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <eigen3/Eigen/Dense>

#include "ilqr_planner/system/AngularKeypoint.h"

namespace ilqr_planner {
namespace sys {
class AngularTimeKeypoint : public AngularKeypoint {
public:
    AngularTimeKeypoint(const Eigen::VectorXd& position, const Eigen::MatrixXd& precision, const double& continuous_time, const int& timestep)
        : AngularKeypoint(position, precision, timestep), continuous_time_(continuous_time) {
        TAG_ = "JNT_TIME";
    }

    AngularTimeKeypoint(const Eigen::VectorXd& position, const Eigen::VectorXd& dposition, const Eigen::MatrixXd& precision, const double& continuous_time, const int& timestep)
        : AngularKeypoint(position, dposition, precision, timestep), continuous_time_(continuous_time) {
        TAG_ = "JNT_TIME";
    }

    double getContinuousTime() { return continuous_time_; }

    virtual Eigen::VectorXd diff(const Eigen::VectorXd& state) const override;
    virtual Eigen::VectorXd getState() const override;
    virtual bool isPartOfEuclideanSpace() const { return true; };

protected:
    double continuous_time_;
};
}  // namespace sys
}  // namespace ilqr_planner
