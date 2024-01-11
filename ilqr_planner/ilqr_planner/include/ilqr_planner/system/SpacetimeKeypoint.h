// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

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
    virtual bool isPartOfEuclideanSpace() const { return false; };

protected:
    double continuous_time_;
};
}  // namespace sys
}  // namespace ilqr_planner
