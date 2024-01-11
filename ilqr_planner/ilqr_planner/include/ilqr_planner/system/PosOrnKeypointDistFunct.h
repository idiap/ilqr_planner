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

class PosOrnKeypointDistFunct : public PosOrnKeypoint {
public:
    PosOrnKeypointDistFunct(const Eigen::VectorXd& position,
                            const Eigen::VectorXd& orientation,
                            const Eigen::MatrixXd& precision,
                            const double& pos_radius,
                            const Eigen::Vector3d& orn_thresh,
                            const int& timestep)
        : PosOrnKeypoint(position, orientation, precision, timestep), pos_radius_(pos_radius), orn_thresh_(orn_thresh) {}

    PosOrnKeypointDistFunct(const Eigen::VectorXd& position,
                            const Eigen::VectorXd& dposition,
                            const Eigen::VectorXd& orientation,
                            const Eigen::VectorXd& dorientation,
                            const Eigen::MatrixXd& precision,
                            const double& pos_radius,
                            const Eigen::Vector3d& orn_thresh,
                            const int& timestep)
        : PosOrnKeypoint(position, dposition, orientation, dorientation, precision, timestep), pos_radius_(pos_radius), orn_thresh_(orn_thresh) {}

    virtual Eigen::VectorXd diff(const Eigen::VectorXd& state) const override;

protected:
    double pos_radius_;
    Eigen::Vector3d orn_thresh_;
};

}  // namespace sys
}  // namespace ilqr_planner
