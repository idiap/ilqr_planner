// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/system/PosOrnKeypointDistFunct.h"
#include "ilqr_planner/utils/sd.h"

namespace ilqr_planner {
namespace sys {

Eigen::VectorXd PosOrnKeypointDistFunct::diff(const Eigen::VectorXd& state) const {
    Eigen::VectorXd residual = PosOrnKeypoint::diff(state);

    // Position bounding box
    double res_pos_norm = residual.head(position_.rows()).norm();
    if (res_pos_norm <= pos_radius_) {
        residual.head(position_.rows()) = Eigen::VectorXd::Zero(position_.rows());
    } else {
        residual.head(position_.rows()) = residual.head(position_.rows()).normalized() * (res_pos_norm - pos_radius_);
    }

    // Orientation bounding box
    for (int i = 0; i < 3; i++) {
        if (abs(residual(position_.rows() + i)) <= orn_thresh_(i)) {
            residual(position_.rows() + i) = 0;
        } else {
            int sign = residual(position_.rows() + i) < 0 ? -1 : 1;
            residual(position_.rows() + i) = residual(position_.rows() + i) - sign * orn_thresh_(i);
        }
    }

    return residual;
}

}  // namespace sys
}  // namespace ilqr_planner
