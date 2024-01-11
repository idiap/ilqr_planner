// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/system/AngularKeypoint.h"
#include "ilqr_planner/utils/sd.h"

namespace ilqr_planner {
namespace sys {

Eigen::VectorXd AngularKeypoint::getState() const {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(state_size_);

    if (type_ == KpType::FIRST_ORDER)
        state << position_;
    else if (type_ == KpType::SECOND_ORDER)
        state << position_, dposition_;

    return state;
}

Eigen::VectorXd AngularKeypoint::diff(const Eigen::VectorXd& state) const {
    Eigen::VectorXd residual = AngularKeypoint::getState() - state;
    return residual;
}

}  // namespace sys
}  // namespace ilqr_planner
