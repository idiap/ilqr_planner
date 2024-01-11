// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

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
