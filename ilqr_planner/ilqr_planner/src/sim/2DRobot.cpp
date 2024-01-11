// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/sim/2DRobot.h"
#include <cmath>
#include <iostream>

namespace ilqr_planner {
namespace sim {
Robot2D::Robot2D(const Eigen::VectorXd& lengths, const Eigen::VectorXd& default_q) {
    this->dof = default_q.rows();
    this->lengths = lengths;
    this->nbCarDim = 2;
    this->q = default_q;
    this->dq = Eigen::VectorXd::Zero(this->dof);
    this->ddq = Eigen::VectorXd::Zero(this->dof);
    this->x = Eigen::VectorXd::Zero(2);
    this->dx = Eigen::VectorXd::Zero(2);
    this->ornQuat = Eigen::VectorXd::Zero(4);
    ornQuat(0) = 1;

    this->updateKinematics();
}

Eigen::VectorXd Robot2D::fkine(const Eigen::VectorXd& q) {
    Eigen::VectorXd xi = Eigen::VectorXd::Zero(2);
    for (int i = 0; i < this->dof; i++) {
        double qi = q(i);

        xi(0) += this->lengths(i) * cos(qi);
        xi(1) += this->lengths(i) * sin(qi);
    }

    return xi;
}

Eigen::VectorXd Robot2D::fkine() {
    return this->fkine(this->q);
}

void Robot2D::updateKinematics() {
    // Forward pass
    this->x = this->fkine();

    // Compute Jacobian
    Eigen::MatrixXd Jt = Eigen::MatrixXd::Zero(2, this->dof);
    double dq = M_PI * 1e-3;

    for (int i = 0; i < this->dof; i++) {
        Eigen::VectorXd qi(this->q);
        qi(i) += dq;

        auto new_pos = this->fkine(qi);
        auto old_pos = this->fkine();

        Jt(0, i) = (new_pos(0) - old_pos(0)) / dq;
        Jt(1, i) = (new_pos(1) - old_pos(1)) / dq;
    }

    Eigen::MatrixXd Jr = Eigen::MatrixXd::Zero(2, this->dof);

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(4, this->dof);
    J << Jt, Jr;
    this->Jac = J;

    this->dx = this->Jt() * this->dq;
}
}  // namespace sim
}  // namespace ilqr_planner
