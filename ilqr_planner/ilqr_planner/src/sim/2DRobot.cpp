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