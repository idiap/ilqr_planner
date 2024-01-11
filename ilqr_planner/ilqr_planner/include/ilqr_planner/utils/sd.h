// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>

namespace ilqr_planner {
namespace Sd {

/**
 * Ensure that the vector <x> has unit norm.
 */
inline Eigen::VectorXd toUnitNorm(Eigen::VectorXd x) {
    return x / x.norm();
}

inline Eigen::MatrixXd dQuatToDxJac(Eigen::VectorXd q) {
    Eigen::MatrixXd J(3, 4);
    J << -q(1), q(0), -q(3), q(2), -q(2), q(3), q(0), -q(1), -q(3), -q(2), q(1), q(0);
    return J;
}

/**
 * Transform a point on the tangent space of <base> to the Sd Sphere
 */
inline Eigen::VectorXd expMap(Eigen::VectorXd base, Eigen::VectorXd u) {
    base = toUnitNorm(base);

    auto norm_u = u.norm();

    // Avoid division by 0
    if (norm_u == 0) {
        return base;
    }

    return toUnitNorm(base * cos(norm_u) + u / norm_u * sin(norm_u));
}

/**
 * Distance of the geodesic between <x> and <y>
 */
inline double distance(Eigen::VectorXd x, Eigen::VectorXd y) {
    double dist = x.dot(y);

    if (dist > 1) {
        dist = 1;
    } else if (dist < -1) {
        dist = -1;
    }

    double ac = acos(dist);
    if (dist < 0) {
        ac -= M_PI;
    }
    return ac;
}

/**
 * Project a point on the Sd Sphere <y> to the tangent space of <base>.
 */
inline Eigen::VectorXd logMap(Eigen::VectorXd base, Eigen::VectorXd y) {
    if (base.isZero() || y.isZero()) {
        return Eigen::VectorXd::Zero(base.rows());
    }

    base = toUnitNorm(base);
    y = toUnitNorm(y);

    auto temp = y - base.transpose().dot(y) * base;

    if (temp.norm() == 0) {
        return Eigen::VectorXd::Zero(base.rows());
    }

    return distance(base, y) * temp / temp.norm();
}

/**
 * Parallel transport of vector <v> relying on the tangent space of <base1> to the tangent space of <base2>.
 */
inline Eigen::VectorXd transport(Eigen::VectorXd v, Eigen::VectorXd base1, Eigen::VectorXd base2) {
    if (base1.isZero() || base2.isZero()) {
        return v;
    }

    double dist_square = pow(distance(base1, base2), 2);

    // If base1 and base2 are too close, distance can be cropped to 0, to avoid division by 0 error, we return just v
    if (dist_square == 0)
        return v;

    return v - (logMap(base1, base2).dot(v) / dist_square) * (logMap(base1, base2) + logMap(base2, base1));
}
}  // namespace Sd
}  // namespace ilqr_planner
