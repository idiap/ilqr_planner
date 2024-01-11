// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <eigen3/Eigen/Core>

namespace ilqr_planner {
int binomialCoefficients(int n, int k);
Eigen::MatrixXd buildPsiRBF(int dim, int K);
Eigen::MatrixXd buildPsiBernstein(int dim, int K);
Eigen::MatrixXd buildPsiUnitstep(int dim, int K);
Eigen::MatrixXd buildPsiSawtooth(int dim, int K);
Eigen::MatrixXd buildPsiLinear(int dim, int k);
}  // namespace ilqr_planner
