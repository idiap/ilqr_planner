// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/utils/primitives.h"
#include <iostream>

using namespace Eigen;

namespace ilqr_planner {
int binomialCoefficients(int n, int k) {
    if (k == 0 || k == n)
        return 1;
    return binomialCoefficients(n - 1, k - 1) + binomialCoefficients(n - 1, k);
}

MatrixXd buildPsiRBF(int dim, int K) {
    VectorXd Ts = VectorXd::LinSpaced(dim, 0, dim - 1);
    double bw = ((double)dim) / K;
    double avg = bw / 2;
    auto sig = bw;

    MatrixXd psi = MatrixXd::Zero(dim, K);

    for (int i = 0; i < K; i++) {
        psi.col(i) << 1 / (2 * M_PI * sig) * exp(-1 * (Ts.array() - avg) * (Ts.array() - avg) / (2 * sig * sig));
        avg += bw;
    }

    return psi;
}

MatrixXd buildPsiBernstein(int dim, int K) {
    ArrayXd Ts = ArrayXd::LinSpaced(dim, 0, dim - 1);
    int order = K - 1;
    Ts = Ts / Ts.maxCoeff();

    MatrixXd psi = MatrixXd::Zero(dim, K);

    for (int i = 0; i < K; i++) {
        auto binom = binomialCoefficients(order, i);
        for (int j = 0; j < dim; j++) {
            psi(j, i) = binom * pow(Ts(j), i) * pow(1 - Ts(j), order - i);
        }
    }

    return psi;
}

MatrixXd buildPsiUnitstep(int dim, int K) {
    ArrayXd Ts = ArrayXd::LinSpaced(dim, 0, dim - 1);
    int bw = round(((double)dim) / K);
    int lowDim = 0;
    int highDim = bw;

    MatrixXd psi = MatrixXd::Zero(dim, K);

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < dim; j++) {
            psi(j, i) = (j >= lowDim && j < highDim) ? 1.0 / bw : 0;
        }
        lowDim += bw;
        highDim += bw;
    }

    return psi;
}

MatrixXd buildPsiSawtooth(int dim, int K) {
    ArrayXd Ts = ArrayXd::LinSpaced(dim, 0, dim - 1);
    int bw = ceil(((double)dim) / K);
    double lowDim = 0;    // Double to avoid auto cast to int
    double highDim = bw;  // Double to avoid auto cast to int

    MatrixXd psi = MatrixXd::Zero(dim, K);

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < dim; j++) {
            psi(j, i) = (j >= lowDim && j < highDim) ? ((j - lowDim) / (bw - 1) - 0.5) : 0;
        }
        lowDim += bw;
        highDim += bw;
    }

    return psi;
}

MatrixXd buildPsiLinear(int dim, int K) {
    MatrixXd PSI_f_1 = buildPsiUnitstep(dim, K);
    MatrixXd PSI_f_2 = buildPsiSawtooth(dim, K);
    MatrixXd PSI_f = MatrixXd::Zero(dim, K * 2);
    PSI_f << PSI_f_1, PSI_f_2;
    return PSI_f;
}
}  // namespace ilqr_planner
