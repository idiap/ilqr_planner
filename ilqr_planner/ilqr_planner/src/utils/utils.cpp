// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/utils/utils.h"

using namespace Eigen;

namespace ilqr_planner {
/**
 * Helper functions to serialize eigen object.
 * It is usefull if you want to perform record & replay
 */
namespace EigenSerialize {

/**
 * Save a list of VectorXd to a CSV.
 */
bool save(const std::vector<VectorXd>& states, std::string filename) {
    IOFormat saveFormat(4, 0, "", ",", "", "", "", "");
    std::ofstream f;
    f.open(filename);

    for (auto st : states) {
        f << st.format(saveFormat) << "\n";
    }
    f.close();
    return true;
}

bool save(const std::vector<MatrixXd>& states, std::string filename) {
    IOFormat saveFormat(4, 0, ",", "\n", "", "", "", "");
    std::ofstream f;
    f.open(filename);

    for (auto st : states) {
        f << st.format(saveFormat) << "\n";
        f << "=================================== \n";
    }
    f.close();
    return true;
}

bool save(const MatrixXd& A, std::string filename) {
    IOFormat saveFormat(4, 0, ",", "\n", "", "", "", "");
    std::ofstream f;
    f.open(filename);
    f << A.format(saveFormat) << "\n";
    f.close();
    return true;
}
bool save(const VectorXd& b, std::string filename) {
    IOFormat saveFormat(4, 0, "", ",", "", "", "", "");
    std::ofstream f;
    f.open(filename);
    f << b.format(saveFormat) << "\n";
    f.close();
    return true;
}
}  // namespace EigenSerialize

MatrixXd computeJacPseudoInverse(const MatrixXd& J, const MatrixXd& Minv) {
    // MatrixXd W = MatrixXd::Identity(7,7);

    auto Jinv = Minv * J.transpose() * (J * Minv * J.transpose()).inverse();
    // auto Jinv = W * J.transpose() * (J * W * J.transpose()).inverse();
    return Jinv;
}

std::vector<double> eigenVectorToStdVector(const VectorXd& u) {
    std::vector<double> res(u.data(), u.data() + u.rows());
    return res;
}

std::vector<double> eigenMatrixToStdVector(const MatrixXd& u) {
    std::vector<double> res(u.data(), u.data() + u.rows() * u.cols());
    return res;
}
}  // namespace ilqr_planner
