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