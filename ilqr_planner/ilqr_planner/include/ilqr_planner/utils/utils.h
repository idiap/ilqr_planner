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

#pragma once

#include <eigen3/Eigen/Dense>
#include <fstream>

#include <string>
#include <vector>

namespace ilqr_planner {
namespace EigenSerialize {
/**
 * Load a CSV corresponding to a list of Eigen::VectorXd.
 * e.g.: A list of joint position.
 */
template <size_t nb>
inline std::vector<Eigen::VectorXd> load(const std::string& filename) {
    std::ifstream f(filename);
    std::vector<Eigen::VectorXd> states;
    std::string line;

    while (getline(f, line)) {
        size_t pos = 0;
        int i = 0;
        std::string delimiter = ",";

        std::array<double, nb> vec;

        while ((pos = line.find(delimiter)) != std::string::npos) {
            auto token = line.substr(0, pos);
            line.erase(0, pos + delimiter.length());
            vec[i] = std::stod(token);
            i++;
        }

        vec[i] = std::stod(line);

        Eigen::Map<const Eigen::Matrix<double, nb, 1>> st(vec.data());
        states.push_back(st);
    }
    f.close();

    return states;
}

/**
 * Save a list of Eigen::VectorXd to a CSV.
 */
bool save(const std::vector<Eigen::VectorXd>& states, std::string filename);
bool save(const Eigen::MatrixXd& A, std::string filename);
bool save(const std::vector<Eigen::MatrixXd>& states, std::string filename);
bool save(const Eigen::VectorXd& b, std::string filename);
}  // namespace EigenSerialize

/**
 * Compute the time derivative of the Jacobian by firstly compute dJ/dq.
 * To get dJ/dt, we perfrorom a tensor multiplication: dJ/dt = dJ/dq * dq/dt
 *
 * Args:
 *      J Eigen::MatrixXd(6,7), the Jacobian.
 *      ddq Eigen::VectorXd(7), the joint angle acceleraitons.
 * Return:
 *      dJ/dt Eigen::MatrixXd(6,7)
 */
template <size_t dof>
inline Eigen::MatrixXd getJacobianDerivative(const Eigen::MatrixXd& J, const Eigen::VectorXd& dq) {
    auto nb_rows = J.rows();
    auto nb_cols = J.cols();

    std::array<Eigen::MatrixXd, dof> J_grad;
    for (int i = 0; i < nb_cols; i++) {
        J_grad[i] = Eigen::MatrixXd::Zero(nb_rows, nb_cols);
    }

    for (int i = 0; i < nb_cols; i++) {
        for (int j = 0; j < nb_cols; j++) {
            auto J_i = J.col(i);
            auto J_j = J.col(j);

            if (j < i) {
                J_grad[j](0, i) = J_j[4] * J_i[2] - J_j[5] * J_i[1];
                J_grad[j](1, i) = J_j[5] * J_i[0] - J_j[3] * J_i[2];
                J_grad[j](2, i) = J_j[3] * J_i[1] - J_j[4] * J_i[0];
                J_grad[j](3, i) = J_j[4] * J_i[5] - J_j[5] * J_i[4];
                J_grad[j](4, i) = J_j[5] * J_i[3] - J_j[3] * J_i[5];
                J_grad[j](5, i) = J_j[3] * J_i[4] - J_j[4] * J_i[3];
            } else if (j > i) {
                J_grad[j](0, i) = -J_j[1] * J_i[5] + J_j[2] * J_i[4];
                J_grad[j](1, i) = -J_j[2] * J_i[3] + J_j[0] * J_i[5];
                J_grad[j](2, i) = -J_j[0] * J_i[4] + J_j[1] * J_i[3];
            } else {
                J_grad[j](0, i) = J_i[4] * J_i[2] - J_i[5] * J_i[1];
                J_grad[j](1, i) = J_i[5] * J_i[0] - J_i[3] * J_i[2];
                J_grad[j](2, i) = J_i[3] * J_i[1] - J_i[4] * J_i[0];
            }
        }
    }
    Eigen::MatrixXd dJ = Eigen::MatrixXd::Zero(nb_rows, nb_cols);

    for (int i = 0; i < nb_cols; i++) {
        Eigen::VectorXd Ji = Eigen::VectorXd::Zero(nb_rows);
        for (int j = 0; j < nb_cols; j++) {
            Ji += J_grad[j].col(i) * dq(j);
        }
        dJ.col(i) = Ji;
    }
    return dJ;
}

/**
 * Compute the pseudo inverse of the Jacobian, weighted with the Mass matrix.
 * Input:
 *      - J, the Full Jacobian (6x7).
 *      - M, the 7x7 inverse Ineria Mass Matrix.
 * Output: The 7x6 Jacobian inverse.
 */
Eigen::MatrixXd computeJacPseudoInverse(const Eigen::MatrixXd& J, const Eigen::MatrixXd& Minv);

/**
 * Convert an eigen vector to a std::array, it is usefull to convert a control command to send it to the robot.
 */
template <size_t s>
inline std::array<double, s> eigenVectorToStdArray(const Eigen::VectorXd& vec) {
    std::array<double, s> arr{};
    Eigen::VectorXd::Map(&arr[0], s) = vec;
    return arr;
}

/**
 * Convert an eigen matrix to a std::array, it is usefull to convert a control command to send it to the robot.
 */
template <size_t s>
std::array<double, s> eigenMatrixToStdArray(const Eigen::MatrixXd& mat) {
    Eigen::VectorXd vec(Eigen::Map<Eigen::VectorXd>(mat.data(), mat.cols() * mat.rows()));
    std::array<double, s> arr{};
    Eigen::VectorXd::Map(&arr[0], s) = vec;
    return arr;
}

std::vector<double> eigenVectorToStdVector(const Eigen::VectorXd& u);
std::vector<double> eigenMatrixToStdVector(const Eigen::MatrixXd& u);
}  // namespace ilqr_planner