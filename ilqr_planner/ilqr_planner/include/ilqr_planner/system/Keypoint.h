// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <eigen3/Eigen/Dense>
#include <string>

namespace ilqr_planner {
namespace sys {

class Keypoint {
public:
    enum KpType { FIRST_ORDER = 1, SECOND_ORDER = 2 };

    Keypoint(const int& timestep, const KpType& type, const std::string& TAG) : timestep_(timestep), type_(type), TAG_(TAG) {}

    virtual ~Keypoint() {}

    virtual Eigen::VectorXd diff(const Eigen::VectorXd& state) const = 0;

    virtual Eigen::VectorXd getState() const = 0;

    virtual Eigen::MatrixXd getPrecision() const = 0;

    int getTimestep() const { return timestep_; }

    std::string getTAG() const { return TAG_; }

    KpType getType() { return type_; }

    virtual bool isPartOfEuclideanSpace() const = 0;

protected:
    std::string TAG_;
    KpType type_;
    int timestep_;
};

}  // namespace sys
}  // namespace ilqr_planner
