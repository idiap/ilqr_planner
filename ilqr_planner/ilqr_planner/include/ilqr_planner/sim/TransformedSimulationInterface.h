// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <eigen3/Eigen/Dense>
#include <memory>
#include "ilqr_planner/sim/SimulationInterface.h"

namespace ilqr_planner {
namespace sim {

class TransformedSimulationInterface : public SimulationInterface {
public:
    ~TransformedSimulationInterface() {}
    TransformedSimulationInterface(const std::shared_ptr<SimulationInterface>& r, const Eigen::MatrixXd& T);
    explicit TransformedSimulationInterface(const Eigen::MatrixXd& T);

    void subscribe(const std::shared_ptr<SimulationInterface>& r);
    void updateKinematics() override;
    Eigen::MatrixXd J() override;
    Eigen::MatrixXd Jp() override;
    Eigen::VectorXd getEEPosition() override;
    Eigen::VectorXd getEEVelocity() override;
    Eigen::VectorXd getEEAngVel() override;
    Eigen::VectorXd getEEOrnQuat() override;
    void sendAcc(double dt, const Eigen::VectorXd& ddq, bool updateKin = true) override;
    void sendVel(double dt, const Eigen::VectorXd& dq, bool updateKin = true) override;
    void setConfiguration(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, bool reset_time = true) override;
    void setTime(double time) override;

private:
    std::shared_ptr<SimulationInterface> r;
    Eigen::MatrixXd T;
    bool initialized;
};

}  // namespace sim
}  // namespace ilqr_planner
