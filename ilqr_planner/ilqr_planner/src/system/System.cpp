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

#include "ilqr_planner/system/System.h"
#include "ilqr_planner/utils/sd.h"
#include "ilqr_planner/utils/utils.h"

#include <algorithm>

using namespace Eigen;

namespace ilqr_planner {
namespace sys {

System::System(const std::shared_ptr<sim::SimulationInterface>& r,
               const std::vector<std::shared_ptr<Keypoint>>& keypoints,
               const VectorXd& RtDiag,
               const VectorXd& qMax,
               const VectorXd& qMin,
               int horizon,
               int nbDeriv)
    : System(r, keypoints, RtDiag, qMax, qMin, VectorXd::Zero(r->getDOF()), VectorXd::Zero(r->getDOF()), horizon, nbDeriv) {}

System::System(const std::shared_ptr<sim::SimulationInterface>& r,
               const std::vector<std::shared_ptr<Keypoint>>& keypoints,
               const VectorXd& RtDiag,
               const VectorXd& qMax,
               const VectorXd& qMin,
               const VectorXd& dqMax,
               const VectorXd& dqMin,
               int horizon,
               int nbDeriv)
    : r(r), keypoints(keypoints), horizon(horizon), nbDeriv(nbDeriv) {
    this->limitsSet = true;
    this->penalty = 1;
    this->R = (RtDiag.asDiagonal());
    this->init();

    int state_space_size = this->nbDeriv * this->r->getDOF();

    this->state_max = VectorXd::Zero(state_space_size);
    this->state_min = VectorXd::Zero(state_space_size);

    this->joint_limits_weight = VectorXi::Ones(state_space_size);

    if (nbDeriv == 1) {
        this->state_max = qMax;
        this->state_min = qMin;
    } else if (nbDeriv == 2) {
        this->state_max << qMax, dqMax;
        this->state_min << qMin, dqMin;

        if (dqMax.isApprox(dqMin)) {
            this->joint_limits_weight.tail(this->r->getDOF()) = VectorXi::Zero(this->r->getDOF());
        }
    }
}

System::System(const std::shared_ptr<sim::SimulationInterface>& r, const std::vector<std::shared_ptr<Keypoint>>& keypoints, const VectorXd& RtDiag, int horizon, int nbDeriv)
    : r(r), keypoints(keypoints), horizon(horizon), nbDeriv(nbDeriv) {
    this->limitsSet = false;
    this->penalty = 0;
    this->R = (RtDiag.asDiagonal());
    this->init();
}

void System::init() {
    for (auto& kp : keypoints) {
        this->keypoints_map[kp->getTimestep()] = kp;
    }

    std::sort(keypoints.begin(), keypoints.end(), [](const std::shared_ptr<Keypoint>& a, const std::shared_ptr<Keypoint>& b) -> bool { return a->getTimestep() < b->getTimestep(); });
}

std::vector<int> System::getKpIndexes() {
    std::vector<int> kpIndexes;
    for (auto const& kp : keypoints) {
        kpIndexes.push_back(kp->getTimestep());
    }
    return kpIndexes;
}

std::shared_ptr<Keypoint> System::getKeypoint(int k) {
    if (this->keypoints_map.find(k) != this->keypoints_map.end()) {
        return this->keypoints_map[k];
    }
    return nullptr;
}

VectorXd System::diff(const VectorXd& actual_state, int k) {
    if (this->keypoints_map.find(k) == this->keypoints_map.end()) {
        return VectorXd::Zero(this->nbQVar);
    }

    return this->keypoints_map[k]->diff(actual_state);
}

VectorXd System::diffBatch(const VectorXd& x) {
    int horizon = this->keypoints.size();
    VectorXd diff = VectorXd::Zero(horizon * this->nbQVar);
    for (int i = 0; i < horizon; i++) {
        VectorXd xt = x.segment(i * this->nbTargetVar, this->nbTargetVar);
        diff.segment(i * this->nbQVar, this->nbQVar) = this->diff(xt, this->keypoints.at(i)->getTimestep());
    }
    return diff;
}

std::pair<MatrixXd, VectorXd> System::inspectJointLimit(VectorXd x_k) {
    MatrixXd L = MatrixXd::Zero(this->nbStateVar, this->nbStateVar);
    VectorXd q = VectorXd::Zero(this->nbStateVar);

    if (this->limitsSet) {
        auto maxViolation = x_k.array() > this->state_max.array();
        auto minViolation = x_k.array() < this->state_min.array();

        for (int i = 0; i < this->nbStateVar; i++) {
            if (this->joint_limits_weight(i) != 0) {
                if (maxViolation(i)) {
                    q(i) = this->state_max(i) - x_k(i);
                    L(i, i) = this->penalty;
                } else if (minViolation(i)) {
                    q(i) = this->state_min(i) - x_k(i);
                    L(i, i) = this->penalty;
                }
            }
        }
    }
    return std::make_pair(L, q);
}

std::tuple<VectorXd, VectorXd, VectorXd, VectorXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd> System::forwardPassWithLimits(const VectorXd& xk, const VectorXd& uk, int k) {
    auto xFxABJ = this->forwardPass(xk, uk, k);

    auto x = std::get<0>(xFxABJ);
    auto fx = std::get<1>(xFxABJ);
    auto A = std::get<2>(xFxABJ);
    auto B = std::get<3>(xFxABJ);
    auto J = std::get<4>(xFxABJ);

    VectorXd u = VectorXd::Zero(this->nbCtrlVar);

    MatrixXd L;
    VectorXd q;

    std::tie(L, q) = inspectJointLimit(xk);

    return std::make_tuple(x, fx, q, u, A, B, J, L);
}

std::tuple<VectorXd, MatrixXd> System::getFxJac(VectorXd xk) {
    VectorXd qk = xk.head(this->r->getDOF());
    VectorXd dqk = VectorXd::Zero(this->r->getDOF());

    if (this->nbDeriv == 2) {
        dqk = xk.segment(this->r->getDOF(), this->r->getDOF());
    }

    VectorXd old_q0 = this->r->getJointsPos();
    VectorXd old_dq0 = this->r->getJointsVel();

    this->r->setConfiguration(qk, dqk);
    auto fx_jac = this->getFxJac();
    this->r->setConfiguration(old_q0, old_dq0);

    return fx_jac;
}

std::tuple<VectorXd, VectorXd, std::vector<std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>>> System::fpBatch(const VectorXd& u) {
    this->reset();
    int horizon = u.rows() / this->nbCtrlVar + 1;

    VectorXd fX = VectorXd::Zero(this->horizon * this->nbTargetVar);
    VectorXd qL = VectorXd::Zero(horizon * this->nbStateVar);
    VectorXd uL = VectorXd::Zero((horizon - 1) * this->nbCtrlVar);

    auto fJ0 = this->getFxJac();
    fX.head(this->nbTargetVar) << std::get<0>(fJ0);
    MatrixXd J0 = std::get<1>(fJ0);

    std::vector<std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>> ABJLs;

    MatrixXd A = MatrixXd::Identity(this->nbStateVar, this->nbStateVar);
    MatrixXd B = MatrixXd::Zero(this->nbStateVar, this->nbCtrlVar);
    MatrixXd L = MatrixXd::Zero(this->nbStateVar, this->nbStateVar);

    ABJLs.push_back(std::make_tuple(A, B, J0, L));

    for (int i = 0; i < horizon - 1; i++) {
        VectorXd ut = u.segment(i * this->nbCtrlVar, this->nbCtrlVar);
        auto xFxquABJLLc = this->forwardPassWithLimits(this->getState(), ut, i + 1);

        fX.segment((i + 1) * this->nbTargetVar, this->nbTargetVar) << std::get<1>(xFxquABJLLc);
        qL.segment((i + 1) * this->nbStateVar, this->nbStateVar) << std::get<2>(xFxquABJLLc);
        uL.segment(i * this->nbCtrlVar, this->nbCtrlVar) << std::get<3>(xFxquABJLLc);
        ABJLs.push_back(std::make_tuple(std::get<4>(xFxquABJLLc), std::get<5>(xFxquABJLLc), std::get<6>(xFxquABJLLc), std::get<7>(xFxquABJLLc)));
    }
    return std::make_tuple(fX, qL, ABJLs);
}

VectorXd System::cost(const VectorXd& xk, const VectorXd& uk, int k) {
    VectorXd cost = VectorXd::Zero(1);

    // Task reaching part
    std::shared_ptr<Keypoint> kp = this->getKeypoint(k);
    if (kp != nullptr) {
        auto fxk = std::get<0>(this->getFxJac(xk));
        VectorXd diff = kp->diff(fxk);
        cost += diff.transpose() * kp->getPrecision() * diff + uk.transpose() * this->R * uk;
    }

    // Limit avoidance part
    if (this->limitsSet) {
        MatrixXd L;
        VectorXd q;

        std::tie(L, q) = this->inspectJointLimit(xk);
        cost += q.transpose() * L * q;
    }

    return cost;
}

VectorXd System::cost_F(const VectorXd& xk) {
    return this->cost(xk, VectorXd::Zero(this->nbCtrlVar), this->horizon - 1);
}

VectorXd System::cost_F_x(const VectorXd& xk) {
    return this->cost_x(xk, VectorXd::Zero(this->nbCtrlVar), this->horizon - 1);
}

MatrixXd System::cost_F_xx(const VectorXd& xk) {
    return this->cost_xx(xk, VectorXd::Zero(this->nbCtrlVar), this->horizon - 1);
}

VectorXd System::cost_x(const VectorXd& xk, const VectorXd& uk, int k) {
    VectorXd cost = VectorXd::Zero(this->nbStateVar);

    // Task reaching part
    std::shared_ptr<Keypoint> kp = this->getKeypoint(k);
    if (kp != nullptr) {
        auto fxk_J = this->getFxJac(xk);
        auto fxk = std::get<0>(fxk_J);
        auto Jk = std::get<1>(fxk_J);

        VectorXd diff = kp->diff(fxk);
        cost += -1 * (Jk).transpose() * kp->getPrecision() * diff;
    }

    // Limit avoidance part
    if (this->limitsSet) {
        MatrixXd L;
        VectorXd q;

        std::tie(L, q) = this->inspectJointLimit(xk);
        cost += -L.transpose() * q;
    }

    return cost;
}

VectorXd System::cost_u(const VectorXd& xk, const VectorXd& uk, int k) {
    return this->R * uk;
}

MatrixXd System::cost_ux(const VectorXd& xk, const VectorXd& uk, int k) {
    return MatrixXd::Zero(uk.rows(), this->nbStateVar);
}

MatrixXd System::cost_uu(const VectorXd& xk, const VectorXd& uk, int k) {
    return this->R;
}

MatrixXd System::cost_xx(const VectorXd& xk, const VectorXd& uk, int k) {
    MatrixXd cost = MatrixXd::Zero(this->nbStateVar, this->nbStateVar);

    // Task reaching part
    std::shared_ptr<Keypoint> kp = this->getKeypoint(k);
    if (kp != nullptr) {
        auto fxk_J = this->getFxJac(xk);
        auto Jk = std::get<1>(fxk_J);

        cost += Jk.transpose() * kp->getPrecision() * Jk;
    }

    // Limit avoidance part
    if (this->limitsSet) {
        MatrixXd L;
        VectorXd q;

        std::tie(L, q) = this->inspectJointLimit(xk);
        cost += L.transpose() * L;
    }

    return cost;
}

MatrixXd System::cost_xu(const VectorXd& xk, const VectorXd& uk, int k) {
    return MatrixXd::Zero(this->nbStateVar, uk.rows());
}

VectorXd System::getInitState() {
    return this->x0;
}
VectorXd System::getInitFoXState() {
    return this->f_x0;
}

VectorXd System::getMuVector(bool sparse) {
    if (sparse) {
        VectorXd mu = VectorXd::Zero(this->nbTargetVar * this->keypoints.size());
        for (int i = 0; i < this->keypoints.size(); i++) {
            mu.segment(i * this->nbTargetVar, this->nbTargetVar) = this->keypoints.at(i)->getState();
        }
        return mu;
    } else {
        VectorXd mu = VectorXd::Zero(this->horizon * this->nbTargetVar);

        int i = 0;
        for (auto const& kp : this->keypoints) {
            mu.segment(kp->getTimestep() * this->nbTargetVar, this->nbTargetVar) = kp->getState();
            i++;
        }

        return mu;
    }
}

MatrixXd System::getQMatrix(bool sparse) {
    if (sparse) {
        MatrixXd Q = MatrixXd::Zero(this->keypoints.size() * this->nbQVar, this->keypoints.size() * this->nbQVar);

        for (int i = 0; i < this->keypoints.size(); i++) {
            Q.block(i * this->nbQVar, i * this->nbQVar, this->nbQVar, this->nbQVar) = this->keypoints.at(i)->getPrecision();
        }

        return Q;
    } else {
        MatrixXd Q = MatrixXd::Zero(this->horizon * this->nbQVar, this->horizon * this->nbQVar);

        int i = 0;
        for (auto const& kp : this->keypoints) {
            Q.block(kp->getTimestep() * this->nbQVar, kp->getTimestep() * this->nbQVar, this->nbQVar, this->nbQVar) = kp->getPrecision();
            i++;
        }

        return Q;
    }
}

void System::checkKeypoints(const std::string& expected_tag) {
    for (auto kp : keypoints) {
        if (kp->getTAG() != expected_tag) {
            throw std::runtime_error("[PosOrnPlannerSys] Wrong keypoint type: Expecting " + expected_tag + " got " + kp->getTAG());
        }
        if (kp->getType() != this->nbDeriv) {
            throw std::runtime_error("[PosOrnPlannerSys] Wrong keypoint order (nbDeriv): Expecting " + std::to_string(this->nbDeriv) + " got " + std::to_string(kp->getType()));
        }
    }
}

}  // namespace sys
}  // namespace ilqr_planner