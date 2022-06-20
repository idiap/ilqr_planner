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

#include "ilqr_planner/system/SequentialSystem.h"
namespace ilqr_planner {
namespace sys {

using namespace Eigen;

SequentialSystem::SequentialSystem(const std::shared_ptr<sim::SimulationInterface>& r, const std::vector<std::shared_ptr<System>>& systems, const VectorXd& RtDiag, int horizon, int nbDeriv)
    : systems(systems) {
    this->r = r;
    this->penalty = 0;
    this->limitsSet = false;
    localInit(RtDiag);
}

void SequentialSystem::localInit(const VectorXd& RtDiag) {
    this->R = RtDiag.asDiagonal();

    int nbStateVar0 = this->systems.at(0)->getNbStateVar();
    int nbTargetVar0 = this->systems.at(0)->getNbTargetVar();
    int nbCtrlVar0 = this->systems.at(0)->getNbCtrlVar();
    int horizon0 = this->systems.at(0)->getHorizon();
    int nbDeriv0 = this->systems.at(0)->getNbDeriv();
    int nbQVar0 = this->systems.at(0)->getNbQVar();

    VectorXd initState0 = this->systems.at(0)->getInitState();

    for (int i = 1; i < this->systems.size(); i++) {
        auto system = this->systems.at(i);

        if (nbStateVar0 != system->getNbStateVar()) {
            std::runtime_error(" All the systems does not have the same number of state variable ");
        }

        if (nbTargetVar0 != system->getNbTargetVar()) {
            std::runtime_error(" All the systems does not have the same number of target variable ");
        }

        if (nbCtrlVar0 != system->getNbCtrlVar()) {
            std::runtime_error(" All the systems does not have the same number of control variable ");
        }

        if (horizon0 != system->getHorizon()) {
            std::runtime_error(" All the systems does not have the same horizon ");
        }

        if (nbDeriv0 != system->getNbDeriv()) {
            std::runtime_error(" All the systems does not have the same number of derivatives ");
        }

        if (nbQVar0 != system->getNbQVar()) {
            std::runtime_error(" All the systems does not have the same number of precision variables ");
        }

        if (initState0 != system->getInitState()) {
            std::runtime_error(" All the systems does not have the same initState ");
        }
    }

    this->nbStateVar = nbStateVar0;
    this->nbTargetVar = nbTargetVar0 * this->systems.size();
    this->nbCtrlVar = nbCtrlVar0;
    this->nbQVar = nbQVar0 * this->systems.size();
    this->horizon = horizon0;
    this->nbDeriv = nbDeriv0;
    this->x0 = initState0;
    this->q0 = this->r->getJointsPos();
    this->dq0 = this->r->getJointsVel();
    this->f_x0 = std::get<0>(this->getFxJac());

    for (auto sys : this->systems) {
        keypoints.insert(keypoints.end(), sys->keypoints.begin(), sys->keypoints.end());
    }

    this->init();
}

std::tuple<VectorXd, VectorXd, MatrixXd, MatrixXd, MatrixXd> SequentialSystem::forwardPass(const VectorXd& xk, const VectorXd& uk, int k) {
    auto xFxABJ = this->systems.at(0)->forwardPass(xk, uk, k);

    for (int i = 1; i < this->systems.size(); i++) {
        this->systems.at(i)->r->updateKinematics();
    }

    VectorXd x = std::get<0>(xFxABJ);
    MatrixXd A = std::get<2>(xFxABJ);
    MatrixXd B = std::get<3>(xFxABJ);
    auto fxJ = this->getFxJac();

    return std::make_tuple(x, std::get<0>(fxJ), A, B, std::get<1>(fxJ));
}

std::tuple<VectorXd, MatrixXd> SequentialSystem::getFxJac() {
    VectorXd fx = VectorXd::Zero(this->nbTargetVar);
    MatrixXd J = MatrixXd::Zero(this->nbQVar, this->nbStateVar);

    int i = 0;
    for (auto sys : this->systems) {
        auto fxJ_k = sys->getFxJac();
        VectorXd fxk = std::get<0>(fxJ_k);
        MatrixXd Jk = std::get<1>(fxJ_k);

        fx.segment(i * fxk.rows(), fxk.rows()) = fxk;
        J.block(i * Jk.rows(), 0, Jk.rows(), Jk.cols()) = Jk;
        i += 1;
    }

    return std::make_tuple(fx, J);
}

VectorXd SequentialSystem::getState() {
    return this->systems.at(0)->getState();
}

VectorXd SequentialSystem::cost_F(const VectorXd& xk) {
    VectorXd c = VectorXd::Zero(1);
    for (auto sys : this->systems) {
        c += sys->cost_F(xk);
    }
    return c;
}

VectorXd SequentialSystem::cost_F_x(const VectorXd& xk) {
    VectorXd c = VectorXd::Zero(this->nbStateVar);
    for (auto sys : this->systems) {
        c += sys->cost_F_x(xk);
    }
    return c;
}

MatrixXd SequentialSystem::cost_F_xx(const VectorXd& xk) {
    MatrixXd c = MatrixXd::Zero(this->nbStateVar, this->nbStateVar);
    for (auto sys : this->systems) {
        c += sys->cost_F_xx(xk);
    }
    return c;
}

VectorXd SequentialSystem::cost(const VectorXd& xk, const VectorXd& uk, int k) {
    VectorXd c = VectorXd::Zero(1);
    for (auto sys : this->systems) {
        c += sys->cost(xk, uk, k);
    }
    return c;
}

VectorXd SequentialSystem::cost_x(const VectorXd& xk, const VectorXd& uk, int k) {
    VectorXd c = VectorXd::Zero(this->nbStateVar);
    for (auto sys : this->systems) {
        c += sys->cost_x(xk, uk, k);
    }
    return c;
}

MatrixXd SequentialSystem::cost_xx(const VectorXd& xk, const VectorXd& uk, int k) {
    MatrixXd c = MatrixXd::Zero(this->nbStateVar, this->nbStateVar);
    for (auto sys : this->systems) {
        c += sys->cost_xx(xk, uk, k);
    }
    return c;
}

VectorXd SequentialSystem::diff(const VectorXd& state, int k) {
    VectorXd diff = VectorXd::Zero(this->nbQVar);

    int i = 0;
    int nbTargetVar_k = this->nbTargetVar / this->systems.size();
    int nbQVar_k = this->nbQVar / this->systems.size();

    for (auto sys : this->systems) {
        diff.segment(i * nbQVar_k, nbQVar_k) = sys->diff(state.segment(i * nbTargetVar_k, nbTargetVar_k), k);
        i += 1;
    }

    return diff;
}

VectorXd SequentialSystem::getMuVector(bool sparse) {
    if (!sparse) {
        VectorXd mu = VectorXd::Zero(this->horizon * this->nbTargetVar);
        int i = 0;
        for (auto sys : this->systems) {
            VectorXd mu_k = sys->getMuVector();
            int nbTargetVar_k = sys->getNbTargetVar();

            for (int j = 0; j < this->horizon; j++) {
                VectorXd mu_k_t = mu_k.segment(j * nbTargetVar_k, nbTargetVar_k);
                mu.segment(j * this->nbTargetVar, this->nbTargetVar).segment(i * nbTargetVar_k, nbTargetVar_k) = mu_k_t;
            }

            i += 1;
        }

        return mu;
    } else {
        VectorXd mu = VectorXd::Zero(this->nbTargetVar * this->keypoints.size());

        for (int i = 0; i < this->keypoints.size(); i++) {
            VectorXd mu_t = VectorXd::Zero(this->nbTargetVar);

            for (int j = 0; j < this->systems.size(); j++) {
                auto kp = this->systems.at(j)->getKeypoint(this->keypoints.at(i)->getTimestep());
                VectorXd mu_t_j = VectorXd::Zero(this->systems.at(j)->getNbTargetVar());

                if (kp != nullptr) {
                    mu_t_j = kp->getState();
                }

                mu_t.segment(j * this->systems.at(j)->getNbTargetVar(), this->systems.at(j)->getNbTargetVar()) = mu_t_j;
            }

            mu.segment(i * this->nbTargetVar, this->nbTargetVar) = mu_t;
        }

        return mu;
    }
}

MatrixXd SequentialSystem::getQMatrix(bool sparse) {
    // This solution is only concatening all the diagonal elements of each sub system matrices.
    // It means that at the moment you can not play with off-diagonal elements.
    // To do this, we need to change the algorithm and make it way more costly.

    if (!sparse) {
        MatrixXd Q = MatrixXd::Zero(this->horizon * this->nbQVar, this->horizon * this->nbQVar);

        int i = 0;
        for (auto sys : this->systems) {
            MatrixXd Q_k = sys->getQMatrix();
            int nbQVar_k = sys->getNbQVar();

            for (int j = 0; j < this->horizon; j++) {
                MatrixXd Q_k_t = Q_k.block(j * nbQVar_k, j * nbQVar_k, nbQVar_k, nbQVar_k);
                Q.block(j * this->nbQVar, j * this->nbQVar, this->nbQVar, this->nbQVar).block(i * nbQVar_k, i * nbQVar_k, nbQVar_k, nbQVar_k) = Q_k_t;
            }

            i += 1;
        }

        return Q;
    } else {
        MatrixXd Q = MatrixXd::Zero(this->nbQVar * this->keypoints.size(), this->nbQVar * this->keypoints.size());

        for (int i = 0; i < this->keypoints.size(); i++) {
            MatrixXd Q_t = MatrixXd::Zero(this->nbQVar, this->nbQVar);

            for (int j = 0; j < this->systems.size(); j++) {
                auto kp = this->systems.at(j)->getKeypoint(this->keypoints.at(i)->getTimestep());
                MatrixXd Q_t_j = MatrixXd::Zero(this->systems.at(j)->getNbQVar(), this->systems.at(j)->getNbQVar());

                if (kp != nullptr) {
                    Q_t_j = kp->getPrecision();
                }
                Q_t.block(j * this->systems.at(j)->getNbQVar(), j * this->systems.at(j)->getNbQVar(), this->systems.at(j)->getNbQVar(), this->systems.at(j)->getNbQVar()) = Q_t_j;
            }

            Q.block(i * this->nbQVar, i * this->nbQVar, this->nbQVar, this->nbQVar) = Q_t;
        }

        return Q;
    }
}

void SequentialSystem::reset() {
    for (auto sys : this->systems) {
        sys->reset();
    }
}

}  // namespace sys
}  // namespace ilqr_planner