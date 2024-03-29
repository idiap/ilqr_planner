// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/solver/BatchILQR.h"
#include "ilqr_planner/utils/utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>

using namespace Eigen;

namespace ilqr_planner {
namespace solver {

BatchILQR::BatchILQR(const std::shared_ptr<sys::System>& s, const MatrixXd& Q) : s(s), Q(Q) {
    this->mu = s->getMuVector(true);
    this->R = s->getRt().diagonal().replicate(s->getHorizon() - 1, 1).asDiagonal();
    this->vp_indexes = s->getKpIndexes();
}

BatchILQR::BatchILQR(const std::shared_ptr<sys::System>& s) : s(s) {
    this->R = s->getRt().diagonal().replicate(s->getHorizon() - 1, 1).asDiagonal();
    this->mu = s->getMuVector(true);
    this->Q = s->getQMatrix(true);
    this->vp_indexes = s->getKpIndexes();
}

MatrixXd BatchILQR::buildL(const std::vector<std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>>& ABJLs) {
    MatrixXd L = MatrixXd::Zero(this->vp_indexes.size() * this->s->getNbStateVar(), this->vp_indexes.size() * this->s->getNbStateVar());
    int i = 0;
    for (int t : this->vp_indexes) {
        MatrixXd Lt = std::get<3>(ABJLs.at(t));
        if (!Lt.isZero()) {
            L.block(i * this->s->getNbStateVar(), i * this->s->getNbStateVar(), this->s->getNbStateVar(), this->s->getNbStateVar()) << Lt;
        }
        i++;
    }

    return L;
}

MatrixXd BatchILQR::buildLc(const std::vector<MatrixXd>& Ls) {
    MatrixXd L = MatrixXd::Zero(Ls.size() * Ls.at(0).rows(), Ls.size() * Ls.at(0).cols());

    for (int i = 0; i < Ls.size(); i++) {
        MatrixXd Lt = Ls.at(i);
        if (!Lt.isZero()) {
            L.block(i * this->s->getNbStateVar(), i * this->s->getNbStateVar(), this->s->getNbStateVar(), this->s->getNbStateVar()) << Lt;
        }
    }

    return L;
}

std::tuple<MatrixXd, MatrixXd, MatrixXd> BatchILQR::buildSuJL(const std::vector<std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>>& ABJLs) {
    MatrixXd Su = MatrixXd::Zero(this->vp_indexes.size() * this->s->getNbStateVar(), this->s->getNbCtrlVar() * (this->s->getHorizon() - 1));

    int Jdim1 = std::get<2>(ABJLs.at(0)).rows();
    int Jdim2 = std::get<2>(ABJLs.at(0)).cols();

    MatrixXd J = MatrixXd::Zero(this->vp_indexes.size() * Jdim1, this->vp_indexes.size() * Jdim2);
    MatrixXd L = MatrixXd::Zero(this->vp_indexes.size() * this->s->getNbStateVar(), this->vp_indexes.size() * this->s->getNbStateVar());

    auto M = std::get<1>(ABJLs.at(0));
    int t = 0;

    for (int i = 0; i < ABJLs.size(); i++) {
        auto At = std::get<0>(ABJLs.at(i));
        auto Bt = std::get<1>(ABJLs.at(i));

        if (std::find(this->vp_indexes.begin(), this->vp_indexes.end(), i) != this->vp_indexes.end()) {
            auto Lt = std::get<3>(ABJLs.at(i));
            L.block(t * this->s->getNbStateVar(), t * this->s->getNbStateVar(), this->s->getNbStateVar(), this->s->getNbStateVar()) << Lt;
            if (i > 0) {
                Su.block(t * this->s->getNbStateVar(), 0, M.rows(), M.cols()) = M;
            }
            auto Jt = std::get<2>(ABJLs.at(i));
            J.block(t * Jdim1, t * Jdim2, Jdim1, Jdim2) = Jt;

            t++;
        }

        if (i > 0) {
            MatrixXd newM(M.rows(), M.cols() + Bt.cols());
            newM << (At * M), Bt;
            M = newM;
        }
    }

    return std::make_tuple(Su, J, L);
}

VectorXd BatchILQR::truncateStates(const VectorXd& states, const int& state_size) {
    VectorXd truncated_states = VectorXd::Zero(this->vp_indexes.size() * state_size);
    int i = 0;
    for (const auto t : vp_indexes) {
        truncated_states.segment(i * state_size, state_size) = states.segment(t * state_size, state_size);
        i++;
    }
    return truncated_states;
}

VectorXd BatchILQR::solve(int nb_iter, const VectorXd& u0, bool early_stop, CallBackMessage* cb) {
    this->s->reset();
    VectorXd u = u0;

    for (int i = 0; i < nb_iter; i++) {
        auto fp = this->s->fpBatch(u);

        auto x = std::get<0>(fp);
        auto ql = std::get<1>(fp);
        auto ABJLs = std::get<2>(fp);

        auto SuJL = this->buildSuJL(ABJLs);
        auto Su = std::get<0>(SuJL);
        auto Jt = std::get<1>(SuJL);
        auto Lt = std::get<2>(SuJL);

        x = this->truncateStates(x, this->s->getNbTargetVar());
        ql = this->truncateStates(ql, this->s->getNbStateVar());
        VectorXd error = this->s->diffBatch(x);

        MatrixXd lstq_A = (Su.transpose() * (Jt.transpose() * Q * Jt + Lt) * Su) + this->R;
        VectorXd lstq_B = (Su.transpose() * (Jt.transpose() * Q * error + Lt * ql) - this->R * u);
        VectorXd du = lstq_A.inverse() * lstq_B;

        double cost0 = (error.transpose() * this->Q * error + u.transpose() * this->R * u + ql.transpose() * Lt * ql)(0);
        double alpha = 1.0;

        while (true) {
            VectorXd utmp = u + alpha * du;
            auto fptmp = this->s->fpBatch(utmp);
            auto xt = std::get<0>(fptmp);
            auto qltmp = std::get<1>(fptmp);
            auto ABJLtmp = std::get<2>(fptmp);
            auto Ltmp = this->buildL(ABJLtmp);

            xt = this->truncateStates(xt, this->s->getNbTargetVar());
            qltmp = this->truncateStates(qltmp, this->s->getNbStateVar());

            VectorXd errorTMP = this->s->diffBatch(xt);
            double cost = (errorTMP.transpose() * this->Q * errorTMP + utmp.transpose() * this->R * utmp + qltmp.transpose() * Ltmp * qltmp)(0);

            if ((cost < cost0) || (alpha < 1e-3)) {
                u = utmp;
                break;
            }

            alpha /= 2;
        }

        std::stringstream msg;
        msg << "Iteration " << i + 1 << ", Cost: " << cost0 << ", alpha= " << alpha;
        if (cb == nullptr)
            std::cout << msg.str() << std::endl;
        else
            cb->notify(msg.str());

        if (early_stop && alpha * du.norm() < 1e-3) {
            break;
        }
    }

    this->s->reset();
    return u;
}
}  // namespace solver
}  // namespace ilqr_planner
