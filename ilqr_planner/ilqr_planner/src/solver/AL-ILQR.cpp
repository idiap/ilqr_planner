// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/solver/AL-ILQR.h"
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include "ilqr_planner/utils/utils.h"

namespace ilqr_planner {
namespace solver {

using namespace Eigen;

AL_ILQR::AL_ILQR(const std::shared_ptr<sys::System>& s, const std::vector<Constraint>& inequality, const std::vector<VectorXd>& initLambda) : s(s), inequality(inequality), multipliers(initLambda) {}

std::tuple<MatrixXd, VectorXd> AL_ILQR::constraints(const VectorXd& xk, const VectorXd& uk, int k) {
    auto inequCons = this->inequality.at(k);

    if (inequCons.b.rows() == 0) {
        MatrixXd Ik = MatrixXd::Zero(xk.rows() + uk.rows(), xk.rows() + uk.rows());
        VectorXd gk = VectorXd::Zero(xk.rows() + uk.rows());
        return std::make_tuple(Ik, gk);
    }

    VectorXd sk(xk.size() + uk.size());
    sk << xk, uk;

    VectorXd gk = inequCons.A * sk - inequCons.b;
    VectorXd Ik_ineq_diag = VectorXd::Ones(gk.rows());
    for (int i = 0; i < gk.rows(); i++) {
        if (gk(i) < 0 && this->multipliers.at(k)(i) == 0) {
            Ik_ineq_diag(i) = 0;
        }
    }

    MatrixXd Ik = MatrixXd::Zero(Ik_ineq_diag.rows(), Ik_ineq_diag.rows());
    Ik.diagonal() = Ik_ineq_diag;
    return std::make_tuple(Ik, gk);
}
VectorXd AL_ILQR::augmentedLossK(const VectorXd& xk, const VectorXd& uk, int k, const VectorXd& lambdak, const VectorXd& Ck, const MatrixXd& Ik) {
    VectorXd J = this->s->cost(xk, uk, k) + ((lambdak + .5 * Ik * Ck).transpose() * Ck);
    return J;
}

std::tuple<std::vector<VectorXd>, std::vector<VectorXd>, std::vector<VectorXd>>
AL_ILQR::solve(const std::vector<VectorXd>& U0, int nb_iter, int lag_update_step, double penalty, double scaling_factor, bool line_search, bool early_stop, CallBackMessage* cb) {
    // Initialize X,loss from U
    auto x0 = s->getInitState();
    auto fx0 = s->getInitFoXState();
    std::vector<VectorXd> U = U0;
    std::vector<VectorXd> X, f_X, losses, Cs;
    std::vector<MatrixXd> As, Bs, Js, Is;

    int horizon = s->getHorizon();

    X.push_back(x0);
    f_X.push_back(std::get<0>(this->s->getFxJac()));
    Js.push_back(std::get<1>(this->s->getFxJac()));
    VectorXd J0 = VectorXd::Zero(1);

    for (int i = 0; i < horizon - 1; i++) {
        auto uk = U.at(i);
        auto xk = X.at(i);
        auto fx = f_X.at(i);

        auto IkCk = this->constraints(xk, uk, i);
        Is.push_back(penalty * std::get<0>(IkCk));
        Cs.push_back(std::get<1>(IkCk));
        losses.push_back(this->s->cost(xk, uk, i));
        J0 += losses.at(i);

        auto XFABJ = this->s->forwardPass(xk, uk, i);
        X.push_back(std::get<0>(XFABJ));
        f_X.push_back(std::get<1>(XFABJ));
        As.push_back(std::get<2>(XFABJ));
        Bs.push_back(std::get<3>(XFABJ));
        Js.push_back(std::get<4>(XFABJ));
    }
    auto finalLoss = this->s->cost_F(X.at(horizon - 1));
    J0 += finalLoss;

    for (int i = 0; i < nb_iter; i++) {
        auto start = std::chrono::steady_clock::now();
        std::vector<MatrixXd> Ks;
        std::vector<VectorXd> ds;

        this->s->reset();

        // BACKWARD PASS
        // We assume that there are no constraint on the final state;
        MatrixXd lN_xx = this->s->cost_F_xx(X.at(horizon - 1));
        MatrixXd P = lN_xx;
        VectorXd p = this->s->cost_F_x(X.at(horizon - 1));

        for (int k = horizon - 2; k >= 0; k--) {
            auto xk = X.at(k);
            auto fxk = f_X.at(k);

            auto uk = U.at(k);

            auto A = As.at(k);
            auto B = Bs.at(k);
            auto J = Js.at(k);

            VectorXd lambdak;
            MatrixXd ckx_ineq, cku_ineq;

            auto ck = Cs.at(k);
            auto Ik = Is.at(k);

            if (this->inequality.at(k).A.cols() > 0) {
                ckx_ineq = this->inequality.at(k).A.leftCols(xk.rows());
                cku_ineq = this->inequality.at(k).A.rightCols(uk.rows());
                lambdak = this->multipliers.at(k);
            } else {
                ckx_ineq = MatrixXd::Zero(Ik.rows(), xk.rows());
                cku_ineq = MatrixXd::Zero(Ik.rows(), uk.rows());
                lambdak = VectorXd::Zero(Ik.rows());
            }

            MatrixXd ckx = ckx_ineq;
            MatrixXd cku = cku_ineq;

            MatrixXd Qux = this->s->cost_ux(xk, uk, k) + B.transpose() * P * A + cku.transpose() * Ik * ckx;
            MatrixXd Quu = this->s->cost_uu(xk, uk, k) + B.transpose() * P * B + cku.transpose() * Ik * cku;
            MatrixXd Qxx = this->s->cost_xx(xk, uk, k) + A.transpose() * P * A + ckx.transpose() * Ik * ckx;
            MatrixXd Qxu = this->s->cost_xu(xk, uk, k) + A.transpose() * P * B + ckx.transpose() * Ik * cku;
            VectorXd Qu = this->s->cost_u(xk, uk, k) + B.transpose() * p + cku.transpose() * (lambdak + Ik * ck);
            VectorXd Qx = this->s->cost_x(xk, uk, k) + A.transpose() * p + ckx.transpose() * (lambdak + Ik * ck);

            MatrixXd Quu_inv = -1 * (Quu + 1e-6 * MatrixXd::Identity(Quu.rows(), Quu.cols())).inverse();
            MatrixXd Kk = Quu_inv * Qux;
            VectorXd dk = Quu_inv * Qu;

            P = Qxx + Kk.transpose() * Quu * Kk + Kk.transpose() * Qux + Qxu * Kk;
            p = Qx + Kk.transpose() * Quu * dk + Kk.transpose() * Qu + Qxu * dk;

            Ks.push_back(Kk);
            ds.push_back(dk);
        }

        // FORWARD PASS

        double alpha = 2;
        std::vector<VectorXd> newX, newF_X, newU;
        VectorXd newJ = VectorXd::Zero(1);
        double du_square_norm = 0;

        do {
            this->s->reset();
            alpha /= 2.0;

            newX.clear();
            newF_X.clear();
            newU.clear();
            du_square_norm = 0;
            newX.push_back(x0);
            newF_X.push_back(std::get<0>(this->s->getFxJac()));
            Js.at(0) = std::get<1>(this->s->getFxJac());

            newJ << 0;

            for (int k = 0; k < horizon - 1; k++) {
                int k2 = horizon - 2 - k;

                // Compute command
                VectorXd du = Ks.at(k2) * (newX.at(k) - X.at(k)) + alpha * ds.at(k2);
                du_square_norm += du.norm();
                VectorXd new_uk = U.at(k) + du;

                // Forward pass
                auto XFABJ = this->s->forwardPass(newX.at(k), new_uk, k);
                auto xkp1 = std::get<0>(XFABJ);
                auto fxkp1 = std::get<1>(XFABJ);

                newX.push_back(xkp1);
                newF_X.push_back(fxkp1);
                newU.push_back(new_uk);

                As.at(k) = std::get<2>(XFABJ);
                Bs.at(k) = std::get<3>(XFABJ);
                Js.at(k) = std::get<4>(XFABJ);

                auto IkCk = this->constraints(newX.at(k), new_uk, k);
                Is.at(k) = penalty * std::get<0>(IkCk);
                Cs.at(k) = std::get<1>(IkCk);

                auto loss = this->s->cost(newX.at(k), new_uk, k);
                newJ += loss;
            }

            finalLoss = this->s->cost_F(newX.at(horizon - 1));
            newJ += finalLoss;
        } while (((newJ(0) >= J0(0)) || (newJ.array().isNaN()).any()) && alpha > 1e-3 && line_search);

        // Update of the multipliers
        if ((i + 1) % lag_update_step == 0) {
            penalty *= scaling_factor;
            for (int k = 0; k < horizon - 1; k++) {
                VectorXd newIneqL = this->multipliers.at(k) + penalty * Cs.at(k).tail(this->multipliers.at(k).rows());
                this->multipliers.at(k) = newIneqL.cwiseMax(0);
            }
        }

        J0 = newJ;
        X = newX;
        f_X = newF_X;
        U = newU;

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        std::stringstream msg;
        msg << "Iteration " << i + 1 << ", Cost: " << J0(0) << ", alpha= " << alpha << ", time= " << elapsed_seconds.count();
        if (cb == nullptr)
            std::cout << msg.str() << std::endl;
        else
            cb->notify(msg.str());

        if (early_stop && alpha * sqrt(du_square_norm) < 1e-3) {
            break;
        }
    }

    this->s->reset();
    return std::make_tuple(X, f_X, U);
}
}  // namespace solver
}  // namespace ilqr_planner
