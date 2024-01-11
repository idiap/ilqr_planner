// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "ilqr_planner/solver/ILQRRecursive.h"
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include "ilqr_planner/utils/utils.h"

using namespace Eigen;

namespace ilqr_planner {
namespace solver {

ILQRRecursive::ILQRRecursive(const std::shared_ptr<sys::System>& s) : s(s) {}

std::tuple<std::vector<VectorXd>, std::vector<VectorXd>, std::vector<VectorXd>, std::vector<MatrixXd>, std::vector<VectorXd>, double> ILQRRecursive::solve(const std::vector<VectorXd>& U0,
                                                                                                                                                           int nb_iter,
                                                                                                                                                           bool line_search,
                                                                                                                                                           bool early_stop,
                                                                                                                                                           CallBackMessage* cb) {
    // Initialize X,loss from U
    s->reset();
    auto x0 = s->getInitState();
    auto fx0 = s->getInitFoXState();
    std::vector<VectorXd> U = U0;
    std::vector<VectorXd> X, f_X, losses;
    std::vector<MatrixXd> As, Bs;

    int horizon = s->getHorizon();

    X.push_back(x0);
    f_X.push_back(std::get<0>(s->getFxJac()));

    VectorXd cost0 = VectorXd::Zero(1);

    for (int i = 0; i < horizon - 1; i++) {
        auto uk = U.at(i);
        auto xk = X.at(i);
        auto fx = f_X.at(i);

        losses.push_back(s->cost(xk, uk, i));
        cost0 += losses.at(i);

        auto XFABJ = s->forwardPass(xk, uk, i);
        X.push_back(std::get<0>(XFABJ));
        f_X.push_back(std::get<1>(XFABJ));
        As.push_back(std::get<2>(XFABJ));
        Bs.push_back(std::get<3>(XFABJ));
    }
    auto finalLoss = s->cost_F(X.at(horizon - 1));
    cost0 += finalLoss;

    std::vector<MatrixXd> Ks;
    std::vector<VectorXd> ds;

    for (int i = 0; i < nb_iter; i++) {
        auto start = std::chrono::steady_clock::now();

        Ks.clear();
        ds.clear();
        s->reset();

        // BACKWARD PASS
        MatrixXd lN_xx = s->cost_F_xx(X.at(horizon - 1));
        MatrixXd P = lN_xx;
        VectorXd p = s->cost_F_x(X.at(horizon - 1));

        for (int k = horizon - 2; k >= 0; k--) {
            auto xk = X.at(k);
            auto fxk = f_X.at(k);

            auto uk = U.at(k);

            auto A = As.at(k);
            auto B = Bs.at(k);

            MatrixXd Qux = s->cost_ux(xk, uk, k) + B.transpose() * P * A;
            MatrixXd Quu = s->cost_uu(xk, uk, k) + B.transpose() * P * B;
            MatrixXd Qxx = s->cost_xx(xk, uk, k) + A.transpose() * P * A;
            MatrixXd Qxu = s->cost_xu(xk, uk, k) + A.transpose() * P * B;
            VectorXd Qu = s->cost_u(xk, uk, k) + B.transpose() * p;
            VectorXd Qx = s->cost_x(xk, uk, k) + A.transpose() * p;

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
        std::vector<VectorXd> new_X, new_F_X, new_U, new_ds;
        std::vector<MatrixXd> new_Ks;
        VectorXd newCost = VectorXd::Zero(1);
        double du_square_norm = 0;

        do {
            s->reset();
            alpha /= 2.0;

            new_X.clear();
            new_F_X.clear();
            new_U.clear();
            new_Ks.clear();
            new_ds.clear();

            new_X.push_back(x0);
            new_F_X.push_back(std::get<0>(s->getFxJac()));

            du_square_norm = 0;

            newCost << 0;

            for (int k = 0; k < horizon - 1; k++) {
                int k2 = horizon - 2 - k;

                MatrixXd new_Kk = Ks.at(k2);
                VectorXd new_dk = alpha * ds.at(k2);

                // Compute command
                VectorXd du = new_Kk * (new_X.at(k) - X.at(k)) + new_dk;
                du_square_norm += du.norm();
                VectorXd new_uk = U.at(k) + du;

                // Forward pass
                auto XFABJ = s->forwardPass(new_X.at(k), new_uk, k);
                auto xkp1 = std::get<0>(XFABJ);
                auto fxkp1 = std::get<1>(XFABJ);

                new_X.push_back(xkp1);
                new_F_X.push_back(fxkp1);
                new_U.push_back(new_uk);
                new_Ks.push_back(new_Kk);
                new_ds.push_back(new_dk);

                As.at(k) = std::get<2>(XFABJ);
                Bs.at(k) = std::get<3>(XFABJ);

                auto loss = s->cost(new_X.at(k), new_uk, k);
                newCost += loss;
            }

            finalLoss = s->cost_F(new_X.at(horizon - 1));
            newCost += finalLoss;
        } while (((newCost(0) >= cost0(0)) || (newCost.array().isNaN()).any()) && alpha > 1e-3 && line_search);

        cost0 = newCost;
        X = new_X;
        f_X = new_F_X;
        U = new_U;
        Ks = new_Ks;
        ds = new_ds;

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        std::stringstream msg;
        msg << "Iteration " << i + 1 << ", Cost: " << cost0(0) << ", alpha= " << alpha << ", time= " << elapsed_seconds.count();
        if (cb == nullptr)
            std::cout << msg.str() << std::endl;
        else
            cb->notify(msg.str());

        if (early_stop && alpha * sqrt(du_square_norm) < 1e-3 && cost0(0) < 1e-3) {
            break;
        }
    }

    s->reset();
    return std::make_tuple(X, f_X, U, Ks, ds, cost0(0));
}
}  // namespace solver
}  // namespace ilqr_planner
