
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

#include <TinyURDFParser/TinyURDFParser.hpp>

#include "ilqr_planner/sim/KDLRobot.h"
#include "ilqr_planner/utils/utils.h"

using namespace Eigen;

namespace ilqr_planner {
namespace sim {

KDLRobot::KDLRobot(const std::string& urdf,
                   const std::string& base_frame,
                   const std::string& tip_frame,
                   const VectorXd& q,
                   const VectorXd& dq,
                   const VectorXd& transform_rpy,
                   const VectorXd& transform_xyz) {
    this->dof = q.rows();
    this->nbCarDim = 3;
    this->q = q;
    this->dq = dq;
    this->ddq = VectorXd::Zero(this->q.rows());

    this->x = VectorXd::Zero(3);
    this->dx = this->x;

    this->ornQuat = VectorXd::Zero(4);
    this->w = VectorXd::Zero(3);

    this->Jac = MatrixXd::Zero(6, this->dof);
    this->dJac = MatrixXd::Zero(6, this->dof);

    // Read robot model and build kinematic chain
    tup::TinyURDFParser parser(urdf);
    KDL::Chain chain;
    if (parser.setKinematicChain(base_frame, tip_frame)) {
        chain = parser.getKinematicChain();
    } else {
        throw std::runtime_error("[KDLRobot] Unable to build kinematic chain from " + base_frame + " to " + tip_frame);
    }

    // Add virtual frame
    KDL::Rotation virtual_rot = KDL::Rotation::EulerZYX(transform_rpy(0), transform_rpy(1), transform_rpy(2));
    KDL::Vector virtual_pos(transform_xyz(0), transform_xyz(1), transform_xyz(2));
    KDL::Frame virtual_frame(virtual_rot, virtual_pos);
    KDL::Joint virtual_joint;
    KDL::Segment virtual_seg("robot_custom_tip", virtual_joint, virtual_frame);
    chain.addSegment(virtual_seg);

    this->initialize(chain);
    this->updateKinematics();
}

bool KDLRobot::initialize(const KDL::Chain& chain) {
    this->chain_ = chain;
    this->jacobianSolver_ = std::make_shared<KDL::ChainJntToJacSolver>(chain_);
    this->fkSolverPos_ = std::make_shared<KDL::ChainFkSolverPos_recursive>(chain_);

    this->jacobian_ = KDL::Jacobian(chain.getNrOfJoints());
    this->positions_.resize(this->dof);

    return true;
}

void KDLRobot::updateKinematics() {
    // Convert Eigen object to KDL one
    for (size_t i = 0; i < this->dof; ++i) {
        positions_(i) = this->q[i];
    }

    int error = 0;

    error += jacobianSolver_->JntToJac(positions_, jacobian_);
    error += fkSolverPos_->JntToCart(positions_, pose_);

    if (error < 0) {
        throw std::runtime_error("[KinModel] Error while computing Jacobian and FK!");
    }

    for (size_t i = 0; i < 3; i++) {
        this->x(i) = pose_.p[i];
    }

    this->ornQuat = VectorXd::Zero(4);
    pose_.M.GetQuaternion(this->ornQuat[1], this->ornQuat[2], this->ornQuat[3], this->ornQuat[0]);

    this->Jac = MatrixXd::Zero(jacobian_.rows(), jacobian_.columns());
    for (size_t i = 0; i < jacobian_.rows(); i++) {
        for (size_t j = 0; j < jacobian_.columns(); j++) {
            this->Jac(i, j) = jacobian_(i, j);
        }
    }

    this->dJac = getJacobianDerivative<7>(this->Jac, this->dq);
    this->dx = this->Jac.topRows(this->nbCarDim) * this->dq;
    this->w = this->Jac.bottomRows(this->nbCarDim) * this->dq;
}
}  // namespace sim
}  // namespace ilqr_planner