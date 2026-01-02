// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * @file ICPHelpers.h
 * @brief Helper functions for ICP solvers.
 * @author Jens Munk Hansen
 * @date 2024-2025
 *
 * @details This header provides shared utility functions used by both
 * Gauss-Newton and Levenberg-Marquardt ICP solvers. Key functionality includes:
 *
 * - **Cost computation**: Evaluate residual cost at a given SE(3) pose
 * - **Normal equations**: Build the Hessian approximation (J^T J) and
 *   gradient (J^T r) for least-squares optimization
 * - **Convergence checking**: Determine if the tangent space update is
 *   below translation and rotation thresholds
 * - **Line search**: Backtracking line search for step size selection
 * - **Pose updates**: Apply SE(3) updates with quaternion normalization
 *
 * All functions operate on the 7D ambient pose representation
 * [qx, qy, qz, qw, tx, ty, tz] with 6D tangent space updates.
 */

#pragma once

// Standard C++ headers
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Internal headers
#include <ICP/Correspondences.h>
#include <ICP/EigenTypes.h>
#include <ICP/ICPParams.h>
#include <ICP/JacobiansAmbient.h>
#include <ICP/SE3.h>

namespace ICP
{

/**
 * @brief Result from the inner solver after iterating to convergence.
 *
 * Contains the final optimized pose and convergence statistics
 * from a single inner loop execution.
 */
struct InnerSolveResult
{
    Pose7 pose;       ///< Final pose after inner iterations [qx, qy, qz, qw, tx, ty, tz]
    double rms;       ///< Final root-mean-square residual
    int iterations;   ///< Number of inner iterations performed
    int valid_count;  ///< Number of valid correspondences used
    bool converged;   ///< True if step tolerance was reached before max iterations
};

/**
 * @brief Compute cost at a given pose.
 *
 * Evaluates the sum of squared residuals for all correspondences
 * at the specified SE(3) pose. Used for cost evaluation in line
 * search and convergence checking.
 *
 * @tparam JacobianPolicy Jacobian computation policy (simplified or consistent)
 * @param fwdCorrs   Forward correspondences (source to target)
 * @param revCorrs   Reverse correspondences (target to source)
 * @param pose       7D pose to evaluate [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDir     Ray direction for projection
 * @param weighting  Geometry weighting parameters for incidence handling
 * @return Sum of squared residuals (cost)
 */
template<typename JacobianPolicy = RayJacobianSimplified>
double computeCostAtPose(
    const std::vector<Correspondence>& fwdCorrs,
    const std::vector<Correspondence>& revCorrs,
    const Pose7& pose,
    const Vector3& rayDir,
    const GeometryWeighting& weighting)
{
    double cost = 0.0;

    for (const auto& corr : fwdCorrs)
    {
        Vector3 pS = corr.srcPoint.cast<double>();
        Vector3 qT = corr.tgtPoint.cast<double>();
        Vector3 nT = corr.tgtNormal.cast<double>();
        ForwardRayCost<JacobianPolicy> costFn(pS, qT, nT, rayDir, weighting);
        double const* pparams[1] = {pose.data()};
        double residual;
        costFn(pparams, &residual, nullptr);
        if (std::abs(residual) < 1e10) cost += residual * residual;
    }

    for (const auto& corr : revCorrs)
    {
        Vector3 pT = corr.srcPoint.cast<double>();
        Vector3 qS = corr.tgtPoint.cast<double>();
        Vector3 nS = corr.tgtNormal.cast<double>();
        ReverseRayCost<JacobianPolicy> costFn(pT, qS, nS, rayDir, weighting);
        double const* pparams[1] = {pose.data()};
        double residual;
        costFn(pparams, &residual, nullptr);
        if (std::abs(residual) < 1e10) cost += residual * residual;
    }

    return cost;
}

/**
 * @brief Check if delta is below convergence thresholds.
 *
 * Uses separate thresholds for translation and rotation components
 * to determine if the optimization has converged.
 *
 * @param delta          Tangent space update [v_x, v_y, v_z, w_x, w_y, w_z]
 * @param transThreshold Translation threshold (same units as data, e.g., mm)
 * @param rotThreshold   Rotation threshold (radians)
 * @return True if both translation and rotation norms are below their thresholds
 */
inline bool isDeltaConverged(const Tangent6& delta, double transThreshold, double rotThreshold)
{
    double transNorm = delta.head<3>().norm();
    double rotNorm = delta.tail<3>().norm();
    return transNorm < transThreshold && rotNorm < rotThreshold;
}

/**
 * @brief Apply SE(3) update and normalize quaternion.
 *
 * Updates the pose using right-multiplication: T_new = T * Exp(delta).
 * The quaternion component is normalized after the update.
 *
 * @param[in,out] pose Current pose, modified in place
 * @param delta        Tangent space update [v_x, v_y, v_z, w_x, w_y, w_z]
 */
inline void applyDeltaAndNormalize(Pose7& pose, const Tangent6& delta)
{
    pose = se3Plus(pose, delta);
    Quaternion q(pose[3], pose[0], pose[1], pose[2]);
    q.normalize();
    pose.head<4>() << q.x(), q.y(), q.z(), q.w();
}

/**
 * @brief Backtracking line search to find optimal step size.
 *
 * Implements simple backtracking line search to find a step size
 * that reduces the cost. Starts with alpha=1 and reduces by factor
 * beta until cost decreases or maximum iterations reached.
 *
 * @tparam JacobianPolicy Jacobian computation policy (simplified or consistent)
 * @param fwdCorrs     Forward correspondences (source to target)
 * @param revCorrs     Reverse correspondences (target to source)
 * @param pose         Current pose [qx, qy, qz, qw, tx, ty, tz]
 * @param delta        Full step direction in tangent space
 * @param currentCost  Cost at current pose (for comparison)
 * @param rayDir       Ray direction for projection
 * @param weighting    Geometry weighting parameters
 * @param params       Line search configuration (alpha, beta, maxIterations)
 * @return Optimal step size alpha in (0, 1]
 */
template<typename JacobianPolicy = RayJacobianSimplified>
double lineSearch(
    const std::vector<Correspondence>& fwdCorrs,
    const std::vector<Correspondence>& revCorrs,
    const Pose7& pose,
    const Tangent6& delta,
    double currentCost,
    const Vector3& rayDir,
    const GeometryWeighting& weighting,
    const LineSearchParams& params)
{
    double alpha = params.alpha;

    for (int i = 0; i < params.maxIterations; ++i)
    {
        // Trial step with scaled delta
        Tangent6 scaledDelta = alpha * delta;
        Pose7 pose_trial = pose;
        applyDeltaAndNormalize(pose_trial, scaledDelta);

        double trialCost = computeCostAtPose<JacobianPolicy>(
            fwdCorrs, revCorrs, pose_trial, rayDir, weighting);

        if (trialCost < currentCost)
        {
            return alpha;
        }

        alpha *= params.beta;
    }

    return alpha;  // Return smallest tried alpha
}

/**
 * @brief Build normal equations and compute cost in one pass.
 *
 * Accumulates the Gauss-Newton normal equations by iterating over
 * all correspondences. Computes the Hessian approximation H = J^T J,
 * gradient b = J^T r, and sum of squared residuals in a single pass.
 *
 * The Jacobians are computed in the 7D ambient space and projected
 * to 6D tangent space using the plus-Jacobian.
 *
 * @tparam JacobianPolicy Jacobian computation policy (simplified or consistent)
 * @param fwdCorrs       Forward correspondences (source to target)
 * @param revCorrs       Reverse correspondences (target to source)
 * @param pose           Current pose [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDir         Ray direction for projection
 * @param weighting      Geometry weighting parameters
 * @param[out] H         Hessian approximation (6x6 matrix J^T J)
 * @param[out] b         Gradient vector (6x1 vector J^T r)
 * @param[out] fwd_valid Number of valid forward correspondences used
 * @param[out] rev_valid Number of valid reverse correspondences used
 * @return Sum of squared residuals (cost)
 */
template<typename JacobianPolicy = RayJacobianSimplified>
double buildNormalEquations(
    const std::vector<Correspondence>& fwdCorrs,
    const std::vector<Correspondence>& revCorrs,
    const Pose7& pose,
    const Vector3& rayDir,
    const GeometryWeighting& weighting,
    Matrix6& H,
    Vector6& b,
    int& fwd_valid,
    int& rev_valid)
{
    H.setZero();
    b.setZero();
    double sum_sq = 0.0;
    fwd_valid = 0;
    rev_valid = 0;

    Matrix7x6 P = plusJacobian7x6(pose);

    // Process forward correspondences (source→target)
    for (const auto& corr : fwdCorrs)
    {
        Vector3 pS = corr.srcPoint.cast<double>();
        Vector3 qT = corr.tgtPoint.cast<double>();
        Vector3 nT = corr.tgtNormal.cast<double>();

        ForwardRayCost<JacobianPolicy> cost(pS, qT, nT, rayDir, weighting);

        double const* pparams[1] = {pose.data()};
        double residual;
        double J7[7];
        double* jacobians[1] = {J7};

        cost(pparams, &residual, jacobians);

        if (std::abs(residual) > 1e10) continue;

        Eigen::Map<Eigen::RowVectorXd> J7_map(J7, 7);
        Eigen::Matrix<double, 1, 6> J6 = J7_map * P;

        H += J6.transpose() * J6;
        b += J6.transpose() * residual;
        sum_sq += residual * residual;
        fwd_valid++;
    }

    // Process reverse correspondences (target→source)
    for (const auto& corr : revCorrs)
    {
        Vector3 pT = corr.srcPoint.cast<double>();
        Vector3 qS = corr.tgtPoint.cast<double>();
        Vector3 nS = corr.tgtNormal.cast<double>();

        ReverseRayCost<JacobianPolicy> cost(pT, qS, nS, rayDir, weighting);

        double const* pparams[1] = {pose.data()};
        double residual;
        double J7[7];
        double* jacobians[1] = {J7};

        cost(pparams, &residual, jacobians);

        if (std::abs(residual) > 1e10) continue;

        Eigen::Map<Eigen::RowVectorXd> J7_map(J7, 7);
        Eigen::Matrix<double, 1, 6> J6 = J7_map * P;

        H += J6.transpose() * J6;
        b += J6.transpose() * residual;
        sum_sq += residual * residual;
        rev_valid++;
    }

    return sum_sq;
}

} // namespace ICP
