// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * @file ICPSolverLeft.h
 * @brief Left-perturbation ICP solver using tangent (6D) Jacobians.
 *
 * This is a validation-only solver that uses left multiplication (space frame):
 *   T_new = Exp(delta) * T
 *
 * Compare to ICPSolver.h which uses right multiplication (body frame):
 *   T_new = T * Exp(delta)
 *
 * The two approaches should converge to the same solution.
 */

#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <ICP/Correspondences.h>
#include <ICP/EigenTypes.h>
#include <ICP/ICPParams.h>
#include <ICP/JacobiansTangent.h>
#include <ICP/SE3.h>

namespace ICP
{

// ============================================================================
// Left-perturbation SE(3) operations
// ============================================================================

/**
 * @brief SE(3) Plus operation using left-multiplication (space-frame perturbation).
 *
 * Applies a tangent-space increment via left multiplication:
 *   T_new = Exp(delta) * T
 *
 * In components:
 *   R_new = deltaR * R
 *   t_new = deltaR * t + V(w) * v
 *
 * @param x     Current pose [qx, qy, qz, qw, tx, ty, tz]
 * @param delta Tangent increment [v_x, v_y, v_z, w_x, w_y, w_z]
 * @return Updated pose
 */
inline Pose7 se3PlusLeft(const Pose7& x, const Tangent6& delta)
{
    // Extract quaternion and translation
    Quaternion q(x[3], x[0], x[1], x[2]);
    q.normalize();
    Vector3 t = x.tail<3>();

    // Extract tangent components
    Vector3 v = delta.head<3>();
    Vector3 w = delta.tail<3>();

    // Left-multiplicative (space-frame) update:
    //   q_new = Exp(w) ⊗ q  (left multiplication)
    //   t_new = deltaR * t + V(w) * v
    Quaternion dq = quatExpSO3(w);
    Quaternion q_new = (dq * q).normalized();

    // Translation update for left multiplication
    Matrix3 deltaR = dq.toRotationMatrix();
    Matrix3 V = Vso3(w);
    Vector3 t_new = deltaR * t + V * v;

    // Pack result
    Pose7 result;
    result << q_new.x(), q_new.y(), q_new.z(), q_new.w(), t_new;
    return result;
}

/**
 * @brief Apply left SE(3) update and normalize quaternion.
 */
inline void applyDeltaLeftAndNormalize(Pose7& pose, const Tangent6& delta)
{
    pose = se3PlusLeft(pose, delta);
    Quaternion q(pose[3], pose[0], pose[1], pose[2]);
    q.normalize();
    pose.head<4>() << q.x(), q.y(), q.z(), q.w();
}

// ============================================================================
// Left-perturbation normal equations builder
// ============================================================================

/**
 * @brief Build normal equations using left-perturbation Jacobians.
 *
 * Uses ForwardRayLeft and ReverseRayLeft which compute Jacobians directly
 * in 6D tangent space (no projection needed).
 *
 * @tparam JacobianPolicy Jacobian computation policy (simplified or consistent)
 * @param fwdCorrs       Forward correspondences (source to target)
 * @param revCorrs       Reverse correspondences (target to source)
 * @param pose           Current pose [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDirSource   Ray direction in source frame (for forward: transformed to target)
 * @param weighting      Geometry weighting parameters
 * @param[out] H         Hessian approximation (6x6 matrix J^T J)
 * @param[out] b         Gradient vector (6x1 vector J^T r)
 * @param[out] fwd_valid Number of valid forward correspondences used
 * @param[out] rev_valid Number of valid reverse correspondences used
 * @return Sum of squared residuals (cost)
 */
template<typename JacobianPolicy = RayJacobianSimplified>
double buildNormalEquationsLeft(
    const std::vector<Correspondence>& fwdCorrs,
    const std::vector<Correspondence>& revCorrs,
    const Pose7& pose,
    const Vector3& rayDirSource,
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

    // Extract R and t from pose
    Quaternion q(pose[3], pose[0], pose[1], pose[2]);
    q.normalize();
    Matrix3 R = q.toRotationMatrix();
    Vector3 t = pose.tail<3>();

    // Ray direction in target frame (for forward Jacobian)
    Eigen::Vector3f rayDirTarget = (R * rayDirSource).cast<float>();
    Eigen::Vector3f rayDirSrc = rayDirSource.cast<float>();

    // Process forward correspondences (source→target)
    // ForwardRayLeft expects ray direction in target frame
    for (const auto& corr : fwdCorrs)
    {
        ForwardRayLeft<JacobianPolicy> cost(
            corr.srcPoint, corr.tgtPoint, corr.tgtNormal, rayDirTarget, weighting);

        Eigen::Matrix<double, 1, 6> J6;
        double residual = cost.Evaluate(R, t, J6);

        if (std::abs(residual) > 1e10) continue;

        H += J6.transpose() * J6;
        b += J6.transpose() * residual;
        sum_sq += residual * residual;
        fwd_valid++;
    }

    // Process reverse correspondences (target→source)
    // ReverseRayLeft expects ray direction in source frame
    for (const auto& corr : revCorrs)
    {
        ReverseRayLeft<JacobianPolicy> cost(
            corr.srcPoint, corr.tgtPoint, corr.tgtNormal, rayDirSrc, weighting);

        Eigen::Matrix<double, 1, 6> J6;
        double residual = cost.Evaluate(R, t, J6);

        if (std::abs(residual) > 1e10) continue;

        H += J6.transpose() * J6;
        b += J6.transpose() * residual;
        sum_sq += residual * residual;
        rev_valid++;
    }

    return sum_sq;
}

/**
 * @brief Compute cost at a given pose using left-perturbation residuals.
 */
template<typename JacobianPolicy = RayJacobianSimplified>
double computeCostAtPoseLeft(
    const std::vector<Correspondence>& fwdCorrs,
    const std::vector<Correspondence>& revCorrs,
    const Pose7& pose,
    const Vector3& rayDirSource,
    const GeometryWeighting& weighting)
{
    double cost = 0.0;

    // Extract R and t from pose
    Quaternion q(pose[3], pose[0], pose[1], pose[2]);
    q.normalize();
    Matrix3 R = q.toRotationMatrix();
    Vector3 t = pose.tail<3>();

    Eigen::Vector3f rayDirTarget = (R * rayDirSource).cast<float>();
    Eigen::Vector3f rayDirSrc = rayDirSource.cast<float>();

    for (const auto& corr : fwdCorrs)
    {
        ForwardRayLeft<JacobianPolicy> costFn(
            corr.srcPoint, corr.tgtPoint, corr.tgtNormal, rayDirTarget, weighting);
        double residual = costFn.Residual(R, t);
        if (std::abs(residual) < 1e10) cost += residual * residual;
    }

    for (const auto& corr : revCorrs)
    {
        ReverseRayLeft<JacobianPolicy> costFn(
            corr.srcPoint, corr.tgtPoint, corr.tgtNormal, rayDirSrc, weighting);
        double residual = costFn.Residual(R, t);
        if (std::abs(residual) < 1e10) cost += residual * residual;
    }

    return cost;
}

// ============================================================================
// Left-perturbation Gauss-Newton inner solver
// ============================================================================

/**
 * @brief Gauss-Newton inner solver using left-perturbation.
 *
 * Uses left multiplication for pose updates: T_new = Exp(delta) * T
 *
 * @param fwdCorrs   Forward correspondences (source→target rays)
 * @param revCorrs   Reverse correspondences (target→source rays)
 * @param initialPose Initial 7D pose [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDir     Ray direction in source frame
 * @param weighting  Geometry weighting parameters
 * @param params     Inner solver parameters
 * @return           Final pose, RMS, iteration count, convergence status
 */
template<typename JacobianPolicy = RayJacobianSimplified>
InnerSolveResult solveInnerGNLeft(
    const std::vector<Correspondence>& fwdCorrs,
    const std::vector<Correspondence>& revCorrs,
    const Pose7& initialPose,
    const Vector3& rayDir,
    const GeometryWeighting& weighting = GeometryWeighting(),
    const InnerParams& params = InnerParams())
{
    InnerSolveResult result;
    result.pose = initialPose;
    result.iterations = 0;
    result.converged = false;

    Pose7 pose = initialPose;
    const double lambda = params.damping;

    for (int iter = 0; iter < params.maxIterations; ++iter)
    {
        result.iterations++;

        // Build normal equations using left-perturbation Jacobians
        Matrix6 H;
        Vector6 b;
        int fwd_valid, rev_valid;
        double current_cost = buildNormalEquationsLeft<JacobianPolicy>(
            fwdCorrs, revCorrs, pose, rayDir, weighting, H, b, fwd_valid, rev_valid);

        int valid_count = fwd_valid + rev_valid;
        result.rms = valid_count > 0 ? std::sqrt(current_cost / valid_count) : 0.0;
        result.valid_count = valid_count;

        if (params.verbose)
        {
            std::cout << "\t\t\tinner(left) " << iter << ": rms=" << std::scientific
                      << std::setprecision(6) << result.rms
                      << ", cost=" << current_cost
                      << ", fwd=" << fwd_valid << ", rev=" << rev_valid << "\n";
        }

        if (valid_count < 6)
        {
            result.pose = pose;
            return result;
        }

        // Solve with damping: (H + lambda*I) delta = -b
        H.diagonal().array() += lambda;
        Tangent6 delta = -H.ldlt().solve(b);

        // Apply step using LEFT multiplication
        applyDeltaLeftAndNormalize(pose, delta);

        // Check convergence
        if (isDeltaConverged(delta, params.translationThreshold, params.rotationThreshold))
        {
            result.converged = true;
            break;
        }
    }

    result.pose = pose;
    return result;
}

/**
 * @brief Full ICP solver using left-perturbation.
 */
template<typename JacobianPolicy = RayJacobianSimplified>
ICPResult solveICPLeft(
    const Grid& source,
    const Grid& target,
    const Pose7& initialPose,
    const Vector3& rayDir,
    const GeometryWeighting& weighting = GeometryWeighting(),
    const InnerParams& innerParams = InnerParams(),
    const OuterParams& outerParams = OuterParams())
{
    ICPResult result;
    result.pose = initialPose;
    result.outer_iterations = 0;
    result.total_inner_iterations = 0;
    result.converged = false;

    Pose7 pose = initialPose;
    double prev_rms = std::numeric_limits<double>::max();

    for (int outer = 0; outer < outerParams.maxIterations; ++outer)
    {
        result.outer_iterations++;

        // Compute correspondences at current pose
        Eigen::Isometry3d srcToTgt = pose7ToIsometry(pose);
        auto corrs = computeBidirectionalCorrs(
            source, target, rayDir.cast<float>(), srcToTgt, outerParams.maxDist);

        if (outerParams.verbose)
        {
            std::cout << "\touter(left) " << outer << ":\n";
            std::cout << "\t\tfwd_corrs=" << corrs.forward.size()
                      << ", rev_corrs=" << corrs.reverse.size() << "\n";
        }

        // Inner loop with fixed correspondences
        auto innerResult = solveInnerGNLeft<JacobianPolicy>(
            corrs.forward, corrs.reverse, pose, rayDir, weighting, innerParams);

        pose = innerResult.pose;
        result.rms = innerResult.rms;
        result.total_inner_iterations += innerResult.iterations;

        if (outerParams.verbose)
        {
            std::cout << "\t\touter(left) " << outer << ": rms=" << std::scientific
                      << std::setprecision(6) << result.rms << ", valid=" << innerResult.valid_count
                      << ", inner_iters=" << innerResult.iterations << "\n";
        }

        // Check convergence
        double rms_change = std::abs(prev_rms - result.rms);
        if (result.rms < outerParams.convergenceTol ||
            rms_change < outerParams.convergenceTol * result.rms)
        {
            result.converged = true;
            break;
        }
        prev_rms = result.rms;
    }

    if (outerParams.verbose)
    {
        std::cout << "Total inner iterations (left): " << result.total_inner_iterations << "\n";
    }

    result.pose = pose;
    return result;
}

} // namespace ICP
