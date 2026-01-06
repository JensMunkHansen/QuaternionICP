// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * @file ICPCeresSolver.h
 * @brief Ceres-based ICP solver with inner and outer loops.
 *
 * This header provides a Ceres Solver-based implementation of the ICP algorithm
 * using the Sophus SE(3) manifold for proper Lie group optimization.
 *
 * @section icpceres_architecture Architecture
 *
 * - **Inner loop**: Build Ceres problem with fixed correspondences, solve using
 *   Levenberg-Marquardt or other configured solver.
 * - **Outer loop**: Recompute bidirectional correspondences at current pose estimate,
 *   run inner loop, check for convergence.
 *
 * @section icpceres_costs Cost Functions
 *
 * Uses ray-projection cost functions from JacobiansAmbient.h:
 * - `ForwardRayCost`: Rays from source points projected onto target surface
 * - `ReverseRayCost`: Rays from target points projected onto source surface
 *
 * @section icpceres_manifold Manifold
 *
 * Uses `Sophus::Manifold<Sophus::SE3>` for proper SE(3) optimization,
 * ensuring poses remain on the manifold during optimization.
 */

#pragma once

// Standard C++ headers
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// Ceres headers
#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

// Sophus headers
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>

// Internal headers
#include <ICP/Correspondences.h>
#include <ICP/EigenTypes.h>
#include <ICP/EigenUtils.h>
#include <ICP/Grid.h>
#include <ICP/ICPParams.h>
#include <ICP/JacobiansAmbient.h>
#include <ICP/SE3.h>
#include <ICP/SE3CeresManifold.h>

namespace ICP
{

// Note: Use toCeresSolverOptions() from ICPParams.h to convert InnerParams to ceres::Solver::Options

/// Sophus SE3 manifold for Ceres (used by two-pose solver)
using SophusSE3Manifold = Sophus::Manifold<Sophus::SE3>;

/// Result from inner solver (after Ceres optimization)
struct CeresInnerSolveResult
{
    Pose7 pose;                  // Final pose after optimization
    double rms;                  // Final RMS
    int iterations;              // Number of Ceres iterations performed
    int valid_count;             // Number of valid correspondences
    bool converged;              // Whether Ceres solver converged
    ceres::Solver::Summary summary;  // Full Ceres solver summary
};

/// Result from full Ceres ICP solve
struct CeresICPResult
{
    Pose7 pose;           // Final pose
    double rms;           // Final RMS
    int outer_iterations;
    int total_inner_iterations;
    bool converged;
};

/**
 * Inner solver using Ceres: build problem with fixed correspondences.
 *
 * Processes bidirectional correspondences:
 * - Forward: rays from source to target (ForwardRayCost)
 * - Reverse: rays from target to source (ReverseRayCost)
 *
 * @param fwdCorrs     Forward correspondences (source→target rays)
 * @param revCorrs     Reverse correspondences (target→source rays)
 * @param initialPose  Initial 7D pose [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDir       Ray direction in local frame (typically [0,0,-1])
 * @param weighting    Geometry weighting parameters
 * @param innerParams  Inner solver parameters (iterations, tolerances, etc.)
 * @tparam JacobianPolicy Jacobian computation policy (default: RayJacobianSimplified)
 * @return Final pose, RMS, iteration count, convergence status
 */
template<typename JacobianPolicy = RayJacobianSimplified>
CeresInnerSolveResult solveInnerCeres(
    const std::vector<Correspondence>& fwdCorrs,
    const std::vector<Correspondence>& revCorrs,
    const Pose7& initialPose,
    const Vector3& rayDir,
    const GeometryWeighting& weighting = GeometryWeighting(),
    const InnerParams& innerParams = InnerParams())
{
    CeresInnerSolveResult result;
    result.iterations = 0;
    result.converged = false;

    // Convert Pose7 to Sophus::SE3d
    Quaternion q0(initialPose[3], initialPose[0], initialPose[1], initialPose[2]);
    Vector3 t0(initialPose[4], initialPose[5], initialPose[6]);
    Sophus::SE3d pose(q0, t0);
    double* pose_data = pose.data();

    // Build Ceres problem
    ceres::Problem problem;

    // Set SE3 manifold for proper Lie group optimization
    if (innerParams.useCustomManifold)
    {
        // Use our custom manifold with rotation scaling support
        problem.AddParameterBlock(pose_data, 7, new SE3ScaledManifold(innerParams.rotationScale));
    }
    else
    {
        // Use Sophus manifold (ignores rotationScale, useful for comparison)
        problem.AddParameterBlock(pose_data, Sophus::SE3d::num_parameters, new SophusSE3Manifold());
    }

    Vector3 rayDirD = rayDir.cast<double>();
    double cosAngleThresh = weighting.enable_gate ? weighting.tau : 0.0;

    // Add forward residuals
    for (const auto& c : fwdCorrs)
    {
        double nDotD = c.tgtNormal.cast<double>().dot(rayDirD);
        if (std::abs(nDotD) < cosAngleThresh)
            continue;

        problem.AddResidualBlock(
            ForwardRayCost<JacobianPolicy>::Create(
                c.srcPoint.cast<double>(),
                c.tgtPoint.cast<double>(),
                c.tgtNormal.cast<double>(),
                rayDir,
                weighting),
            nullptr, pose_data);
    }

    // Add reverse residuals
    for (const auto& c : revCorrs)
    {
        double nDotD = c.tgtNormal.cast<double>().dot(rayDirD);
        if (std::abs(nDotD) < cosAngleThresh)
            continue;

        problem.AddResidualBlock(
            ReverseRayCost<JacobianPolicy>::Create(
                c.srcPoint.cast<double>(),  // tgtPoint (q)
                c.tgtPoint.cast<double>(),  // srcPoint (p)
                c.tgtNormal.cast<double>(), // srcNormal
                rayDir,
                weighting),
            nullptr, pose_data);
    }

    result.valid_count = problem.NumResidualBlocks();

    if (result.valid_count == 0)
    {
        result.pose = initialPose;
        result.rms = std::numeric_limits<double>::max();
        return result;
    }

    // Configure and solve
    ceres::Solver::Options options = toCeresSolverOptions(innerParams);

    ceres::Solve(options, &problem, &result.summary);

    result.iterations = result.summary.iterations.size();
    result.converged = (result.summary.termination_type == ceres::CONVERGENCE);
    // Ceres uses 0.5 * sum(r²), so multiply by 2 to get sum_sq for RMS
    result.rms = std::sqrt(2.0 * result.summary.final_cost / result.valid_count);

    // Convert Sophus::SE3d back to Pose7
    Quaternion q_final = pose.unit_quaternion();
    Vector3 t_final = pose.translation();

    Pose7 finalPose;
    finalPose << q_final.x(), q_final.y(), q_final.z(), q_final.w(),
                 t_final.x(), t_final.y(), t_final.z();

    result.pose = finalPose;

    if (innerParams.verbose)
    {
        std::cout << "\t\tCeres: " << result.iterations << " iterations, "
                  << "cost=" << 2.0 * result.summary.final_cost  // Convert from Ceres's 0.5*sum to sum
                  << ", rms=" << result.rms << "\n";
    }

    return result;
}

/**
 * Full Ceres ICP solver with outer loop.
 *
 * @param source       Source grid
 * @param target       Target grid
 * @param initialPose  Initial pose estimate
 * @param rayDir       Ray direction in local frame (typically [0,0,-1])
 * @param weighting    Geometry weighting parameters
 * @param innerParams  Inner solver parameters
 * @param outerParams  Outer loop parameters
 * @tparam JacobianPolicy Jacobian computation policy (default: RayJacobianSimplified)
 * @return ICP result with final pose and statistics
 */
template<typename JacobianPolicy = RayJacobianSimplified>
CeresICPResult solveICPCeres(
    const Grid& source,
    const Grid& target,
    const Pose7& initialPose,
    const Vector3& rayDir,
    const GeometryWeighting& weighting = GeometryWeighting(),
    const InnerParams& innerParams = InnerParams(),
    const OuterParams& outerParams = OuterParams())
{
    CeresICPResult result;
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
            std::cout << "\touter " << outer << ":\n";
            std::cout << "\t\tfwd_corrs=" << corrs.forward.size()
                      << ", rev_corrs=" << corrs.reverse.size() << "\n";
        }

        // Inner loop with Ceres
        auto innerResult = solveInnerCeres<JacobianPolicy>(
            corrs.forward, corrs.reverse, pose, rayDir, weighting, innerParams);

        pose = innerResult.pose;
        result.rms = innerResult.rms;
        result.total_inner_iterations += innerResult.iterations;

        if (outerParams.verbose)
        {
            std::cout << "\t\touter " << outer << ": rms=" << std::scientific
                      << std::setprecision(6) << result.rms << ", valid=" << innerResult.valid_count
                      << ", ceres_iters=" << innerResult.iterations << "\n";
        }

        // Check convergence based on RMS change across outer iterations
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
        std::cout << "Total Ceres iterations: " << result.total_inner_iterations << "\n";
    }

    result.pose = pose;
    return result;
}

} // namespace ICP
