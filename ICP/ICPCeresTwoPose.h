// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once
/**
 * Ceres-based two-pose ICP solver.
 *
 * Optimizes two poses simultaneously using bidirectional correspondences.
 * Uses ForwardRayCostTwoPose and ReverseRayCostTwoPose as Ceres cost functions
 * with <1, 7, 7> dimensions.
 */

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
#include <ICP/Grid.h>
#include <ICP/ICPCeresSolver.h>
#include <ICP/ICPParams.h>
#include <ICP/JacobiansAmbientTwoPose.h>
#include <ICP/SE3.h>

namespace ICP
{

/// Result from two-pose inner solver
struct CeresTwoPoseInnerResult
{
    Pose7 poseA;                 // Final pose A after optimization
    Pose7 poseB;                 // Final pose B after optimization
    double rms;                  // Final RMS
    int iterations;              // Number of Ceres iterations performed
    int valid_count;             // Number of valid correspondences
    bool converged;              // Whether Ceres solver converged
    ceres::Solver::Summary summary;  // Full Ceres solver summary
};

/// Result from full two-pose ICP solve
struct CeresTwoPoseResult
{
    Pose7 poseA;          // Final pose A
    Pose7 poseB;          // Final pose B
    double rms;           // Final RMS
    int outer_iterations;
    int total_inner_iterations;
    bool converged;
};

/**
 * Inner solver using Ceres for two-pose optimization.
 *
 * Processes bidirectional correspondences:
 * - Forward: rays from grid A to grid B (ForwardRayCostTwoPose)
 * - Reverse: rays from grid B to grid A (ReverseRayCostTwoPose)
 *
 * @param fwdCorrs     Forward correspondences (A→B rays)
 * @param revCorrs     Reverse correspondences (B→A rays)
 * @param initialPoseA Initial 7D pose for grid A [qx, qy, qz, qw, tx, ty, tz]
 * @param initialPoseB Initial 7D pose for grid B [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDirA      Ray direction in grid A frame (typically [0,0,-1])
 * @param rayDirB      Ray direction in grid B frame (typically [0,0,-1])
 * @param weighting    Geometry weighting parameters
 * @param ceresOptions Ceres solver options
 * @param fixPoseA     If true, hold pose A fixed (only optimize B)
 * @return             Final poses, RMS, iteration count, convergence status
 */
template<typename JacobianPolicy = RayJacobianSimplified>
CeresTwoPoseInnerResult solveInnerCeresTwoPose(
    const std::vector<Correspondence>& fwdCorrs,
    const std::vector<Correspondence>& revCorrs,
    const Pose7& initialPoseA,
    const Pose7& initialPoseB,
    const Vector3& rayDirA,
    const Vector3& rayDirB,
    const GeometryWeighting& weighting,
    const ceres::Solver::Options& ceresOptions,
    bool fixPoseA = false)
{
    CeresTwoPoseInnerResult result;
    result.iterations = 0;
    result.converged = false;

    // Convert Pose7 to Sophus::SE3d for pose A
    Quaternion qA0(initialPoseA[3], initialPoseA[0], initialPoseA[1], initialPoseA[2]);
    Vector3 tA0(initialPoseA[4], initialPoseA[5], initialPoseA[6]);
    Sophus::SE3d poseA(qA0, tA0);
    double* poseA_data = poseA.data();

    // Convert Pose7 to Sophus::SE3d for pose B
    Quaternion qB0(initialPoseB[3], initialPoseB[0], initialPoseB[1], initialPoseB[2]);
    Vector3 tB0(initialPoseB[4], initialPoseB[5], initialPoseB[6]);
    Sophus::SE3d poseB(qB0, tB0);
    double* poseB_data = poseB.data();

    // Build Ceres problem
    ceres::Problem problem;

    // Set SE3 manifold for both poses
    problem.AddParameterBlock(poseA_data, Sophus::SE3d::num_parameters, new SophusSE3Manifold());
    problem.AddParameterBlock(poseB_data, Sophus::SE3d::num_parameters, new SophusSE3Manifold());

    // Optionally fix pose A (useful for anchoring one grid)
    if (fixPoseA)
    {
        problem.SetParameterBlockConstant(poseA_data);
    }

    double cosAngleThresh = weighting.enable_gate ? weighting.tau : 0.0;

    // Add forward residuals (rays from A → B)
    for (const auto& c : fwdCorrs)
    {
        // c.srcPoint is in frame A, c.tgtPoint/tgtNormal are in frame B
        double nDotD = c.tgtNormal.cast<double>().dot(rayDirA);
        if (std::abs(nDotD) < cosAngleThresh)
            continue;

        problem.AddResidualBlock(
            ForwardRayCostTwoPose<JacobianPolicy>::Create(
                c.srcPoint.cast<double>(),   // pA: point in frame A
                c.tgtPoint.cast<double>(),   // qB: surface point in frame B
                c.tgtNormal.cast<double>(),  // nB: surface normal in frame B
                rayDirA,
                weighting),
            nullptr, poseA_data, poseB_data);
    }

    // Add reverse residuals (rays from B → A)
    for (const auto& c : revCorrs)
    {
        // For reverse: c.srcPoint is in frame B, c.tgtPoint/tgtNormal are in frame A
        double nDotD = c.tgtNormal.cast<double>().dot(rayDirB);
        if (std::abs(nDotD) < cosAngleThresh)
            continue;

        problem.AddResidualBlock(
            ReverseRayCostTwoPose<JacobianPolicy>::Create(
                c.srcPoint.cast<double>(),   // pB: point in frame B
                c.tgtPoint.cast<double>(),   // qA: surface point in frame A
                c.tgtNormal.cast<double>(),  // nA: surface normal in frame A
                rayDirB,
                weighting),
            nullptr, poseA_data, poseB_data);
    }

    result.valid_count = problem.NumResidualBlocks();

    if (result.valid_count == 0)
    {
        result.poseA = initialPoseA;
        result.poseB = initialPoseB;
        result.rms = std::numeric_limits<double>::max();
        return result;
    }

    // Solve
    ceres::Solve(ceresOptions, &problem, &result.summary);

    result.iterations = result.summary.iterations.size();
    result.converged = (result.summary.termination_type == ceres::CONVERGENCE);
    // Ceres uses 0.5 * sum(r²), so multiply by 2 to get sum_sq for RMS
    result.rms = std::sqrt(2.0 * result.summary.final_cost / result.valid_count);

    // Convert Sophus::SE3d back to Pose7 for pose A
    Quaternion qA_final = poseA.unit_quaternion();
    Vector3 tA_final = poseA.translation();
    result.poseA << qA_final.x(), qA_final.y(), qA_final.z(), qA_final.w(),
                    tA_final.x(), tA_final.y(), tA_final.z();

    // Convert Sophus::SE3d back to Pose7 for pose B
    Quaternion qB_final = poseB.unit_quaternion();
    Vector3 tB_final = poseB.translation();
    result.poseB << qB_final.x(), qB_final.y(), qB_final.z(), qB_final.w(),
                    tB_final.x(), tB_final.y(), tB_final.z();

    return result;
}

/**
 * Compute relative pose from A to B: T_B^{-1} * T_A
 * This transforms points from A's local frame to B's local frame.
 */
inline Pose7 computeRelativePose(const Pose7& poseA, const Pose7& poseB)
{
    // Convert to Isometry3d
    Quaternion qA(poseA[3], poseA[0], poseA[1], poseA[2]);
    Vector3 tA(poseA[4], poseA[5], poseA[6]);
    Eigen::Isometry3d isoA = Eigen::Isometry3d::Identity();
    isoA.linear() = qA.normalized().toRotationMatrix();
    isoA.translation() = tA;

    Quaternion qB(poseB[3], poseB[0], poseB[1], poseB[2]);
    Vector3 tB(poseB[4], poseB[5], poseB[6]);
    Eigen::Isometry3d isoB = Eigen::Isometry3d::Identity();
    isoB.linear() = qB.normalized().toRotationMatrix();
    isoB.translation() = tB;

    // Compute srcToTgt: B^{-1} * A (transforms points from A to B frame)
    Eigen::Isometry3d rel = isoB.inverse() * isoA;

    // Convert back to Pose7
    Quaternion qRel(rel.rotation());
    qRel.normalize();
    Pose7 result;
    result << qRel.x(), qRel.y(), qRel.z(), qRel.w(),
              rel.translation().x(), rel.translation().y(), rel.translation().z();
    return result;
}

/**
 * Full two-pose Ceres ICP solver with outer loop.
 *
 * @param gridA         First grid
 * @param gridB         Second grid
 * @param initialPoseA  Initial pose for grid A
 * @param initialPoseB  Initial pose for grid B
 * @param rayDir        Ray direction in local frame (typically [0,0,-1])
 * @param weighting     Geometry weighting parameters
 * @param innerParams   Inner loop parameters (converted to Ceres options)
 * @param outerParams   Outer loop parameters
 * @param fixPoseA      If true, hold pose A fixed
 * @return              ICP result with final poses and statistics
 */
template<typename JacobianPolicy = RayJacobianSimplified>
CeresTwoPoseResult solveICPCeresTwoPose(
    const Grid& gridA,
    const Grid& gridB,
    const Pose7& initialPoseA,
    const Pose7& initialPoseB,
    const Vector3& rayDir,
    const GeometryWeighting& weighting,
    const InnerParams& innerParams,
    const OuterParams& outerParams,
    bool fixPoseA = false)
{
    CeresTwoPoseResult result;
    result.poseA = initialPoseA;
    result.poseB = initialPoseB;
    result.outer_iterations = 0;
    result.total_inner_iterations = 0;
    result.converged = false;

    // Convert InnerParams to Ceres options once
    ceres::Solver::Options ceresOptions = toCeresSolverOptions(innerParams);

    Pose7 poseA = initialPoseA;
    Pose7 poseB = initialPoseB;
    double prev_rms = std::numeric_limits<double>::max();

    for (int outer = 0; outer < outerParams.maxIterations; ++outer)
    {
        result.outer_iterations++;

        // Compute relative pose for correspondence computation: A_to_B = A * B^{-1}
        Pose7 relPose = computeRelativePose(poseA, poseB);
        Eigen::Isometry3d aToB = pose7ToIsometry(relPose);

        // Compute bidirectional correspondences
        // Forward: rays from A hitting B
        // Reverse: rays from B hitting A
        auto corrs = computeBidirectionalCorrs(
            gridA, gridB, rayDir.cast<float>(), aToB, outerParams.maxDist,
            outerParams.subsampleX, outerParams.subsampleY);

        if (outerParams.verbose)
        {
            std::cout << "\touter " << outer << ":\n";
            std::cout << "\t\tfwd_corrs=" << corrs.forward.size()
                      << ", rev_corrs=" << corrs.reverse.size() << "\n";
        }

        // Inner loop with Ceres two-pose solver
        auto innerResult = solveInnerCeresTwoPose<JacobianPolicy>(
            corrs.forward, corrs.reverse,
            poseA, poseB,
            rayDir, rayDir,
            weighting, ceresOptions, fixPoseA);

        poseA = innerResult.poseA;
        poseB = innerResult.poseB;
        result.rms = innerResult.rms;
        result.total_inner_iterations += innerResult.iterations;

        if (outerParams.verbose)
        {
            std::cout << "\t\touter " << outer << ": rms=" << std::scientific
                      << std::setprecision(6) << result.rms << ", valid=" << innerResult.valid_count
                      << ", ceres_iters=" << innerResult.iterations << "\n";
        }

        // Check convergence based on RMS change
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
        std::cout << "Total Ceres TwoPose iterations: " << result.total_inner_iterations << "\n";
    }

    result.poseA = poseA;
    result.poseB = poseB;
    return result;
}

} // namespace ICP
