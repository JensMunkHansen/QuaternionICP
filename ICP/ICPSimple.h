#pragma once
/**
 * Simple ICP solver with inner and outer loops.
 *
 * Inner loop: Build and solve normal equations with fixed correspondences.
 * Outer loop: Recompute correspondences, run inner loop, repeat.
 *
 * Uses ForwardRayCost and ReverseRayCost for bidirectional correspondences.
 */

#include <ICP/EigenTypes.h>
#include <ICP/SE3.h>
#include <ICP/JacobiansAmbient.h>
#include <ICP/ICPParams.h>
#include <ICP/Correspondences.h>
#include <ICP/Grid.h>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

namespace ICP
{

/// Parameters for inner solver (fixed correspondences)
struct InnerParams
{
    int maxIterations = 12;
    double stepTol = 1e-9;
    double damping = 0.0;  // LM damping (0 = Gauss-Newton)
};

/// Parameters for outer loop (correspondence updates)
struct OuterParams
{
    int maxIterations = 6;
    double convergenceTol = 1e-9;  // Relative RMS change threshold
    float maxDist = 100.0f;        // Max ray distance for correspondences
};

/// Result from inner solver (after iterating to convergence)
struct InnerSolveResult
{
    Pose7 pose;       // Final pose after inner iterations
    double rms;       // Final RMS
    int iterations;   // Number of inner iterations performed
    int valid_count;  // Number of valid correspondences
    bool converged;   // Whether step tolerance was reached
};

/// Result from full ICP solve
struct ICPResult
{
    Pose7 pose;           // Final pose
    double rms;           // Final RMS
    int outer_iterations;
    int total_inner_iterations;
    bool converged;
};

/**
 * Inner solver: iterate with fixed correspondences until convergence.
 *
 * Processes bidirectional correspondences:
 * - Forward: rays from source to target (ForwardRayCost)
 * - Reverse: rays from target to source (ReverseRayCost)
 *
 * @param fwdCorrs   Forward correspondences (source→target rays)
 * @param revCorrs   Reverse correspondences (target→source rays)
 * @param initialPose Initial 7D pose [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDir     Ray direction in local frame (same for both, typically [0,0,-1])
 * @param weighting  Geometry weighting parameters
 * @param params     Inner solver parameters (iterations, tolerance, damping)
 * @return           Final pose, RMS, iteration count, convergence status
 */
template<typename JacobianPolicy = RayJacobianSimplified>
InnerSolveResult solveInner(
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

    for (int iter = 0; iter < params.maxIterations; ++iter)
    {
        result.iterations++;

        Matrix6 AtA = Matrix6::Zero();
        Vector6 Atb = Vector6::Zero();
        double sum_sq = 0.0;
        int valid_count = 0;

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

            AtA += J6.transpose() * J6;
            Atb += -J6.transpose() * residual;
            sum_sq += residual * residual;
            valid_count++;
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

            AtA += J6.transpose() * J6;
            Atb += -J6.transpose() * residual;
            sum_sq += residual * residual;
            valid_count++;
        }

        result.rms = valid_count > 0 ? std::sqrt(sum_sq / valid_count) : 0.0;
        result.valid_count = valid_count;

        if (valid_count < 6)
        {
            result.pose = pose;
            return result;
        }

        AtA.diagonal().array() += params.damping;
        Tangent6 delta = AtA.ldlt().solve(Atb);

        // Apply delta to get updated pose
        pose = se3Plus(pose, delta);

        // Renormalize quaternion
        Quaternion q_new(pose[3], pose[0], pose[1], pose[2]);
        q_new.normalize();
        pose.head<4>() << q_new.x(), q_new.y(), q_new.z(), q_new.w();

        // Check convergence
        if (delta.norm() < params.stepTol)
        {
            result.converged = true;
            break;
        }
    }

    result.pose = pose;
    return result;
}

/**
 * Convert Pose7 to Isometry3d for correspondence computation.
 */
inline Eigen::Isometry3d pose7ToIsometry(const Pose7& pose)
{
    Quaternion q(pose[3], pose[0], pose[1], pose[2]);
    q.normalize();
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.linear() = q.toRotationMatrix();
    iso.translation() = pose.tail<3>();
    return iso;
}

/**
 * Full ICP solver with outer loop (recompute correspondences) and inner loop.
 *
 * @param source        Source grid
 * @param target        Target grid
 * @param initialPose   Initial pose estimate
 * @param rayDir        Ray direction in local frame (typically [0,0,-1])
 * @param weighting     Geometry weighting parameters
 * @param innerParams   Inner solver parameters
 * @param outerParams   Outer loop parameters
 * @return              ICP result with final pose and statistics
 */
template<typename JacobianPolicy = RayJacobianSimplified>
ICPResult solveICP(
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

        // Inner loop with fixed correspondences
        auto innerResult = solveInner<JacobianPolicy>(
            corrs.forward, corrs.reverse, pose, rayDir, weighting, innerParams);

        pose = innerResult.pose;
        result.rms = innerResult.rms;
        result.total_inner_iterations += innerResult.iterations;

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

    result.pose = pose;
    return result;
}

} // namespace ICP
