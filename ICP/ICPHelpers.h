#pragma once
/**
 * Helper functions for ICP solvers.
 *
 * Shared utilities used by both Gauss-Newton and Levenberg-Marquardt solvers.
 * Includes normal equation building, cost evaluation, and convergence checks.
 */

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

/// Result from inner solver (after iterating to convergence)
struct InnerSolveResult
{
    Pose7 pose;       // Final pose after inner iterations
    double rms;       // Final RMS
    int iterations;   // Number of inner iterations performed
    int valid_count;  // Number of valid correspondences
    bool converged;   // Whether step tolerance was reached
};

/**
 * Compute cost at a given pose.
 *
 * @param fwdCorrs   Forward correspondences
 * @param revCorrs   Reverse correspondences
 * @param pose       7D pose to evaluate
 * @param rayDir     Ray direction
 * @param weighting  Geometry weighting parameters
 * @return           Sum of squared residuals
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
 * Check if delta is below convergence thresholds.
 *
 * Uses separate thresholds for translation and rotation components.
 * delta = [translation (3), rotation (3)] in tangent space.
 *
 * @param delta             Tangent space update
 * @param transThreshold    Translation threshold (same units as data, e.g., mm)
 * @param rotThreshold      Rotation threshold (radians)
 * @return                  True if both components are below threshold
 */
inline bool isDeltaConverged(const Tangent6& delta, double transThreshold, double rotThreshold)
{
    double transNorm = delta.head<3>().norm();
    double rotNorm = delta.tail<3>().norm();
    return transNorm < transThreshold && rotNorm < rotThreshold;
}

/**
 * Apply SE(3) update and normalize quaternion.
 *
 * @param pose   Current pose (modified in place)
 * @param delta  Tangent space update
 */
inline void applyDeltaAndNormalize(Pose7& pose, const Tangent6& delta)
{
    pose = se3Plus(pose, delta);
    Quaternion q(pose[3], pose[0], pose[1], pose[2]);
    q.normalize();
    pose.head<4>() << q.x(), q.y(), q.z(), q.w();
}

/**
 * Backtracking line search to find optimal step size.
 *
 * Finds alpha that reduces cost, using simple backtracking.
 * Starts with alpha=1 and reduces by factor beta until cost decreases.
 *
 * @param fwdCorrs     Forward correspondences
 * @param revCorrs     Reverse correspondences
 * @param pose         Current pose
 * @param delta        Full step direction
 * @param currentCost  Cost at current pose
 * @param rayDir       Ray direction
 * @param weighting    Geometry weighting
 * @param params       Line search parameters
 * @return             Optimal step size alpha
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
 * Build normal equations and compute cost in one pass.
 *
 * Accumulates H = J^T J, b = J^T r, and sum of squared residuals.
 *
 * @param fwdCorrs   Forward correspondences
 * @param revCorrs   Reverse correspondences
 * @param pose       Current pose
 * @param rayDir     Ray direction
 * @param weighting  Geometry weighting
 * @param H          Output: Hessian approximation (J^T J)
 * @param b          Output: gradient (J^T r)
 * @param fwd_valid  Output: number of valid forward correspondences
 * @param rev_valid  Output: number of valid reverse correspondences
 * @return           Sum of squared residuals (cost)
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
