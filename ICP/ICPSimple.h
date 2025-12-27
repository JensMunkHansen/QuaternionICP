#pragma once
/**
 * Simple ICP solver with inner and outer loops.
 *
 * Inner loop: Build and solve normal equations with fixed correspondences.
 * Outer loop: Recompute correspondences, run inner loop, repeat.
 *
 * Uses ForwardRayCost and ReverseRayCost for bidirectional correspondences.
 */

// Standard C++ headers
#include <limits>

// Internal headers
#include <ICP/EigenUtils.h>
#include <ICP/ICPHelpers.h>
#include <ICP/ICPLM.h>

namespace ICP
{

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
 * Gauss-Newton inner solver with optional line search.
 *
 * @param fwdCorrs   Forward correspondences (source→target rays)
 * @param revCorrs   Reverse correspondences (target→source rays)
 * @param initialPose Initial 7D pose [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDir     Ray direction in local frame (same for both, typically [0,0,-1])
 * @param weighting  Geometry weighting parameters
 * @param params     Inner solver parameters (iterations, tolerance, damping, line search)
 * @return           Final pose, RMS, iteration count, convergence status
 */
template<typename JacobianPolicy = RayJacobianSimplified>
InnerSolveResult solveInnerGN(
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

        // Build normal equations and get current cost in one pass
        Matrix6 H;
        Vector6 b;
        int fwd_valid, rev_valid;
        double current_cost = buildNormalEquations<JacobianPolicy>(
            fwdCorrs, revCorrs, pose, rayDir, weighting,
            H, b, fwd_valid, rev_valid);

        int valid_count = fwd_valid + rev_valid;
        result.rms = valid_count > 0 ? std::sqrt(current_cost / valid_count) : 0.0;
        result.valid_count = valid_count;

        if (params.verbose)
        {
            std::cout << "\t\t\tinner " << iter << ": rms=" << std::scientific
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

        // Line search if enabled
        if (params.lineSearch.enabled)
        {
            double alpha = lineSearch<JacobianPolicy>(
                fwdCorrs, revCorrs, pose, delta, current_cost,
                rayDir, weighting, params.lineSearch);

            if (params.verbose && alpha < 1.0)
            {
                std::cout << "\t\t\t  line search: alpha=" << alpha << "\n";
            }

            delta *= alpha;
        }

        // Apply step
        applyDeltaAndNormalize(pose, delta);

        // Check convergence using separate translation/rotation thresholds
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
 * Inner solver: iterate with fixed correspondences until convergence.
 *
 * Dispatches to either Gauss-Newton or Levenberg-Marquardt based on params.solverType.
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
    switch (params.solverType)
    {
        case SolverType::GaussNewton:
            return solveInnerGN<JacobianPolicy>(fwdCorrs, revCorrs, initialPose, rayDir, weighting, params);
        case SolverType::LevenbergMarquardt:
            return solveInnerLM<JacobianPolicy>(fwdCorrs, revCorrs, initialPose, rayDir, weighting, params);
        default:
            return solveInnerGN<JacobianPolicy>(fwdCorrs, revCorrs, initialPose, rayDir, weighting, params);
    }
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

        if (outerParams.verbose)
        {
            std::cout << "\touter " << outer << ":\n";
            std::cout << "\t\tfwd_corrs=" << corrs.forward.size()
                      << ", rev_corrs=" << corrs.reverse.size() << "\n";
        }

        // Inner loop with fixed correspondences
        auto innerResult = solveInner<JacobianPolicy>(
            corrs.forward, corrs.reverse, pose, rayDir, weighting, innerParams);

        pose = innerResult.pose;
        result.rms = innerResult.rms;
        result.total_inner_iterations += innerResult.iterations;

        if (outerParams.verbose)
        {
            std::cout << "\t\touter " << outer << ": rms=" << std::scientific
                      << std::setprecision(6) << result.rms << ", valid=" << innerResult.valid_count
                      << ", inner_iters=" << innerResult.iterations << "\n";
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
        std::cout << "Total inner iterations: " << result.total_inner_iterations << "\n";
    }

    result.pose = pose;
    return result;
}

} // namespace ICP
