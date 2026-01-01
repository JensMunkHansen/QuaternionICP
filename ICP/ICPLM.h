// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once
/**
 * Levenberg-Marquardt solvers for ICP.
 *
 * Provides two LM strategies:
 * - Simple: Binary accept/reject with fixed lambda multipliers
 * - GainRatio: Ceres-style gain ratio with Nielsen radius update
 *
 * @see ICPSolver.h for Gauss-Newton solver and outer loop
 */

// Standard C++ headers
#include <cmath>
#include <iomanip>
#include <iostream>

// Internal headers
#include <ICP/ICPHelpers.h>

namespace ICP
{

// ============================================================================
// LM Helper Functions
// ============================================================================

/**
 * Compute the predicted cost reduction from the quadratic model.
 *
 * For the linearized model: cost(x + delta) ≈ cost(x) + delta^T * b + 0.5 * delta^T * H * delta
 * Model cost change = -delta^T * b - 0.5 * delta^T * H * delta
 *
 * Since we solve H * delta = -b, we have:
 *   model_change = -delta^T * b - 0.5 * delta^T * H * delta
 *                = delta^T * H * delta - 0.5 * delta^T * H * delta
 *                = 0.5 * delta^T * H * delta
 *
 * But actually for LM, we also need to account for the damping contribution.
 * The full model is: model_change = -delta^T * (b + 0.5 * H_damped * delta)
 *                                 = -delta^T * b - 0.5 * delta^T * H_damped * delta
 *
 * Since H_damped * delta = -b (from the solve), we get:
 *   model_change = -delta^T * b + 0.5 * delta^T * b = -0.5 * delta^T * b
 *
 * @param delta  The computed step
 * @param b      The gradient vector (J^T * r)
 * @return       Predicted cost reduction (positive if cost decreases)
 */
inline double computeModelCostChange(const Tangent6& delta, const Vector6& b)
{
    // model_change = -0.5 * delta^T * b
    // This is positive when delta and b point in opposite directions (which they should)
    return -0.5 * delta.dot(b);
}

/**
 * Compute Nielsen's trust region radius update factor.
 *
 * From "Methods for Non-Linear Least Squares Problems" by Madsen, Nielsen, Tingleff.
 * radius_new = radius / max(1/3, 1 - (2*rho - 1)^3)
 *
 * This formula:
 * - When rho ≈ 1 (perfect model): expands radius by factor of 3
 * - When rho ≈ 0.5: keeps radius roughly the same
 * - When rho close to 0: contracts radius
 *
 * @param rho  Gain ratio (actual_reduction / model_reduction)
 * @return     Factor to divide radius by (< 1 expands, > 1 contracts)
 */
inline double nielsenRadiusFactor(double rho)
{
    double tmp = 2.0 * rho - 1.0;
    return std::max(1.0 / 3.0, 1.0 - tmp * tmp * tmp);
}

// ============================================================================
// LM Solvers
// ============================================================================

/**
 * Levenberg-Marquardt inner solver with gain ratio strategy (Ceres-style).
 *
 * Uses the gain ratio rho = actual_reduction / model_reduction to:
 * - Decide step acceptance: accept if rho > minRelativeDecrease
 * - Adapt trust region: radius = radius / nielsenFactor(rho)
 * - Double decrease factor on successive rejections
 *
 * @param fwdCorrs   Forward correspondences (source→target rays)
 * @param revCorrs   Reverse correspondences (target→source rays)
 * @param initialPose Initial 7D pose [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDir     Ray direction in local frame (same for both, typically [0,0,-1])
 * @param weighting  Geometry weighting parameters
 * @param params     Inner solver parameters (iterations, tolerance, LM settings)
 * @return           Final pose, RMS, iteration count, convergence status
 */
template<typename JacobianPolicy = RayJacobianSimplified>
InnerSolveResult solveInnerLMGainRatio(
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

    // Trust region radius = 1 / lambda
    double radius = 1.0 / params.lm.lambda;
    const double maxRadius = 1.0 / params.lm.lambdaMin;
    const double minRadius = 1.0 / params.lm.lambdaMax;
    double decreaseFactor = 2.0;  // Doubles on successive rejections

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

        double lambda = 1.0 / radius;

        if (params.verbose)
        {
            std::cout << "\t\t\tinner " << iter << ": rms=" << std::scientific
                      << std::setprecision(6) << result.rms
                      << ", cost=" << current_cost
                      << ", fwd=" << fwd_valid << ", rev=" << rev_valid
                      << ", lambda=" << lambda << ", radius=" << radius << "\n";
        }

        if (valid_count < 6)
        {
            result.pose = pose;
            return result;
        }

        // Solve with LM damping: (H + lambda*diag(H)) delta = -b
        Matrix6 H_damped = H;
        H_damped.diagonal().array() *= (1.0 + lambda);
        Tangent6 delta = -H_damped.ldlt().solve(b);

        // Compute model cost change (predicted reduction)
        double model_cost_change = computeModelCostChange(delta, b);

        // Trial step
        Pose7 pose_trial = pose;
        applyDeltaAndNormalize(pose_trial, delta);

        // Evaluate actual cost at trial pose
        double trial_cost = computeCostAtPose<JacobianPolicy>(
            fwdCorrs, revCorrs, pose_trial, rayDir, weighting);

        double actual_reduction = current_cost - trial_cost;
        double rho = (model_cost_change > 0) ? actual_reduction / model_cost_change : 0.0;

        if (params.verbose)
        {
            std::cout << "\t\t\t  trial_cost=" << trial_cost
                      << ", rho=" << rho
                      << ", model_change=" << model_cost_change
                      << ", delta_norm=" << delta.norm() << "\n";
        }

        // Check for parameter-tolerance convergence BEFORE step acceptance (like Ceres)
        // This allows termination even if the step would be rejected
        if (isDeltaConverged(delta, params.translationThreshold, params.rotationThreshold))
        {
            result.converged = true;
            result.pose = pose;
            return result;
        }

        // Accept step if gain ratio exceeds threshold
        bool step_accepted = (rho > params.lm.minRelativeDecrease);

        // Also accept if we're at a minimum (tiny changes)
        // Use relative function tolerance like Ceres: |cost_change|/cost < tol
        bool at_minimum = (std::abs(actual_reduction) < params.translationThreshold * current_cost) && (delta.norm() < 1e-6);

        if (step_accepted || at_minimum)
        {
            // Step accepted
            pose = pose_trial;

            // Update radius using Nielsen formula
            double factor = nielsenRadiusFactor(rho);
            radius = std::min(maxRadius, radius / factor);

            // Reset decrease factor
            decreaseFactor = 2.0;

            if (params.verbose && factor != 1.0)
            {
                std::cout << "\t\t\t  accepted: new_radius=" << radius << "\n";
            }

            // Check convergence
            if (isDeltaConverged(delta, params.translationThreshold, params.rotationThreshold) || at_minimum)
            {
                result.converged = true;
                result.pose = pose;
                return result;
            }
        }
        else
        {
            // Step rejected - shrink trust region
            radius = std::max(minRadius, radius / decreaseFactor);
            decreaseFactor *= 2.0;  // Double for next rejection

            if (params.verbose)
            {
                std::cout << "\t\t\t  rejected: new_radius=" << radius
                          << ", decrease_factor=" << decreaseFactor << "\n";
            }

            // Check if radius is too small (converged to local minimum)
            if (radius <= minRadius)
            {
                result.pose = pose;
                return result;
            }
        }
    }

    result.pose = pose;
    return result;
}

/**
 * Levenberg-Marquardt inner solver with simple adaptive damping.
 *
 * When params.lm.fixedLambda=true: uses constant lambda (like damped Gauss-Newton).
 * When params.lm.fixedLambda=false: adapts lambda based on cost reduction.
 *
 * Uses simple cost comparison for step acceptance (like MultiRegistration):
 * - If trialCost < currentCost: accept step, decrease lambda
 * - Otherwise: reject step, increase lambda, retry with same H and b
 *
 * @param fwdCorrs   Forward correspondences (source→target rays)
 * @param revCorrs   Reverse correspondences (target→source rays)
 * @param initialPose Initial 7D pose [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDir     Ray direction in local frame (same for both, typically [0,0,-1])
 * @param weighting  Geometry weighting parameters
 * @param params     Inner solver parameters (iterations, tolerance, LM settings)
 * @return           Final pose, RMS, iteration count, convergence status
 */
template<typename JacobianPolicy = RayJacobianSimplified>
InnerSolveResult solveInnerLMSimple(
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
    double lambda = params.lm.lambda;
    const bool use_adaptive = !params.lm.fixedLambda;

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
                      << ", fwd=" << fwd_valid << ", rev=" << rev_valid
                      << ", lambda=" << lambda << "\n";
        }

        if (valid_count < 6)
        {
            result.pose = pose;
            return result;
        }

        // Solve with LM damping: (H + lambda*diag(H)) delta = -b
        Matrix6 H_damped = H;
        H_damped.diagonal().array() *= (1.0 + lambda);
        Tangent6 delta = -H_damped.ldlt().solve(b);

        // Trial step
        Pose7 pose_trial = pose;
        applyDeltaAndNormalize(pose_trial, delta);

        // Evaluate cost at trial pose (no Jacobians needed)
        double trial_cost = computeCostAtPose<JacobianPolicy>(
            fwdCorrs, revCorrs, pose_trial, rayDir, weighting);

        if (params.verbose)
        {
            std::cout << "\t\t\t  trial_cost=" << trial_cost
                      << ", delta_norm=" << delta.norm() << "\n";
        }

        // Check for parameter-tolerance convergence BEFORE step acceptance (like Ceres)
        // This allows termination even if the step would be rejected
        if (isDeltaConverged(delta, params.translationThreshold, params.rotationThreshold))
        {
            result.converged = true;
            result.pose = pose;
            return result;
        }

        // Accept step if cost decreases OR if we're at convergence (cost unchanged, small step)
        // Use relative function tolerance like Ceres: |cost_change|/cost < tol
        double cost_reduction = current_cost - trial_cost;
        bool cost_decreased = trial_cost < current_cost;
        bool at_minimum = (std::abs(cost_reduction) < params.translationThreshold * current_cost) && (delta.norm() < 1e-6);

        if (cost_decreased || at_minimum)
        {
            // Step accepted - optionally apply line search for better step size
            if (params.lineSearch.enabled && cost_decreased)
            {
                double alpha = lineSearch<JacobianPolicy>(
                    fwdCorrs, revCorrs, pose, delta, current_cost,
                    rayDir, weighting, params.lineSearch);

                if (params.verbose && alpha < 1.0)
                {
                    std::cout << "\t\t\t  line search: alpha=" << alpha << "\n";
                }

                // Apply scaled step
                Tangent6 scaledDelta = alpha * delta;
                applyDeltaAndNormalize(pose, scaledDelta);
                delta = scaledDelta;  // Update delta for convergence check
            }
            else
            {
                pose = pose_trial;
            }

            if (use_adaptive)
            {
                // Decrease lambda (trust model more)
                lambda = std::max(params.lm.lambdaMin, lambda * params.lm.lambdaDown);
            }

            // Check convergence using separate translation/rotation thresholds
            if (isDeltaConverged(delta, params.translationThreshold, params.rotationThreshold) || at_minimum)
            {
                result.converged = true;
                result.pose = pose;
                return result;
            }
        }
        else
        {
            // Reject step
            if (use_adaptive)
            {
                // Increase lambda (trust model less)
                lambda = std::min(params.lm.lambdaMax, lambda * params.lm.lambdaUp);

                // Retry with same H and b but new lambda (up to one retry)
                H_damped = H;
                H_damped.diagonal().array() += lambda;
                delta = -H_damped.ldlt().solve(b);

                pose_trial = pose;
                applyDeltaAndNormalize(pose_trial, delta);

                trial_cost = computeCostAtPose<JacobianPolicy>(
                    fwdCorrs, revCorrs, pose_trial, rayDir, weighting);

                if (params.verbose)
                {
                    std::cout << "\t\t\t  retry: lambda=" << lambda
                              << ", trial_cost=" << trial_cost << "\n";
                }

                if (trial_cost < current_cost)
                {
                    // Retry succeeded - optionally apply line search
                    if (params.lineSearch.enabled)
                    {
                        double alpha = lineSearch<JacobianPolicy>(
                            fwdCorrs, revCorrs, pose, delta, current_cost,
                            rayDir, weighting, params.lineSearch);

                        if (params.verbose && alpha < 1.0)
                        {
                            std::cout << "\t\t\t  line search (retry): alpha=" << alpha << "\n";
                        }

                        Tangent6 scaledDelta = alpha * delta;
                        applyDeltaAndNormalize(pose, scaledDelta);
                        delta = scaledDelta;
                    }
                    else
                    {
                        pose = pose_trial;
                    }

                    if (isDeltaConverged(delta, params.translationThreshold, params.rotationThreshold))
                    {
                        result.converged = true;
                        result.pose = pose;
                        return result;
                    }
                }
                // If still rejected, continue to next iteration (lambda was already increased)
            }
            // Fixed lambda mode: just continue to next iteration
        }
    }

    result.pose = pose;
    return result;
}

/**
 * Levenberg-Marquardt inner solver dispatcher.
 *
 * Dispatches to either Simple or GainRatio strategy based on params.lm.strategy.
 *
 * @param fwdCorrs   Forward correspondences (source→target rays)
 * @param revCorrs   Reverse correspondences (target→source rays)
 * @param initialPose Initial 7D pose [qx, qy, qz, qw, tx, ty, tz]
 * @param rayDir     Ray direction in local frame (same for both, typically [0,0,-1])
 * @param weighting  Geometry weighting parameters
 * @param params     Inner solver parameters (iterations, tolerance, LM settings)
 * @return           Final pose, RMS, iteration count, convergence status
 */
template<typename JacobianPolicy = RayJacobianSimplified>
InnerSolveResult solveInnerLM(
    const std::vector<Correspondence>& fwdCorrs,
    const std::vector<Correspondence>& revCorrs,
    const Pose7& initialPose,
    const Vector3& rayDir,
    const GeometryWeighting& weighting = GeometryWeighting(),
    const InnerParams& params = InnerParams())
{
    switch (params.lm.strategy)
    {
        case LMStrategy::GainRatio:
            return solveInnerLMGainRatio<JacobianPolicy>(fwdCorrs, revCorrs, initialPose, rayDir, weighting, params);
        case LMStrategy::Simple:
        default:
            return solveInnerLMSimple<JacobianPolicy>(fwdCorrs, revCorrs, initialPose, rayDir, weighting, params);
    }
}

} // namespace ICP
