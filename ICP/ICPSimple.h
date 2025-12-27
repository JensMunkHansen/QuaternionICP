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
#include <iomanip>

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
    const ICPLineSearchParams& params)
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

        // Accept step if gain ratio exceeds threshold
        bool step_accepted = (rho > params.lm.minRelativeDecrease);

        // Also accept if we're at a minimum (tiny changes)
        bool at_minimum = (std::abs(actual_reduction) < 1e-12 * current_cost) && (delta.norm() < 1e-6);

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

        // Accept step if cost decreases OR if we're at convergence (cost unchanged, small step)
        double cost_reduction = current_cost - trial_cost;
        bool cost_decreased = trial_cost < current_cost;
        bool at_minimum = (std::abs(cost_reduction) < 1e-12 * current_cost) && (delta.norm() < 1e-6);

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
