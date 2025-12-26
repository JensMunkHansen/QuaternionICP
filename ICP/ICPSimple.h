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
 * Gauss-Newton inner solver with fixed damping.
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

        Matrix6 AtA = Matrix6::Zero();
        Vector6 Atb = Vector6::Zero();
        double sum_sq = 0.0;
        int fwd_valid = 0;
        int rev_valid = 0;

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

            AtA += J6.transpose() * J6;
            Atb += -J6.transpose() * residual;
            sum_sq += residual * residual;
            rev_valid++;
        }

        int valid_count = fwd_valid + rev_valid;
        result.rms = valid_count > 0 ? std::sqrt(sum_sq / valid_count) : 0.0;
        result.valid_count = valid_count;

        if (params.verbose)
        {
            std::cout << "\t\t\tinner " << iter << ": rms=" << std::scientific
                      << std::setprecision(6) << result.rms
                      << ", fwd_valid=" << fwd_valid << ", rev_valid=" << rev_valid << "\n";
        }

        if (valid_count < 6)
        {
            result.pose = pose;
            return result;
        }

        // Solve with damping: (J^T J + lambda*I) delta = -J^T r
        AtA.diagonal().array() += lambda;
        Tangent6 delta = AtA.ldlt().solve(Atb);

        // Apply step
        pose = se3Plus(pose, delta);
        Quaternion q(pose[3], pose[0], pose[1], pose[2]);
        q.normalize();
        pose.head<4>() << q.x(), q.y(), q.z(), q.w();

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
 * Levenberg-Marquardt inner solver with fixed or adaptive damping.
 *
 * When params.lm.fixedLambda=true: uses constant lambda (like damped Gauss-Newton).
 * When params.lm.fixedLambda=false: uses gain ratio to adapt lambda.
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
    InnerSolveResult result;
    result.pose = initialPose;
    result.iterations = 0;
    result.converged = false;

    Pose7 pose = initialPose;
    double lambda = params.lm.lambda;
    const bool use_adaptive = !params.lm.fixedLambda;
    double current_cost = std::numeric_limits<double>::max();

    for (int iter = 0; iter < params.maxIterations; ++iter)
    {
        result.iterations++;

        Matrix6 AtA = Matrix6::Zero();
        Vector6 Atb = Vector6::Zero();
        double sum_sq = 0.0;
        int fwd_valid = 0;
        int rev_valid = 0;

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

            AtA += J6.transpose() * J6;
            Atb += -J6.transpose() * residual;
            sum_sq += residual * residual;
            rev_valid++;
        }

        int valid_count = fwd_valid + rev_valid;
        result.rms = valid_count > 0 ? std::sqrt(sum_sq / valid_count) : 0.0;
        result.valid_count = valid_count;
        current_cost = sum_sq;

        if (params.verbose)
        {
            std::cout << "\t\t\tinner " << iter << ": rms=" << std::scientific
                      << std::setprecision(6) << result.rms
                      << ", fwd_valid=" << fwd_valid << ", rev_valid=" << rev_valid
                      << ", lambda=" << lambda << "\n";
        }

        if (valid_count < 6)
        {
            result.pose = pose;
            return result;
        }

        // Solve with damping: (J^T J + lambda*I) delta = -J^T r
        AtA.diagonal().array() += lambda;
        Tangent6 delta = AtA.ldlt().solve(Atb);

        // Trial step
        Pose7 pose_trial = se3Plus(pose, delta);
        Quaternion q_trial(pose_trial[3], pose_trial[0], pose_trial[1], pose_trial[2]);
        q_trial.normalize();
        pose_trial.head<4>() << q_trial.x(), q_trial.y(), q_trial.z(), q_trial.w();

        bool accept_step = true;

        // Adaptive LM: evaluate gain ratio and adjust lambda
        if (use_adaptive)
        {
            // Predicted reduction: pred = delta^T * J^T r = delta^T * Atb
            double predicted_reduction = delta.dot(Atb);

            // Check for convergence based on very small predicted reduction
            // (means we're at a local minimum where linear model predicts no improvement)
            if (predicted_reduction < 1e-12)
            {
                if (params.verbose)
                {
                    std::cout << "\t\t\t  Converged: predicted reduction too small ("
                              << predicted_reduction << ")\n";
                }
                result.converged = true;
                result.pose = pose;
                return result;
            }

            // Compute actual cost at trial pose
            double trial_cost = 0.0;
            for (const auto& corr : fwdCorrs)
            {
                Vector3 pS = corr.srcPoint.cast<double>();
                Vector3 qT = corr.tgtPoint.cast<double>();
                Vector3 nT = corr.tgtNormal.cast<double>();
                ForwardRayCost<JacobianPolicy> cost(pS, qT, nT, rayDir, weighting);
                double const* pparams[1] = {pose_trial.data()};
                double residual;
                cost(pparams, &residual, nullptr);
                if (std::abs(residual) < 1e10) trial_cost += residual * residual;
            }

            for (const auto& corr : revCorrs)
            {
                Vector3 pT = corr.srcPoint.cast<double>();
                Vector3 qS = corr.tgtPoint.cast<double>();
                Vector3 nS = corr.tgtNormal.cast<double>();
                ReverseRayCost<JacobianPolicy> cost(pT, qS, nS, rayDir, weighting);
                double const* pparams[1] = {pose_trial.data()};
                double residual;
                cost(pparams, &residual, nullptr);
                if (std::abs(residual) < 1e10) trial_cost += residual * residual;
            }

            double actual_reduction = current_cost - trial_cost;
            double rho = predicted_reduction > 0 ? actual_reduction / predicted_reduction : -1.0;

            if (params.verbose)
            {
                std::cout << "\t\t\t  rho=" << rho << ", pred=" << predicted_reduction
                          << ", actual=" << actual_reduction << "\n";
            }

            // Adjust lambda based on gain ratio
            if (rho > 0.75)
            {
                // Good step: decrease lambda (trust model more)
                lambda = std::max(params.lm.lambdaMin, lambda * params.lm.lambdaDown);
            }
            else if (rho < 0.25)
            {
                // Bad step: increase lambda (trust model less)
                double new_lambda = std::min(params.lm.lambdaMax, lambda * params.lm.lambdaUp);

                // Escape clause: if lambda already at max and step is still bad, accept anyway
                // (we can't get more conservative, so we're stuck at a local minimum)
                if (lambda >= params.lm.lambdaMax && new_lambda >= params.lm.lambdaMax)
                {
                    if (params.verbose)
                    {
                        std::cout << "\t\t\t  Lambda at max (" << params.lm.lambdaMax
                                  << "), accepting step despite rho=" << rho << "\n";
                    }
                    accept_step = true;
                }
                else
                {
                    lambda = new_lambda;
                    accept_step = false;
                    result.iterations--;  // Don't count rejected iterations
                }
            }
            // else: 0.25 <= rho <= 0.75, accept step but keep lambda unchanged
        }

        if (accept_step)
        {
            pose = pose_trial;

            // Check convergence
            if (delta.norm() < params.stepTol)
            {
                result.converged = true;
                break;
            }
        }
        else
        {
            // Step rejected, retry with increased lambda (don't advance iteration)
            iter--;
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
