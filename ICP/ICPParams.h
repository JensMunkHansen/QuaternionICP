// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * @file ICPParams.h
 * @brief ICP algorithm parameters and configuration structures.
 * @author Jens Munk Hansen
 * @date 2024-2025
 *
 * @details This header defines a three-level parameter hierarchy for
 * configuring the Iterative Closest Point (ICP) algorithm:
 *
 * - **SessionParams**: Session-level configuration including backend selection
 *   (hand-rolled vs Ceres), initial pose handling, and CUDA options.
 *
 * - **OuterParams**: Outer loop parameters controlling correspondence updates,
 *   subsampling rates, ray direction, and geometry-based weighting.
 *
 * - **InnerParams**: Inner loop solver parameters including solver type
 *   (Gauss-Newton or Levenberg-Marquardt), iteration limits, convergence
 *   thresholds, and Jacobian computation policy.
 *
 * The Defaults namespace provides pre-configured parameter sets for common
 * use cases: SinglePose, TwoPose, MultiView, and MultiViewCuda.
 */

#pragma once

// Standard C++ headers
#include <cmath>

// Ceres headers
#include <ceres/solver.h>

// Internal headers
#include <ICP/EigenTypes.h>

namespace ICP
{

// ============================================================================
// Enums
// ============================================================================

/**
 * @brief Solver backend selection.
 *
 * Determines which optimization backend is used for the ICP inner loop.
 */
enum class SolverBackend
{
    HandRolled,  ///< Hand-rolled implementation (Gauss-Newton or Levenberg-Marquardt)
    Ceres        ///< Ceres Solver-based implementation with automatic differentiation
};

/**
 * @brief Inner solver type selection.
 *
 * Selects between Gauss-Newton (no damping) and Levenberg-Marquardt
 * (adaptive damping) for the nonlinear least-squares optimization.
 */
enum class SolverType
{
    GaussNewton,        ///< Gauss-Newton solver (faster but may diverge)
    LevenbergMarquardt  ///< Levenberg-Marquardt solver (more robust with damping)
};

/**
 * @brief Jacobian computation policy for ray-projection residuals.
 *
 * Controls whether the Jacobian includes the derivative of the
 * denominator in the ray-plane intersection formula.
 */
enum class JacobianPolicy
{
    Simplified,  ///< Ignore db/dq term (faster, approximate but often sufficient)
    Consistent   ///< Full quotient rule including db/dq (mathematically exact)
};

/**
 * @brief Lambda update strategy for Levenberg-Marquardt.
 *
 * Determines how the damping parameter lambda is adjusted during optimization.
 */
enum class LMStrategy
{
    Simple,     ///< Binary accept/reject with fixed multipliers (lambdaUp/lambdaDown)
    GainRatio   ///< Ceres-style gain ratio with Nielsen update rule
};

/**
 * @brief Linear solver type for Ceres backend.
 *
 * Selects the algorithm used to solve the linear system at each iteration.
 * The optimal choice depends on problem size and available hardware.
 */
enum class LinearSolverType
{
    DenseQR,              ///< Dense QR decomposition (default, good for small problems)
    DenseSchur,           ///< Dense Schur complement (for bundle adjustment style problems)
    SparseSchur,          ///< Sparse Schur complement (for larger sparse problems)
    IterativeSchur,       ///< Iterative Schur (good for large multi-view problems)
    CudaDenseCholesky,    ///< GPU-accelerated dense Cholesky (requires CUDA build)
    CudaSparseCholesky    ///< GPU-accelerated sparse Cholesky (requires CUDA + SuiteSparse)
};

/**
 * @brief Preconditioner type for iterative solvers.
 *
 * Used with IterativeSchur linear solver to improve convergence.
 */
enum class PreconditionerType
{
    Identity,    ///< No preconditioning
    Jacobi,      ///< Jacobi (diagonal) preconditioner
    SchurJacobi  ///< Block Jacobi on the Schur complement
};

/**
 * @brief Incidence weighting mode for grazing angle handling.
 *
 * Controls how the cosine of the incidence angle affects correspondence weights.
 * Larger weights for perpendicular incidence reduce sensitivity to grazing angles.
 */
enum class WeightingMode
{
    Abs,      ///< Linear weighting: weight = |cos(angle)|
    SqrtAbs   ///< Square root weighting: weight = sqrt(|cos(angle)|)
};

// ============================================================================
// Helper structs
// ============================================================================

/**
 * @brief Geometry weighting parameters for incidence-based weighting.
 *
 * Controls how correspondences are weighted based on the angle between
 * the ray direction and surface normal. This helps reduce the influence
 * of grazing-angle measurements which are typically less reliable.
 */
struct GeometryWeighting
{
    bool enable_weight = true;   ///< Enable incidence-angle weighting
    bool enable_gate = true;     ///< Enable gating (reject if |cos| < tau)
    double tau = 0.2;            ///< Cosine threshold for gating and clamping
    WeightingMode mode = WeightingMode::SqrtAbs;  ///< Weighting function type

    /**
     * @brief Compute the weight for a correspondence.
     * @param c Cosine of the incidence angle (dot product of ray and normal)
     * @return Weight in range [0, 1], or 0 if gated out
     */
    double weight(double c) const
    {
        double ac = std::abs(c);
        if (enable_gate && ac < tau)
            return 0.0;
        if (!enable_weight)
            return 1.0;
        ac = std::max(tau, std::min(1.0, ac));
        return (mode == WeightingMode::SqrtAbs) ? std::sqrt(ac) : ac;
    }
};

/**
 * @brief Line search parameters for step size selection.
 *
 * Configures backtracking line search to find an optimal step size
 * that sufficiently decreases the cost function.
 */
struct LineSearchParams
{
    bool enabled = false;     ///< Enable line search (if false, uses full step)
    int maxIterations = 10;   ///< Maximum backtracking iterations
    double alpha = 1.0;       ///< Initial step size
    double beta = 0.5;        ///< Step size reduction factor per iteration
};

/**
 * @brief Levenberg-Marquardt damping parameters.
 *
 * Controls the damping strategy for LM optimization. The damping parameter
 * lambda interpolates between Gauss-Newton (lambda=0) and gradient descent
 * (large lambda).
 */
struct LMParams
{
    LMStrategy strategy = LMStrategy::Simple;  ///< Lambda update strategy
    double lambda = 1e-3;          ///< Initial damping parameter
    bool fixedLambda = true;       ///< If true, lambda stays constant
    double lambdaUp = 10.0;        ///< Multiplier when step is rejected
    double lambdaDown = 0.1;       ///< Multiplier when step is accepted
    double lambdaMin = 1e-10;      ///< Minimum allowed lambda value
    double lambdaMax = 1e10;       ///< Maximum allowed lambda value
    double minRelativeDecrease = 1e-3;  ///< Minimum gain ratio for GainRatio strategy
};

// ============================================================================
// Main parameter structs
// ============================================================================

/**
 * @brief Session-level parameters.
 *
 * Controls global settings that apply across all ICP iterations,
 * including backend selection, initial pose handling, and GPU options.
 */
struct SessionParams
{
    SolverBackend backend = SolverBackend::HandRolled;  ///< Solver backend to use
    bool useGridPoses = true;   ///< Initialize from grid poses (vs identity)
    bool fixFirstPose = true;   ///< Keep first grid fixed (gauge freedom)
    bool verbose = false;       ///< Enable verbose output

    /// @name CUDA Options
    /// @brief Only used when backend == Ceres
    /// @{
    bool useCuda = false;       ///< Enable CUDA acceleration for dense linear algebra
    int cudaDeviceId = 0;       ///< GPU device to use (for multi-GPU systems)
    /// @}
};

/**
 * @brief Outer loop parameters for correspondence updates.
 *
 * Controls how correspondences are computed and filtered between
 * ICP iterations. The outer loop re-establishes correspondences
 * after each pose update.
 */
struct OuterParams
{
    /// @name Iteration Control
    /// @{
    int maxIterations = 10;      ///< Maximum outer loop iterations
    double convergenceTol = 1e-9; ///< Cost change threshold for convergence
    /// @}

    /// @name Correspondence Computation
    /// @{
    Vector3 rayDir{0.0, 0.0, -1.0};  ///< Ray direction for projection (typically -Z)
    float maxDist = 100.0f;          ///< Maximum ray distance for intersection
    int subsampleX = 4;              ///< Subsampling factor in X direction
    int subsampleY = 4;              ///< Subsampling factor in Y direction
    /// @}

    /// @name Geometry Weighting
    /// @{
    GeometryWeighting weighting;     ///< Incidence-angle weighting parameters
    /// @}

    /// @name Multi-view Correspondence Limits
    /// @{
    int minMatch = 50;               ///< Minimum correspondences to consider an edge
    int maxCorrespondences = 0;      ///< Max correspondences per edge (0 = unlimited)
    int maxNeighbors = 0;            ///< Max neighboring grids to match (0 = unlimited)
    /// @}

    bool verbose = false;            ///< Enable verbose output
};

/**
 * @brief Inner loop parameters for solver iterations.
 *
 * Controls the nonlinear least-squares solver that optimizes poses
 * given fixed correspondences. The inner loop runs until convergence
 * or maximum iterations before correspondences are updated.
 */
struct InnerParams
{
    /// @name Solver Selection
    /// @{
    SolverType solverType = SolverType::LevenbergMarquardt;  ///< GN or LM
    /// @}

    /// @name Iteration Control
    /// @{
    int maxIterations = 12;           ///< Maximum inner loop iterations
    double translationThreshold = 1e-4; ///< Translation step convergence threshold
    double rotationThreshold = 1e-4;    ///< Rotation step convergence threshold (radians)
    /// @}

    /// @name Gauss-Newton Options
    /// @{
    double damping = 0.0;             ///< Fixed damping for GN (ignored for LM)
    /// @}

    /// @name Levenberg-Marquardt Options
    /// @{
    LMParams lm;                      ///< LM-specific parameters
    /// @}

    /// @name Line Search
    /// @{
    LineSearchParams lineSearch;      ///< Line search parameters
    /// @}

    /// @name Jacobian Options
    /// @{
    JacobianPolicy jacobianPolicy = JacobianPolicy::Simplified;  ///< Jacobian computation policy
    /// @}

    /// @name Ceres Backend Options
    /// @brief Only used when SessionParams::backend == Ceres
    /// @{
    LinearSolverType linearSolverType = LinearSolverType::DenseQR;  ///< Linear solver type
    PreconditionerType preconditionerType = PreconditionerType::SchurJacobi;  ///< Preconditioner
    /// @}

    bool verbose = false;             ///< Enable verbose output

    /// @name Factory Methods
    /// @{

    /**
     * @brief Create Gauss-Newton solver parameters.
     * @param maxIterations Maximum iterations
     * @param translationThreshold Translation convergence threshold
     * @param rotationThreshold Rotation convergence threshold (radians)
     * @param damping Fixed damping value (0 = pure GN)
     * @return Configured InnerParams for Gauss-Newton
     */
    static InnerParams gaussNewton(
        int maxIterations = 12,
        double translationThreshold = 1e-4,
        double rotationThreshold = 1e-4,
        double damping = 0.0)
    {
        InnerParams p;
        p.solverType = SolverType::GaussNewton;
        p.maxIterations = maxIterations;
        p.translationThreshold = translationThreshold;
        p.rotationThreshold = rotationThreshold;
        p.damping = damping;
        return p;
    }

    /**
     * @brief Create Levenberg-Marquardt solver parameters.
     * @param maxIterations Maximum iterations
     * @param translationThreshold Translation convergence threshold
     * @param rotationThreshold Rotation convergence threshold (radians)
     * @param lambda Initial damping parameter
     * @param adaptiveLambda If true, lambda adapts; if false, stays fixed
     * @return Configured InnerParams for Levenberg-Marquardt
     */
    static InnerParams levenbergMarquardt(
        int maxIterations = 12,
        double translationThreshold = 1e-4,
        double rotationThreshold = 1e-4,
        double lambda = 1e-3,
        bool adaptiveLambda = false)
    {
        InnerParams p;
        p.solverType = SolverType::LevenbergMarquardt;
        p.maxIterations = maxIterations;
        p.translationThreshold = translationThreshold;
        p.rotationThreshold = rotationThreshold;
        p.lm.lambda = lambda;
        p.lm.fixedLambda = !adaptiveLambda;
        return p;
    }
    /// @}
};

// ============================================================================
// Ceres conversion (internal use)
// ============================================================================

/**
 * @brief Convert InnerParams to Ceres solver options.
 *
 * Maps the ICP parameter structure to Ceres-specific options.
 * Used internally by the Ceres backend.
 *
 * @param inner Inner loop parameters to convert
 * @return Configured Ceres solver options
 */
inline ceres::Solver::Options toCeresSolverOptions(const InnerParams& inner)
{
    ceres::Solver::Options options;

    options.max_num_iterations = inner.maxIterations;
    options.function_tolerance = inner.translationThreshold;
    options.gradient_tolerance = inner.translationThreshold;
    options.parameter_tolerance = inner.translationThreshold;

    // Solver type
    if (inner.solverType == SolverType::LevenbergMarquardt)
    {
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.initial_trust_region_radius = (inner.lm.lambda > 0) ? 1.0 / inner.lm.lambda : 1e4;
        options.max_trust_region_radius = (inner.lm.lambdaMin > 0) ? 1.0 / inner.lm.lambdaMin : 1e8;
    }
    else
    {
        options.minimizer_type = ceres::TRUST_REGION;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.initial_trust_region_radius = 1e16;  // Very large = effectively GN
        options.max_trust_region_radius = 1e32;
    }

    // Linear solver
    switch (inner.linearSolverType)
    {
        case LinearSolverType::DenseQR:
            options.linear_solver_type = ceres::DENSE_QR;
            break;
        case LinearSolverType::DenseSchur:
            options.linear_solver_type = ceres::DENSE_SCHUR;
            break;
        case LinearSolverType::SparseSchur:
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            break;
        case LinearSolverType::IterativeSchur:
            options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            break;
        case LinearSolverType::CudaDenseCholesky:
            options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
            options.dense_linear_algebra_library_type = ceres::CUDA;
            break;
        case LinearSolverType::CudaSparseCholesky:
            // GPU-accelerated sparse via SuiteSparse CHOLMOD (requires CHOLMOD built with CUDA)
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
            break;
    }

    // Preconditioner
    switch (inner.preconditionerType)
    {
        case PreconditionerType::Identity:
            options.preconditioner_type = ceres::IDENTITY;
            break;
        case PreconditionerType::Jacobi:
            options.preconditioner_type = ceres::JACOBI;
            break;
        case PreconditionerType::SchurJacobi:
            options.preconditioner_type = ceres::SCHUR_JACOBI;
            break;
    }

    options.logging_type = inner.verbose ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
    options.minimizer_progress_to_stdout = inner.verbose;

    return options;
}

// ============================================================================
// Default Parameter Presets
// ============================================================================

/**
 * @brief Default parameter presets for common ICP configurations.
 *
 * Provides factory functions returning pre-configured parameter sets
 * optimized for different use cases. These serve as sensible starting
 * points that can be further customized.
 */
namespace Defaults
{

/**
 * @brief Single-pose ICP defaults.
 *
 * Configuration for aligning one moving grid to a fixed target grid.
 * Suitable for pairwise registration problems.
 */
namespace SinglePose
{
    inline InnerParams inner()
    {
        InnerParams p;
        p.solverType = SolverType::LevenbergMarquardt;
        p.maxIterations = 5;
        p.translationThreshold = 1e-4;
        p.rotationThreshold = 1e-4;
        p.damping = 0.0;

        p.lm.lambda = 1e-3;
        p.lm.fixedLambda = true;
        p.lm.lambdaUp = 10.0;
        p.lm.lambdaDown = 0.1;
        p.lm.lambdaMin = 1e-10;
        p.lm.lambdaMax = 1e10;

        p.lineSearch.enabled = false;
        p.lineSearch.maxIterations = 10;
        p.lineSearch.alpha = 1.0;
        p.lineSearch.beta = 0.5;

        p.jacobianPolicy = JacobianPolicy::Simplified;
        p.linearSolverType = LinearSolverType::DenseQR;
        p.verbose = false;
        return p;
    }

    inline OuterParams outer()
    {
        OuterParams p;
        p.maxIterations = 10;
        p.convergenceTol = 1e-9;
        p.rayDir = Vector3(0.0, 0.0, -1.0);
        p.maxDist = 100.0f;
        p.subsampleX = 4;
        p.subsampleY = 4;

        p.weighting.enable_weight = true;
        p.weighting.enable_gate = true;
        p.weighting.tau = 0.2;

        p.minMatch = 50;
        p.maxCorrespondences = 0;  // Unlimited for single-pose
        p.maxNeighbors = 0;
        p.verbose = false;
        return p;
    }

    inline SessionParams session()
    {
        SessionParams p;
        p.backend = SolverBackend::HandRolled;
        p.useGridPoses = false;  // Synthetic grids typically don't have poses
        p.fixFirstPose = true;
        p.verbose = false;
        return p;
    }
}

/**
 * @brief Two-pose ICP defaults.
 *
 * Configuration for simultaneously optimizing two grid poses.
 * Both grids are allowed to move (unless fixFirstPose is set).
 */
namespace TwoPose
{
    inline InnerParams inner()
    {
        InnerParams p;
        p.solverType = SolverType::LevenbergMarquardt;
        p.maxIterations = 12;  // More iterations for two-pose
        p.translationThreshold = 1e-4;
        p.rotationThreshold = 1e-4;
        p.damping = 0.0;

        p.lm.lambda = 1e-3;
        p.lm.fixedLambda = true;
        p.lm.lambdaUp = 10.0;
        p.lm.lambdaDown = 0.1;
        p.lm.lambdaMin = 1e-10;
        p.lm.lambdaMax = 1e10;

        p.lineSearch.enabled = false;
        p.lineSearch.maxIterations = 10;
        p.lineSearch.alpha = 1.0;
        p.lineSearch.beta = 0.5;

        p.jacobianPolicy = JacobianPolicy::Simplified;
        p.linearSolverType = LinearSolverType::DenseQR;
        p.verbose = false;
        return p;
    }

    inline OuterParams outer()
    {
        OuterParams p;
        p.maxIterations = 10;
        p.convergenceTol = 1e-9;
        p.rayDir = Vector3(0.0, 0.0, -1.0);
        p.maxDist = 100.0f;
        p.subsampleX = 4;
        p.subsampleY = 4;

        p.weighting.enable_weight = true;
        p.weighting.enable_gate = true;
        p.weighting.tau = 0.2;

        p.minMatch = 50;
        p.maxCorrespondences = 0;  // Unlimited for two-pose
        p.maxNeighbors = 0;
        p.verbose = false;
        return p;
    }
}

/**
 * @brief Multi-view ICP defaults.
 *
 * Configuration for jointly optimizing many grid poses.
 * Uses Ceres backend with iterative Schur complement solver
 * for efficient handling of the large sparse problem.
 */
namespace MultiView
{
    inline InnerParams inner()
    {
        InnerParams p;
        p.solverType = SolverType::LevenbergMarquardt;
        p.maxIterations = 20;  // More iterations for larger problem
        p.translationThreshold = 1e-4;
        p.rotationThreshold = 1e-4;
        p.damping = 0.0;

        p.lm.lambda = 1e-3;
        p.lm.fixedLambda = true;
        p.lm.lambdaUp = 10.0;
        p.lm.lambdaDown = 0.1;
        p.lm.lambdaMin = 1e-10;
        p.lm.lambdaMax = 1e10;

        p.lineSearch.enabled = false;
        p.lineSearch.maxIterations = 10;
        p.lineSearch.alpha = 1.0;
        p.lineSearch.beta = 0.5;

        p.jacobianPolicy = JacobianPolicy::Simplified;
        p.linearSolverType = LinearSolverType::IterativeSchur;  // Better for multi-view
        p.preconditionerType = PreconditionerType::SchurJacobi;
        p.verbose = false;
        return p;
    }

    inline OuterParams outer()
    {
        OuterParams p;
        p.maxIterations = 10;
        p.convergenceTol = 1e-9;
        p.rayDir = Vector3(0.0, 0.0, -1.0);
        p.maxDist = 100.0f;
        p.subsampleX = 4;
        p.subsampleY = 4;

        p.weighting.enable_weight = true;
        p.weighting.enable_gate = true;
        p.weighting.tau = 0.2;

        p.minMatch = 50;
        p.maxCorrespondences = 200;  // Limit memory for multi-view
        p.maxNeighbors = 0;
        p.verbose = false;
        return p;
    }

    inline SessionParams session()
    {
        SessionParams p;
        p.backend = SolverBackend::Ceres;  // Multi-view requires Ceres
        p.useGridPoses = true;
        p.fixFirstPose = true;
        p.verbose = false;
        return p;
    }
}

/**
 * @brief Multi-view ICP with CUDA acceleration.
 *
 * Same as MultiView but uses GPU-accelerated dense Cholesky solver.
 * Requires Ceres built with CUDA support.
 */
namespace MultiViewCuda
{
    inline InnerParams inner()
    {
        InnerParams p = MultiView::inner();
        p.linearSolverType = LinearSolverType::CudaDenseCholesky;
        return p;
    }

    inline OuterParams outer()
    {
        return MultiView::outer();
    }

    inline SessionParams session()
    {
        SessionParams p = MultiView::session();
        p.useCuda = true;
        return p;
    }
}

} // namespace Defaults

} // namespace ICP
