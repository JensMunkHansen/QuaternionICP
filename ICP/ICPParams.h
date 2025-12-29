#pragma once
/**
 * ICP parameters and configuration structures.
 *
 * Three-level hierarchy:
 *   ICPSessionParams - session-level (backend, grid poses, first pose fixed)
 *   ICPOuterParams   - outer loop (correspondences, subsampling, weighting)
 *   ICPInnerParams   - inner loop (solver type, iterations, thresholds)
 */

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
 * Solver backend selection.
 */
enum class SolverBackend
{
    HandRolled,  // Hand-rolled implementation (GN or LM)
    Ceres        // Ceres-based implementation
};

/**
 * Inner solver type (GN vs LM).
 */
enum class SolverType
{
    GaussNewton,
    LevenbergMarquardt
};

/**
 * Jacobian computation policy for ray-projection residuals.
 */
enum class JacobianPolicy
{
    Simplified,  // Ignore db/dq term (faster, approximate)
    Consistent   // Full quotient rule including db/dq (exact)
};

/**
 * Lambda update strategy for Levenberg-Marquardt.
 */
enum class LMStrategy
{
    Simple,     // Binary accept/reject with fixed multipliers
    GainRatio   // Ceres-style gain ratio with Nielsen update
};

/**
 * Linear solver type for Ceres backend.
 */
enum class LinearSolverType
{
    DenseQR,         // Default, good for small problems
    DenseSchur,      // For bundle adjustment style problems
    SparseSchur,     // Sparse version of Schur
    IterativeSchur   // Iterative, good for large multi-view problems
};

/**
 * Preconditioner type for iterative solvers.
 */
enum class PreconditionerType
{
    Identity,
    Jacobi,
    SchurJacobi
};

/**
 * Incidence weighting mode for grazing angle handling.
 */
enum class WeightingMode
{
    Abs,      // weight = |c|
    SqrtAbs   // weight = sqrt(|c|)
};

// ============================================================================
// Helper structs
// ============================================================================

/**
 * Geometry weighting parameters for incidence-based weighting.
 */
struct GeometryWeighting
{
    bool enable_weight = true;
    bool enable_gate = true;
    double tau = 0.2;
    WeightingMode mode = WeightingMode::SqrtAbs;

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
 * Line search parameters.
 */
struct LineSearchParams
{
    bool enabled = false;
    int maxIterations = 10;
    double alpha = 1.0;
    double beta = 0.5;
};

/**
 * Levenberg-Marquardt parameters.
 */
struct LMParams
{
    LMStrategy strategy = LMStrategy::Simple;
    double lambda = 1e-3;
    bool fixedLambda = true;
    double lambdaUp = 10.0;
    double lambdaDown = 0.1;
    double lambdaMin = 1e-10;
    double lambdaMax = 1e10;
    double minRelativeDecrease = 1e-3;  // For GainRatio strategy
};

// ============================================================================
// Main parameter structs
// ============================================================================

/**
 * Session-level parameters.
 *
 * Controls backend selection and initial pose handling.
 */
struct SessionParams
{
    SolverBackend backend = SolverBackend::HandRolled;
    bool useGridPoses = true;
    bool fixFirstPose = true;
    bool verbose = false;
};

/**
 * Outer loop parameters (correspondence updates).
 */
struct OuterParams
{
    // Iteration control
    int maxIterations = 10;
    double convergenceTol = 1e-9;

    // Correspondence computation
    Vector3 rayDir{0.0, 0.0, -1.0};
    float maxDist = 100.0f;
    int subsampleX = 4;
    int subsampleY = 4;

    // Geometry weighting
    GeometryWeighting weighting;

    // Multi-view correspondence limits
    int minMatch = 50;
    int maxCorrespondences = 0;  // 0 = unlimited
    int maxNeighbors = 0;        // 0 = unlimited

    bool verbose = false;
};

/**
 * Inner loop parameters (solver iterations).
 */
struct InnerParams
{
    // Solver selection
    SolverType solverType = SolverType::LevenbergMarquardt;

    // Iteration control
    int maxIterations = 12;
    double translationThreshold = 1e-4;
    double rotationThreshold = 1e-4;

    // GN damping (ignored for LM)
    double damping = 0.0;

    // LM parameters
    LMParams lm;

    // Line search
    LineSearchParams lineSearch;

    // Jacobian policy
    JacobianPolicy jacobianPolicy = JacobianPolicy::Simplified;

    // Ceres-specific (used when backend == Ceres)
    LinearSolverType linearSolverType = LinearSolverType::DenseQR;
    PreconditionerType preconditionerType = PreconditionerType::SchurJacobi;

    bool verbose = false;

    // Factory methods
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
};

// ============================================================================
// Ceres conversion (internal use)
// ============================================================================

/**
 * Convert InnerParams to Ceres solver options.
 * Used internally by Ceres backend.
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
// All defaults for each solver mode are grouped here for easy viewing/editing.

namespace Defaults
{

/// Single-pose ICP defaults (one moving grid aligned to fixed target)
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

/// Two-pose ICP defaults (two grids optimized simultaneously)
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

/// Multi-view ICP defaults (many grids optimized together)
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

} // namespace Defaults

} // namespace ICP
