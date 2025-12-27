#pragma once
/**
 * ICP parameters and configuration structures.
 */

// Standard C++ headers
#include <cmath>

// Ceres headers
#include <ceres/solver.h>

namespace ICP
{

/**
 * Incidence weighting mode for grazing angle handling.
 */
enum class WeightingMode
{
    Abs,      // weight = |c|
    SqrtAbs   // weight = sqrt(|c|)
};

/**
 * Geometry weighting parameters for incidence-based weighting.
 *
 * Controls how grazing angles are handled in ray-projection ICP.
 * When the ray direction is nearly parallel to the surface (small |n^T d|),
 * the correspondence is either down-weighted or rejected entirely.
 */
struct GeometryWeighting
{
    bool enable_weight = true;   // Apply incidence-based weighting
    bool enable_gate = true;     // Reject correspondences below tau threshold
    double tau = 0.2;            // Threshold for gating (0.1-0.4 typical)
    WeightingMode mode = WeightingMode::SqrtAbs;

    /**
     * Compute incidence weight from denominator c = n^T d.
     *
     * @param c  The denominator (dot product of normal and ray direction)
     * @return   Weight in [0, 1], or 0 if gated out
     */
    double weight(double c) const
    {
        double ac = std::abs(c);
        if (enable_gate && ac < tau)
        {
            return 0.0;
        }
        if (!enable_weight)
        {
            return 1.0;
        }
        ac = std::max(tau, std::min(1.0, ac));
        switch (mode)
        {
            case WeightingMode::Abs:
                return ac;
            case WeightingMode::SqrtAbs:
                return std::sqrt(ac);
        }
        return 1.0;  // fallback
    }
};

/**
 * Solver backend selection.
 */
enum class SolverBackend
{
    HandRolled,  // Hand-rolled implementation (GN or LM)
    Ceres        // Ceres-based implementation
};

/**
 * Inner solver type.
 */
enum class SolverType
{
    GaussNewton,         // Gauss-Newton solver (uses params.damping)
    LevenbergMarquardt   // Levenberg-Marquardt solver (uses params.lm.lambda, can be fixed or adaptive)
};

/**
 * Lambda update strategy for Levenberg-Marquardt.
 *
 * Controls how the damping parameter is adjusted based on step quality.
 */
enum class LMStrategy
{
    /// Simple: binary accept/reject with fixed multipliers.
    /// Accept: lambda *= lambdaDown, Reject: lambda *= lambdaUp
    Simple,

    /// Gain ratio with Nielsen update (Ceres-style).
    /// Uses rho = actual_reduction / model_reduction.
    /// Accept: radius = radius / max(1/3, 1 - (2*rho - 1)^3)
    /// Reject: radius /= decrease_factor, decrease_factor *= 2
    GainRatio
};

/**
 * Line search parameters for inner solver.
 */
struct ICPLineSearchParams
{
    bool enabled = false;
    int maxIterations = 10;
    double alpha = 1.0;      ///< Initial step size
    double beta = 0.5;       ///< Step reduction factor
};

/**
 * Parameters for inner solver (fixed correspondences).
 */
struct InnerParams
{
    SolverType solverType = SolverType::GaussNewton;  ///< Solver algorithm to use
    int maxIterations = 12;
    double stepTol = 1e-9;  ///< Legacy: single threshold for delta norm

    /// Translation convergence threshold (same units as data).
    /// Converged when ||delta_translation|| < translationThreshold.
    double translationThreshold = 1e-4;

    /// Rotation convergence threshold (radians).
    /// Converged when ||delta_rotation|| < rotationThreshold.
    ///
    /// Relationship to translation: at characteristic length L from origin,
    /// rotation θ causes displacement ≈ L*θ. For balanced convergence:
    ///   rotationThreshold ≈ translationThreshold / characteristicLength
    ///
    /// Default: 1e-4 rad ≈ 0.006° ≈ 20 arcsec (assumes L ≈ 1.0)
    double rotationThreshold = 1e-4;

    double damping = 0.0;  ///< Damping for Gauss-Newton (ignored if using LM)
    bool verbose = false;  ///< Print per-iteration RMS
    ICPLineSearchParams lineSearch;

    struct LevenbergMarquardt
    {
        LMStrategy strategy = LMStrategy::Simple;  ///< Lambda update strategy

        double lambda = 1e-3;       ///< Initial/fixed damping parameter
        bool fixedLambda = true;    ///< If true, use fixed lambda; if false, adapt lambda
        double lambdaUp = 10.0;     ///< Factor to increase lambda on reject (Simple strategy)
        double lambdaDown = 0.1;    ///< Factor to decrease lambda on accept (Simple strategy)
        double lambdaMin = 1e-10;   ///< Minimum lambda
        double lambdaMax = 1e10;    ///< Maximum lambda

        /// Minimum gain ratio (rho) for step acceptance (GainRatio strategy).
        /// Step accepted if rho > minRelativeDecrease.
        /// Ceres default: 1e-3
        double minRelativeDecrease = 1e-3;
    } lm;

    /**
     * Factory method for Gauss-Newton with all parameters explicit.
     * Use this to ensure no parameters are accidentally omitted.
     */
    static InnerParams gaussNewton(
        int maxIterations,
        double translationThreshold,
        double rotationThreshold,
        double damping = 0.0,
        bool lineSearchEnabled = false,
        bool verbose = false)
    {
        InnerParams p;
        p.solverType = SolverType::GaussNewton;
        p.maxIterations = maxIterations;
        p.translationThreshold = translationThreshold;
        p.rotationThreshold = rotationThreshold;
        p.damping = damping;
        p.lineSearch.enabled = lineSearchEnabled;
        p.verbose = verbose;
        return p;
    }

    /**
     * Factory method for Levenberg-Marquardt with all parameters explicit.
     * Use this to ensure no parameters are accidentally omitted.
     */
    static InnerParams levenbergMarquardt(
        int maxIterations,
        double translationThreshold,
        double rotationThreshold,
        double lambda,
        bool adaptiveLambda = true,
        bool lineSearchEnabled = false,
        bool verbose = false)
    {
        InnerParams p;
        p.solverType = SolverType::LevenbergMarquardt;
        p.maxIterations = maxIterations;
        p.translationThreshold = translationThreshold;
        p.rotationThreshold = rotationThreshold;
        p.lm.lambda = lambda;
        p.lm.fixedLambda = !adaptiveLambda;
        p.lineSearch.enabled = lineSearchEnabled;
        p.verbose = verbose;
        return p;
    }
};

/**
 * Correspondence sampling strategy.
 */
enum class SamplingMode
{
    Fixed,     // Fixed correspondence set throughout optimization
    Manual,    // Manually controlled sampling
    Adaptive   // Adaptive sampling based on convergence
};

/**
 * Parameters for outer loop (correspondence updates).
 */
struct OuterParams
{
    int maxIterations = 6;
    double convergenceTol = 1e-9;  // Relative RMS change threshold
    float maxDist = 100.0f;        // Max ray distance for correspondences
    bool verbose = false;          // Print per-iteration information
};

/**
 * Session-level parameters for ICP registration.
 *
 * Controls how grids are loaded and how the initial alignment is computed.
 * These parameters apply to the overall registration session, not individual
 * solver iterations.
 */
struct ICPSessionParams
{
    /// Solver backend selection
    SolverBackend backend = SolverBackend::HandRolled;

    /// If true, compute initial alignment from grid poses: T_source * T_target^{-1}
    /// If false, start from identity (or user-supplied initial pose)
    bool useGridPoses = false;

    /// For two-pose solver: hold first pose fixed (removes gauge freedom)
    bool fixPoseA = false;

    /// Verbose output for session-level operations
    bool verbose = false;
};

/**
 * Ceres-specific solver options.
 */
struct CeresICPOptions
{
    int maxIterations = 12;
    double functionTolerance = 1e-9;
    double gradientTolerance = 1e-9;
    double parameterTolerance = 1e-9;

    bool useLM = false;  // Use Levenberg-Marquardt (true) or Gauss-Newton (false)
    double initialTrustRegionRadius = 1e4;
    double maxTrustRegionRadius = 1e8;

    ceres::LinearSolverType linearSolverType = ceres::DENSE_QR;

    bool verbose = false;
    bool silent = false;  // Suppress all output
};

} // namespace ICP
