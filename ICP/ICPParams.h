#pragma once
/**
 * ICP parameters and configuration structures.
 */

#include <cmath>
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
 * Inner solver type.
 */
enum class SolverType
{
    GaussNewton,         // Gauss-Newton solver (damping = 0)
    LevenbergMarquardt   // Levenberg-Marquardt solver (adaptive damping)
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
    int maxIterations = 12;
    double stepTol = 1e-9;
    double damping = 0.0;  // LM damping (0 = Gauss-Newton)
    bool verbose = false;  // Print per-iteration RMS
    ICPLineSearchParams lineSearch;

    struct LevenbergMarquardt
    {
        double lambda = 1e-3;       ///< Initial damping parameter
        bool fixedLambda = true;    ///< If true, don't adapt lambda
        double lambdaUp = 10.0;     ///< Factor to increase lambda on reject
        double lambdaDown = 0.1;    ///< Factor to decrease lambda on accept
        double lambdaMin = 1e-10;   ///< Minimum lambda
        double lambdaMax = 1e10;    ///< Maximum lambda
    } lm;
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
