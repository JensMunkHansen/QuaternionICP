#pragma once

// Standard C++ headers
#include <string>
#include <vector>

// args headers
#include <args.hxx>

namespace ICP
{

struct CommonOptions
{
    // Backend selection
    enum class Backend
    {
        HandRolled7D,  // Hand-rolled 7D solver
        Ceres7         // Ceres solver with 7D parameterization
    };

    // Test grid type
    enum class GridType
    {
        Heightfield,      // Sinusoidal heightfield
        TwoHemispheres    // Two hemispheres (constrains all 6 DOF)
    };

    // Mode
    bool useTestGrid = false;
    Backend backend = Backend::HandRolled7D;
    GridType gridType = GridType::TwoHemispheres;

    // Input files
    std::string sourceFile;
    std::string targetFile;

    // Synthetic grid parameters (test mode)
    int gridWidth = 32;
    int gridHeight = 32;

    // Heightfield parameters
    float gridAmplitude = 0.3f;
    float gridFrequency = 2.0f;

    // Two hemispheres parameters
    float hemisphereRadius = 2.0f;
    float hemisphereSeparation = 6.0f;

    // Perturbation
    float rotationNoise = 0.0f;      // degrees
    float translationNoise = 0.0f;   // units
    float depthNoise = 0.0f;         // depth noise

    // ========================================================================
    // ICP parameters
    // ========================================================================

    // Outer loop
    int outerIterations = 10;
    double rmsTol = 1e-9;  // Outer convergence tolerance

    // Inner loop
    int innerIterations = 5;
    double translationThreshold = 1e-4;  // Translation convergence (same units as data)
    double rotationThreshold = 1e-4;     // Rotation convergence (radians)

    // Solver type selection
    enum class Solver
    {
        GaussNewton,
        LevenbergMarquardt
    };
    Solver solver = Solver::GaussNewton;

    // Line search parameters (nested struct for grouping)
    struct LineSearch
    {
        bool enabled = false;
        int maxIterations = 10;
        double alpha = 1.0;   // Initial step size
        double beta = 0.5;    // Step reduction factor
    } lineSearch;

    // Levenberg-Marquardt parameters (nested struct for grouping)
    struct LevenbergMarquardt
    {
        double lambda = 1e-3;       // Initial/fixed lambda
        bool fixedLambda = true;    // Use fixed lambda (true) or adaptive (false)
        double lambdaUp = 10.0;     // Lambda increase factor
        double lambdaDown = 0.1;    // Lambda decrease factor
        double lambdaMin = 1e-10;   // Minimum lambda
        double lambdaMax = 1e10;    // Maximum lambda
    } lm;

    // Incidence weighting
    bool enableIncidenceWeight = true;
    bool enableGrazingGate = true;
    double incidenceTau = 0.2;

    // Subsampling
    int subsampleX = 1;
    int subsampleY = 1;

    // Output
    bool verbose = false;
};

// Parse command-line arguments into CommonOptions
// Returns true on success, false if help was requested or error occurred
bool parseArgs(int argc, char** argv, CommonOptions& opts, const std::string& programDesc = "ICP");

// Forward declarations
struct InnerParams;
struct OuterParams;
struct CeresICPOptions;

/**
 * Convert CommonOptions to InnerParams.
 */
InnerParams commonOptionsToInnerParams(const CommonOptions& opts);

/**
 * Convert CommonOptions to OuterParams.
 */
OuterParams commonOptionsToOuterParams(const CommonOptions& opts);

/**
 * Convert CommonOptions to CeresICPOptions.
 */
CeresICPOptions commonOptionsToCeresOptions(const CommonOptions& opts);

} // namespace ICP
