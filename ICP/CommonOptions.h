#pragma once

#include <args.hxx>
#include <string>
#include <vector>

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

    // ICP parameters
    int outerIterations = 10;
    int innerIterations = 5;
    double stepTol = 1e-9;
    double rmsTol = 1e-9;
    double damping = 0.0;

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

} // namespace ICP
