#pragma once

#include <args.hxx>
#include <string>
#include <vector>

namespace ICP
{

struct CommonOptions
{
    // Input files
    std::string sourceFile;
    std::string targetFile;

    // ICP parameters
    int outerIterations = 6;
    int innerIterations = 12;
    double stepTol = 1e-9;
    double rmsTol = 1e-9;
    double damping = 1e-6;

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
