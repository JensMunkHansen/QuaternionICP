#include <ICP/CommonOptions.h>
#include <iostream>

namespace ICP
{

bool parseArgs(int argc, char** argv, CommonOptions& opts, const std::string& programDesc)
{
    args::ArgumentParser parser(programDesc);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    // Positional arguments
    args::Positional<std::string> sourceFile(parser, "source", "Source grid EXR file");
    args::Positional<std::string> targetFile(parser, "target", "Target grid EXR file");

    // ICP parameters
    args::ValueFlag<int> outerIter(parser, "N", "Outer iterations (default: 6)", {"outer"});
    args::ValueFlag<int> innerIter(parser, "N", "Inner iterations (default: 12)", {"inner"});
    args::ValueFlag<double> stepTol(parser, "tol", "Step tolerance (default: 1e-9)", {"step-tol"});
    args::ValueFlag<double> rmsTol(parser, "tol", "RMS tolerance (default: 1e-9)", {"rms-tol"});
    args::ValueFlag<double> damping(parser, "val", "Damping factor (default: 1e-6)", {"damping"});

    // Incidence weighting
    args::Flag noIncidenceWeight(parser, "no-weight", "Disable incidence weighting", {"no-incidence-weight"});
    args::Flag noGrazingGate(parser, "no-gate", "Disable grazing angle gate", {"no-grazing-gate"});
    args::ValueFlag<double> incidenceTau(parser, "tau", "Incidence tau threshold (default: 0.2)", {"incidence-tau"});

    // Subsampling
    args::ValueFlag<int> subsampleX(parser, "N", "Subsample X stride (default: 1)", {"subsample-x"});
    args::ValueFlag<int> subsampleY(parser, "N", "Subsample Y stride (default: 1)", {"subsample-y"});
    args::ValueFlag<int> subsample(parser, "N", "Subsample both X and Y (default: 1)", {"subsample"});

    // Verbose
    args::Flag verbose(parser, "verbose", "Verbose output", {'v', "verbose"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Help&)
    {
        std::cout << parser;
        return false;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << "\n";
        std::cerr << parser;
        return false;
    }

    if (!sourceFile || !targetFile)
    {
        std::cerr << "Error: source and target files required\n";
        std::cerr << parser;
        return false;
    }

    // Fill options struct
    opts.sourceFile = args::get(sourceFile);
    opts.targetFile = args::get(targetFile);

    if (outerIter) opts.outerIterations = args::get(outerIter);
    if (innerIter) opts.innerIterations = args::get(innerIter);
    if (stepTol) opts.stepTol = args::get(stepTol);
    if (rmsTol) opts.rmsTol = args::get(rmsTol);
    if (damping) opts.damping = args::get(damping);

    opts.enableIncidenceWeight = !noIncidenceWeight;
    opts.enableGrazingGate = !noGrazingGate;
    if (incidenceTau) opts.incidenceTau = args::get(incidenceTau);

    if (subsample)
    {
        opts.subsampleX = args::get(subsample);
        opts.subsampleY = args::get(subsample);
    }
    if (subsampleX) opts.subsampleX = args::get(subsampleX);
    if (subsampleY) opts.subsampleY = args::get(subsampleY);

    opts.verbose = verbose;

    return true;
}

} // namespace ICP
