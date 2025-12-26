#include <ICP/CommonOptions.h>
#include <iostream>

namespace ICP
{

bool parseArgs(int argc, char** argv, CommonOptions& opts, const std::string& programDesc)
{
    args::ArgumentParser parser(programDesc);
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::CompletionFlag completion(parser, {"complete"});

    // Backend
    args::MapFlag<std::string, CommonOptions::Backend> backend_flag(parser, "backend",
        "Solver backend", {"backend"},
        {{"handrolled7D", CommonOptions::Backend::HandRolled7D},
         {"ceres7", CommonOptions::Backend::Ceres7}},
        CommonOptions::Backend::HandRolled7D);

    // Mode
    args::Flag test_flag(parser, "test", "Use synthetic test grids", {"test"});
    args::ValueFlag<std::string> source_flag(parser, "path", "Source EXR file", {"source"});
    args::ValueFlag<std::string> target_flag(parser, "path", "Target EXR file", {"target"});

    // Grid type
    args::MapFlag<std::string, CommonOptions::GridType> gridtype_flag(parser, "type",
        "Test grid type", {"grid-type"},
        {{"heightfield", CommonOptions::GridType::Heightfield},
         {"hemispheres", CommonOptions::GridType::TwoHemispheres}},
        CommonOptions::GridType::TwoHemispheres);

    // Synthetic grid dimensions
    args::ValueFlag<int> width_flag(parser, "int", "Grid width (test mode)", {"width"}, 32);
    args::ValueFlag<int> height_flag(parser, "int", "Grid height (test mode)", {"height"}, 32);

    // Heightfield parameters
    args::ValueFlag<float> amp_flag(parser, "float", "Surface amplitude (heightfield)", {"amplitude"}, 0.3f);
    args::ValueFlag<float> freq_flag(parser, "float", "Surface frequency (heightfield)", {"frequency"}, 2.0f);

    // Two hemispheres parameters
    args::ValueFlag<float> radius_flag(parser, "float", "Hemisphere radius", {"hemisphere-radius"}, 2.0f);
    args::ValueFlag<float> sep_flag(parser, "float", "Hemisphere separation", {"hemisphere-separation"}, 6.0f);

    // Perturbation
    args::ValueFlag<float> rot_flag(parser, "deg", "Rotation noise (degrees)", {'r', "rot-stddev"}, 0.0f);
    args::ValueFlag<float> trans_flag(parser, "units", "Translation noise", {'t', "trans-stddev"}, 0.0f);
    args::ValueFlag<float> depth_flag(parser, "float", "Depth noise", {'d', "depth-noise"}, 0.0f);

    // ICP parameters
    args::ValueFlag<int> outer_flag(parser, "int", "Max outer iterations", {'o', "outer-iterations"}, 10);
    args::ValueFlag<int> inner_flag(parser, "int", "Max inner iterations", {'i', "inner-iterations"}, 5);
    args::ValueFlag<double> stepTol(parser, "tol", "Step tolerance (default: 1e-9)", {"step-tol"});
    args::ValueFlag<double> rmsTol(parser, "tol", "RMS tolerance (default: 1e-9)", {"rms-tol"});
    args::ValueFlag<double> damping(parser, "val", "Damping factor (default: 0.0)", {"damping"});

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

    // Backend selection
    if (backend_flag) opts.backend = args::get(backend_flag);

    // Grid type selection
    if (gridtype_flag) opts.gridType = args::get(gridtype_flag);

    // Validate mode: either --test or both --source and --target
    opts.useTestGrid = test_flag;

    if (opts.useTestGrid)
    {
        // Test mode: source and target files not needed
        if (source_flag || target_flag)
        {
            std::cerr << "Warning: --source and --target ignored in --test mode\n";
        }
    }
    else
    {
        // File mode: both source and target required
        if (!source_flag || !target_flag)
        {
            std::cerr << "Error: --source and --target required (or use --test for synthetic grids)\n";
            std::cerr << parser;
            return false;
        }
        opts.sourceFile = args::get(source_flag);
        opts.targetFile = args::get(target_flag);
    }

    // Synthetic grid parameters
    if (width_flag) opts.gridWidth = args::get(width_flag);
    if (height_flag) opts.gridHeight = args::get(height_flag);
    if (amp_flag) opts.gridAmplitude = args::get(amp_flag);
    if (freq_flag) opts.gridFrequency = args::get(freq_flag);
    if (radius_flag) opts.hemisphereRadius = args::get(radius_flag);
    if (sep_flag) opts.hemisphereSeparation = args::get(sep_flag);

    // Perturbation parameters
    if (rot_flag) opts.rotationNoise = args::get(rot_flag);
    if (trans_flag) opts.translationNoise = args::get(trans_flag);
    if (depth_flag) opts.depthNoise = args::get(depth_flag);

    if (outer_flag) opts.outerIterations = args::get(outer_flag);
    if (inner_flag) opts.innerIterations = args::get(inner_flag);
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
