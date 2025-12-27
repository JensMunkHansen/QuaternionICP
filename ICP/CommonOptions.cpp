// Standard C++ headers
#include <iostream>
#include <regex>
#include <set>
#include <sstream>

// Internal headers
#include <ICP/CommonOptions.h>
#include <ICP/ICPParams.h>

namespace ICP
{

/**
 * Parse grid indices from a string.
 * Supports: "0,1,3" or "0-9" or "0-5,10,15-20" or "" (empty = all)
 * Returns sorted, deduplicated indices.
 */
std::vector<int> parseGridIndices(const std::string& str)
{
    std::set<int> indices;
    if (str.empty())
    {
        return {};  // Empty means load all
    }

    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start == std::string::npos)
            continue;
        token = token.substr(start, end - start + 1);

        // Check for range (e.g., "0-5")
        size_t dashPos = token.find('-');
        if (dashPos != std::string::npos && dashPos > 0)
        {
            int rangeStart = std::stoi(token.substr(0, dashPos));
            int rangeEnd = std::stoi(token.substr(dashPos + 1));
            for (int i = rangeStart; i <= rangeEnd; ++i)
            {
                indices.insert(i);
            }
        }
        else
        {
            indices.insert(std::stoi(token));
        }
    }

    return std::vector<int>(indices.begin(), indices.end());
}

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

    // Multi-grid mode
    args::ValueFlag<std::string> grid_folder(parser, "path", "Folder containing EXR files", {'g', "grid-folder"});
    args::ValueFlag<std::string> grid_indices(parser, "indices", "Grid indices: 0,1,3 or 0-9 or 0-5,10 (empty=all)", {"indices"});

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
    args::ValueFlag<double> rmsTol(parser, "tol", "RMS tolerance (default: 1e-9)", {"rms-tol"});

    // Convergence thresholds
    args::ValueFlag<double> transThreshold(parser, "tol", "Translation convergence threshold (default: 1e-4)", {"trans-threshold"});
    args::ValueFlag<double> rotThreshold(parser, "tol", "Rotation convergence threshold in radians (default: 1e-4)", {"rot-threshold"});

    // Solver type
    args::MapFlag<std::string, CommonOptions::Solver> solver_flag(parser, "solver",
        "Solver type", {"solver"},
        {{"gn", CommonOptions::Solver::GaussNewton},
         {"lm", CommonOptions::Solver::LevenbergMarquardt}},
        CommonOptions::Solver::GaussNewton);

    // Linear solver type for Ceres
    args::MapFlag<std::string, CommonOptions::LinearSolver> linear_solver_flag(parser, "linear-solver",
        "Linear solver type for Ceres", {"linear-solver"},
        {{"dense-qr", CommonOptions::LinearSolver::DenseQR},
         {"dense-schur", CommonOptions::LinearSolver::DenseSchur},
         {"sparse-schur", CommonOptions::LinearSolver::SparseSchur},
         {"iterative-schur", CommonOptions::LinearSolver::IterativeSchur}},
        CommonOptions::LinearSolver::DenseQR);

    // Jacobian policy
    args::MapFlag<std::string, CommonOptions::Jacobian> jacobian_flag(parser, "jacobian",
        "Jacobian policy: simplified (default) or consistent", {"jacobian"},
        {{"simplified", CommonOptions::Jacobian::Simplified},
         {"consistent", CommonOptions::Jacobian::Consistent}},
        CommonOptions::Jacobian::Simplified);

    // Line search parameters
    args::Flag lineSearchFlag(parser, "line-search", "Enable line search", {"line-search"});
    args::ValueFlag<int> lsMaxIter(parser, "int", "Line search max iterations (default: 10)", {"ls-max-iter"});
    args::ValueFlag<double> lsAlpha(parser, "val", "Line search initial step size (default: 1.0)", {"ls-alpha"});
    args::ValueFlag<double> lsBeta(parser, "val", "Line search step reduction factor (default: 0.5)", {"ls-beta"});

    // Levenberg-Marquardt parameters
    args::ValueFlag<double> lmLambda(parser, "val", "LM lambda (default: 1e-3)", {"lm-lambda"});
    args::Flag lmAdaptive(parser, "adaptive", "Use adaptive LM (default: fixed lambda)", {"lm-adaptive"});
    args::ValueFlag<double> lmLambdaUp(parser, "val", "LM lambda increase factor (default: 10.0)", {"lm-lambda-up"});
    args::ValueFlag<double> lmLambdaDown(parser, "val", "LM lambda decrease factor (default: 0.1)", {"lm-lambda-down"});
    args::ValueFlag<double> lmLambdaMin(parser, "val", "LM minimum lambda (default: 1e-10)", {"lm-lambda-min"});
    args::ValueFlag<double> lmLambdaMax(parser, "val", "LM maximum lambda (default: 1e10)", {"lm-lambda-max"});

    // Incidence weighting
    args::Flag noIncidenceWeight(parser, "no-weight", "Disable incidence weighting", {"no-incidence-weight"});
    args::Flag noGrazingGate(parser, "no-gate", "Disable grazing angle gate", {"no-grazing-gate"});
    args::ValueFlag<double> incidenceTau(parser, "tau", "Incidence tau threshold (default: 0.2)", {"incidence-tau"});

    // Subsampling
    args::ValueFlag<int> subsampleX(parser, "N", "Subsample X stride (default: 4)", {"subsample-x"});
    args::ValueFlag<int> subsampleY(parser, "N", "Subsample Y stride (default: 4)", {"subsample-y"});
    args::ValueFlag<int> subsample(parser, "N", "Subsample both X and Y (default: 4)", {"subsample"});

    // Session options
    args::Flag useGridPoses(parser, "use-grid-poses",
        "Compute initial alignment from grid poses (T_source * T_target^-1)", {"use-grid-poses"});
    args::Flag fixFirstPose(parser, "fix-first-pose",
        "For multi-pose solver: hold first pose fixed", {"fix-first-pose"});

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

    // Validate mode: --test, --source/--target pair, or --grid-folder
    opts.useTestGrid = test_flag;

    if (opts.useTestGrid)
    {
        // Test mode: source and target files not needed
        if (source_flag || target_flag || grid_folder)
        {
            std::cerr << "Warning: --source/--target/--grid-folder ignored in --test mode\n";
        }
    }
    else if (grid_folder)
    {
        // Multi-grid mode: load from folder
        opts.gridFolder = args::get(grid_folder);
        if (grid_indices)
        {
            opts.gridIndices = parseGridIndices(args::get(grid_indices));
        }
        if (source_flag || target_flag)
        {
            std::cerr << "Warning: --source and --target ignored in --grid-folder mode\n";
        }
    }
    else
    {
        // Two-file mode: both source and target required
        if (!source_flag || !target_flag)
        {
            std::cerr << "Error: --source and --target required (or use --test or --grid-folder)\n";
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
    if (rmsTol) opts.rmsTol = args::get(rmsTol);

    // Convergence thresholds
    if (transThreshold) opts.translationThreshold = args::get(transThreshold);
    if (rotThreshold) opts.rotationThreshold = args::get(rotThreshold);

    // Solver type
    if (solver_flag) opts.solver = args::get(solver_flag);
    if (linear_solver_flag) opts.linearSolver = args::get(linear_solver_flag);
    if (jacobian_flag) opts.jacobianPolicy = args::get(jacobian_flag);

    // Line search parameters
    opts.lineSearch.enabled = lineSearchFlag;
    if (lsMaxIter) opts.lineSearch.maxIterations = args::get(lsMaxIter);
    if (lsAlpha) opts.lineSearch.alpha = args::get(lsAlpha);
    if (lsBeta) opts.lineSearch.beta = args::get(lsBeta);

    // LM parameters
    if (lmLambda) opts.lm.lambda = args::get(lmLambda);
    opts.lm.fixedLambda = !lmAdaptive;  // Default is fixed, --lm-adaptive makes it adaptive
    if (lmLambdaUp) opts.lm.lambdaUp = args::get(lmLambdaUp);
    if (lmLambdaDown) opts.lm.lambdaDown = args::get(lmLambdaDown);
    if (lmLambdaMin) opts.lm.lambdaMin = args::get(lmLambdaMin);
    if (lmLambdaMax) opts.lm.lambdaMax = args::get(lmLambdaMax);

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
    opts.useGridPoses = useGridPoses;
    opts.fixFirstPose = fixFirstPose;

    return true;
}

InnerParams commonOptionsToInnerParams(const CommonOptions& opts)
{
    InnerParams params;

    // Solver type
    params.solverType = (opts.solver == CommonOptions::Solver::LevenbergMarquardt)
                            ? SolverType::LevenbergMarquardt
                            : SolverType::GaussNewton;

    // Iteration limits
    params.maxIterations = opts.innerIterations;

    // Convergence thresholds
    params.translationThreshold = opts.translationThreshold;
    params.rotationThreshold = opts.rotationThreshold;

    // GN damping (always 0 for pure GN; use LM for regularization)
    params.damping = 0.0;

    // Line search parameters
    params.lineSearch.enabled = opts.lineSearch.enabled;
    params.lineSearch.maxIterations = opts.lineSearch.maxIterations;
    params.lineSearch.alpha = opts.lineSearch.alpha;
    params.lineSearch.beta = opts.lineSearch.beta;

    // LM parameters
    params.lm.lambda = opts.lm.lambda;
    params.lm.fixedLambda = opts.lm.fixedLambda;
    params.lm.lambdaUp = opts.lm.lambdaUp;
    params.lm.lambdaDown = opts.lm.lambdaDown;
    params.lm.lambdaMin = opts.lm.lambdaMin;
    params.lm.lambdaMax = opts.lm.lambdaMax;

    params.verbose = opts.verbose;

    return params;
}

OuterParams commonOptionsToOuterParams(const CommonOptions& opts)
{
    OuterParams params;
    params.maxIterations = opts.outerIterations;
    params.convergenceTol = opts.rmsTol;
    params.subsampleX = opts.subsampleX;
    params.subsampleY = opts.subsampleY;
    params.verbose = opts.verbose;
    return params;
}

CeresICPOptions commonOptionsToCeresOptions(const CommonOptions& opts)
{
    CeresICPOptions ceresOpts;

    // Iteration and tolerance settings
    ceresOpts.maxIterations = opts.innerIterations;
    ceresOpts.functionTolerance = opts.translationThreshold;
    ceresOpts.gradientTolerance = opts.translationThreshold;
    ceresOpts.parameterTolerance = opts.translationThreshold;

    // Solver type: LM vs GN
    if (opts.solver == CommonOptions::Solver::LevenbergMarquardt)
    {
        ceresOpts.useLM = true;
        // Convert lambda to trust region radius: radius = 1 / lambda
        double lambda = opts.lm.lambda;
        ceresOpts.initialTrustRegionRadius = (lambda > 0) ? 1.0 / lambda : 1e4;
        ceresOpts.maxTrustRegionRadius = (opts.lm.lambdaMin > 0) ? 1.0 / opts.lm.lambdaMin : 1e8;
    }
    else  // Gauss-Newton
    {
        ceresOpts.useLM = false;
        // Large trust region for GN approximation
        ceresOpts.initialTrustRegionRadius = 1e16;
        ceresOpts.maxTrustRegionRadius = 1e32;
    }

    // Linear solver type
    switch (opts.linearSolver)
    {
        case CommonOptions::LinearSolver::DenseQR:
            ceresOpts.linearSolverType = ceres::DENSE_QR;
            break;
        case CommonOptions::LinearSolver::DenseSchur:
            ceresOpts.linearSolverType = ceres::DENSE_SCHUR;
            break;
        case CommonOptions::LinearSolver::SparseSchur:
            ceresOpts.linearSolverType = ceres::SPARSE_SCHUR;
            break;
        case CommonOptions::LinearSolver::IterativeSchur:
            ceresOpts.linearSolverType = ceres::ITERATIVE_SCHUR;
            ceresOpts.preconditionerType = ceres::SCHUR_JACOBI;
            break;
    }

    // Jacobian policy
    ceresOpts.jacobianPolicy = (opts.jacobianPolicy == CommonOptions::Jacobian::Consistent)
        ? JacobianPolicy::Consistent : JacobianPolicy::Simplified;

    ceresOpts.verbose = opts.verbose;
    ceresOpts.silent = !opts.verbose;

    return ceresOpts;
}

ICPSessionParams commonOptionsToSessionParams(const CommonOptions& opts)
{
    ICPSessionParams params;
    params.backend = (opts.backend == CommonOptions::Backend::Ceres7)
                         ? SolverBackend::Ceres
                         : SolverBackend::HandRolled;
    params.useGridPoses = opts.useGridPoses;
    params.fixFirstPose = opts.fixFirstPose;
    params.verbose = opts.verbose;
    return params;
}

void printCommonConfig(const CommonOptions& opts)
{
    std::cout << "\n=== ICP Configuration ===\n";
    std::cout << "Backend: " << (opts.backend == CommonOptions::Backend::Ceres7 ? "Ceres" : "HandRolled") << "\n";

    if (opts.backend == CommonOptions::Backend::Ceres7)
    {
        std::cout << "Jacobian: "
                  << (opts.jacobianPolicy == CommonOptions::Jacobian::Consistent
                      ? "consistent" : "simplified") << "\n";

        std::cout << "Linear solver: ";
        switch (opts.linearSolver)
        {
            case CommonOptions::LinearSolver::DenseQR: std::cout << "DENSE_QR"; break;
            case CommonOptions::LinearSolver::DenseSchur: std::cout << "DENSE_SCHUR"; break;
            case CommonOptions::LinearSolver::SparseSchur: std::cout << "SPARSE_SCHUR"; break;
            case CommonOptions::LinearSolver::IterativeSchur: std::cout << "ITERATIVE_SCHUR"; break;
        }
        std::cout << "\n";
    }

    std::cout << "\nOuter loop:\n";
    std::cout << "  Max iterations: " << opts.outerIterations << "\n";
    std::cout << "  Convergence tolerance: " << opts.rmsTol << "\n";

    std::cout << "\nInner loop:\n";
    std::cout << "  Solver: " << (opts.solver == CommonOptions::Solver::LevenbergMarquardt ? "LM" : "GN") << "\n";
    std::cout << "  Max iterations: " << opts.innerIterations << "\n";
    std::cout << "  Translation threshold: " << opts.translationThreshold << "\n";
    std::cout << "  Rotation threshold: " << opts.rotationThreshold << " rad\n";

    if (opts.lineSearch.enabled && opts.backend == CommonOptions::Backend::HandRolled7D)
    {
        std::cout << "  Line search: enabled (alpha=" << opts.lineSearch.alpha
                  << ", beta=" << opts.lineSearch.beta << ")\n";
    }

    if (opts.solver == CommonOptions::Solver::LevenbergMarquardt)
    {
        std::cout << "  LM lambda: " << opts.lm.lambda
                  << (opts.lm.fixedLambda ? " (fixed)" : " (adaptive)") << "\n";
    }

    std::cout << "\nGeometry weighting:\n";
    std::cout << "  Incidence weighting: " << (opts.enableIncidenceWeight ? "enabled" : "disabled") << "\n";
    std::cout << "  Grazing angle gate: " << (opts.enableGrazingGate ? "enabled" : "disabled") << "\n";
    std::cout << "  Tau threshold: " << opts.incidenceTau << "\n";

    std::cout << "\nSubsampling: " << opts.subsampleX << "x" << opts.subsampleY << "\n";
}

} // namespace ICP
