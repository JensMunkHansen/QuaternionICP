#include <ICP/CommonOptions.h>
#include <ICP/TestGridUtils.h>
#include <ICP/EigenUtils.h>
#include <ICP/SE3.h>
#include <ICP/ICPSimple.h>
#include <ICP/ICPCeres.h>
#include <ICP/ICPParams.h>
#include <random>
#include <iostream>

using namespace ICP;

int main(int argc, char** argv)
{
    CommonOptions opts;
    if (!parseArgs(argc, argv, opts, "SingleICP - Ray-projection ICP for two grids"))
    {
        return 1;
    }

    // Create or load grids
    Grid source, target;

    if (opts.useTestGrid)
    {
        auto grids = createTestGrids(opts, opts.verbose);
        source = grids.first;
        target = grids.second;
    }
    else
    {
        // TODO: Load grids from EXR files
        std::cerr << "Error: EXR file loading not yet implemented\n";
        return 1;
    }

    // Create initial pose with perturbations (if requested)
    Pose7 initialPose = identityPose();

    if (opts.rotationNoise > 0.0f || opts.translationNoise > 0.0f)
    {
        // Initialize RNGs with deterministic seeds for reproducibility
        PerturbationRNGs rngs;

        if (opts.verbose)
        {
            std::cout << "\nApplying pose perturbations:\n";
        }

        // Perturb rotation (in-place)
        if (opts.rotationNoise > 0.0f)
        {
            perturbPoseRotation(initialPose, opts.rotationNoise, rngs.rotation);

            if (opts.verbose)
            {
                Quaternion q(initialPose[3], initialPose[0], initialPose[1], initialPose[2]);
                Eigen::AngleAxisd aa(q);
                std::cout << "  Rotation: " << (aa.angle() * 180.0 / M_PI) << " deg around ["
                          << aa.axis().transpose() << "]\n";
            }
        }

        // Perturb translation (in-place)
        if (opts.translationNoise > 0.0f)
        {
            perturbPoseTranslation(initialPose, opts.translationNoise, rngs.translation);

            if (opts.verbose)
            {
                std::cout << "  Translation: [" << initialPose[4] << ", "
                          << initialPose[5] << ", " << initialPose[6] << "]\n";
            }
        }

        if (opts.verbose)
        {
            std::cout << "  Initial pose: q=[" << initialPose[0] << ", " << initialPose[1] << ", "
                      << initialPose[2] << ", " << initialPose[3] << "], t=["
                      << initialPose[4] << ", " << initialPose[5] << ", " << initialPose[6] << "]\n";
        }
    }

    // TODO: Apply depth noise if requested
    if (opts.depthNoise > 0.0f)
    {
        std::cerr << "Warning: Depth noise not yet implemented\n";
    }

    // Set up ICP parameters from command-line options
    InnerParams innerParams = commonOptionsToInnerParams(opts);
    OuterParams outerParams = commonOptionsToOuterParams(opts);

    GeometryWeighting weighting;
    weighting.enable_weight = opts.enableIncidenceWeight;
    weighting.enable_gate = opts.enableGrazingGate;
    weighting.tau = opts.incidenceTau;

    // Ray direction (looking down -Z in camera frame)
    Vector3 rayDir(0.0, 0.0, -1.0);

    // Display ICP configuration
    std::cout << "\n=== ICP Configuration ===\n";
    std::cout << "Backend: " << (opts.backend == CommonOptions::Backend::Ceres7 ? "Ceres" : "HandRolled") << "\n";
    std::cout << "\nOuter loop:\n";
    std::cout << "\tMax iterations: " << outerParams.maxIterations << "\n";
    std::cout << "\tConvergence tolerance: " << outerParams.convergenceTol << "\n";
    std::cout << "\tMax correspondence distance: " << outerParams.maxDist << "\n";
    std::cout << "\nInner loop:\n";
    std::cout << "\tSolver: " << (innerParams.solverType == SolverType::LevenbergMarquardt ? "LM" : "GN") << "\n";
    std::cout << "\tMax iterations: " << innerParams.maxIterations << "\n";
    std::cout << "\tTranslation threshold: " << innerParams.translationThreshold << "\n";
    std::cout << "\tRotation threshold: " << innerParams.rotationThreshold << " rad\n";
    if (innerParams.lineSearch.enabled && opts.backend == CommonOptions::Backend::HandRolled7D)
    {
        std::cout << "\tLine search: enabled (alpha=" << innerParams.lineSearch.alpha
                  << ", beta=" << innerParams.lineSearch.beta << ")\n";
    }
    if (innerParams.solverType == SolverType::LevenbergMarquardt)
    {
        std::cout << "\tLM lambda: " << innerParams.lm.lambda
                  << (innerParams.lm.fixedLambda ? " (fixed)" : " (adaptive)") << "\n";
    }
    std::cout << "\nGeometry weighting:\n";
    std::cout << "\tIncidence weighting: " << (weighting.enable_weight ? "enabled" : "disabled") << "\n";
    std::cout << "\tGrazing angle gate: " << (weighting.enable_gate ? "enabled" : "disabled") << "\n";
    std::cout << "\tTau threshold: " << weighting.tau << "\n";

    // Run ICP solver based on backend selection
    Pose7 finalPose;
    double finalRms;
    int outerIterations;
    int totalInnerIterations;
    bool converged;

    if (opts.backend == CommonOptions::Backend::Ceres7)
    {
        if (opts.verbose)
        {
            std::cout << "\nRunning Ceres ICP solver...\n";
        }
        else
        {
            std::cout << "\nRunning ICP (Ceres)...\n";
        }

        CeresICPOptions ceresOpts = commonOptionsToCeresOptions(opts);
        CeresICPResult ceresResult = solveICPCeres(source, target, initialPose, rayDir,
                                                    weighting, ceresOpts, outerParams);

        finalPose = ceresResult.pose;
        finalRms = ceresResult.rms;
        outerIterations = ceresResult.outer_iterations;
        totalInnerIterations = ceresResult.total_inner_iterations;
        converged = ceresResult.converged;
    }
    else  // HandRolled7D
    {
        if (opts.verbose)
        {
            std::cout << "\nRunning hand-rolled ICP solver...\n";
        }
        else
        {
            std::cout << "\nRunning ICP...\n";
        }

        ICPResult icpResult = solveICP(source, target, initialPose, rayDir,
                                       weighting, innerParams, outerParams);

        finalPose = icpResult.pose;
        finalRms = icpResult.rms;
        outerIterations = icpResult.outer_iterations;
        totalInnerIterations = icpResult.total_inner_iterations;
        converged = icpResult.converged;
    }

    // Display results
    std::cout << "\n=== ICP Results ===\n";
    std::cout << "\tConverged: " << (converged ? "yes" : "no") << "\n";
    std::cout << "\tOuter iterations: " << outerIterations << "\n";
    std::cout << "\tTotal inner iterations: " << totalInnerIterations << "\n";
    std::cout << "\tFinal RMS: " << std::scientific << std::setprecision(6)
              << finalRms << "\n";

    if (opts.verbose)
    {
        std::cout << "\n\tFinal pose:\n";
        std::cout << "\t\tq = [" << finalPose[0] << ", " << finalPose[1] << ", "
                  << finalPose[2] << ", " << finalPose[3] << "]\n";
        std::cout << "\t\tt = [" << finalPose[4] << ", " << finalPose[5] << ", "
                  << finalPose[6] << "]\n";

        // Display rotation as axis-angle
        Quaternion q_final(finalPose[3], finalPose[0],
                          finalPose[1], finalPose[2]);
        Eigen::AngleAxisd aa_final(q_final);
        std::cout << "\t\tRotation: " << (aa_final.angle() * 180.0 / M_PI)
                  << " deg around [" << aa_final.axis().transpose() << "]\n";
    }

    return 0;
}
