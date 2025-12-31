// Standard C++ headers
#include <iostream>
#include <random>

// Internal headers
#include <ICP/CommonOptions.h>
#include <ICP/EigenUtils.h>
#include <ICP/ICPCeresSolver.h>
#include <ICP/ICPParams.h>
#include <ICP/ICPSolver.h>
#include <ICP/SE3.h>
#include <ICP/TestGridUtils.h>

using namespace ICP;

int main(int argc, char** argv)
{
    CommonOptions opts;
    opts.useGridPoses = false;  // SingleICP default: synthetic grids don't have poses
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
        // Load grids from EXR files
        if (!source.loadFromExr(opts.sourceFile))
        {
            std::cerr << "Error: Failed to load source EXR: " << opts.sourceFile << "\n";
            return 1;
        }
        if (!target.loadFromExr(opts.targetFile))
        {
            std::cerr << "Error: Failed to load target EXR: " << opts.targetFile << "\n";
            return 1;
        }

        if (opts.verbose)
        {
            std::cout << "Loaded source: " << source.filename << " (" << source.width << "x" << source.height << ")\n";
            std::cout << "Loaded target: " << target.filename << " (" << target.width << "x" << target.height << ")\n";
        }
    }

    // Create initial pose
    Pose7 initialPose;
    if (opts.useGridPoses)
    {
        // Compute initial alignment from grid poses: T_source * T_target^{-1}
        Eigen::Isometry3d relPose = source.initialPose * target.initialPose.inverse();
        initialPose = isometryToPose7(relPose);

        if (opts.verbose)
        {
            std::cout << "\nUsing grid poses for initial alignment:\n";
            std::cout << "  Initial pose: q=[" << initialPose[0] << ", " << initialPose[1] << ", "
                      << initialPose[2] << ", " << initialPose[3] << "], t=["
                      << initialPose[4] << ", " << initialPose[5] << ", " << initialPose[6] << "]\n";
        }
    }
    else
    {
        initialPose = identityPose();
    }

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
    printCommonConfig(opts);

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

        CeresICPResult ceresResult = solveICPCeres(source, target, initialPose, rayDir,
                                                    weighting, innerParams, outerParams);

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
