// Standard C++ headers
#include <iomanip>
#include <iostream>

// Internal headers
#include <ICP/CommonOptions.h>
#include <ICP/EigenUtils.h>
#include <ICP/Grid.h>
#include <ICP/ICPCeresTwoPose.h>
#include <ICP/ICPParams.h>
#include <ICP/SE3.h>
#include <ICP/TestGridUtils.h>

using namespace ICP;

int main(int argc, char** argv)
{
    CommonOptions opts;
    if (!parseArgs(argc, argv, opts, "MultiICP - Two-pose ray-projection ICP"))
    {
        return 1;
    }

    // Create or load grids
    Grid gridA, gridB;

    if (opts.useTestGrid)
    {
        auto grids = createTestGrids(opts, opts.verbose);
        gridA = grids.first;
        gridB = grids.second;
    }
    else
    {
        // Load grids from EXR files
        if (!gridA.loadFromExr(opts.sourceFile))
        {
            std::cerr << "Error: Failed to load source EXR: " << opts.sourceFile << "\n";
            return 1;
        }
        if (!gridB.loadFromExr(opts.targetFile))
        {
            std::cerr << "Error: Failed to load target EXR: " << opts.targetFile << "\n";
            return 1;
        }

        if (opts.verbose)
        {
            std::cout << "Loaded grid A: " << gridA.filename << " (" << gridA.width << "x" << gridA.height << ")\n";
            std::cout << "Loaded grid B: " << gridB.filename << " (" << gridB.width << "x" << gridB.height << ")\n";
        }
    }

    // Create initial poses
    Pose7 poseA, poseB;
    if (opts.useGridPoses)
    {
        // Use poses from the grid files
        poseA = isometryToPose7(gridA.pose);
        poseB = isometryToPose7(gridB.pose);

        if (opts.verbose)
        {
            std::cout << "\nUsing grid poses:\n";
            std::cout << "  Pose A: q=[" << poseA[0] << ", " << poseA[1] << ", "
                      << poseA[2] << ", " << poseA[3] << "], t=["
                      << poseA[4] << ", " << poseA[5] << ", " << poseA[6] << "]\n";
            std::cout << "  Pose B: q=[" << poseB[0] << ", " << poseB[1] << ", "
                      << poseB[2] << ", " << poseB[3] << "], t=["
                      << poseB[4] << ", " << poseB[5] << ", " << poseB[6] << "]\n";
        }
    }
    else
    {
        poseA = identityPose();
        poseB = identityPose();
    }

    // Apply perturbations if requested (for testing)
    if (opts.rotationNoise > 0.0f || opts.translationNoise > 0.0f)
    {
        PerturbationRNGs rngs;

        if (opts.verbose)
        {
            std::cout << "\nApplying pose perturbations to grid B:\n";
        }

        if (opts.rotationNoise > 0.0f)
        {
            perturbPoseRotation(poseB, opts.rotationNoise, rngs.rotation);

            if (opts.verbose)
            {
                Quaternion q(poseB[3], poseB[0], poseB[1], poseB[2]);
                Eigen::AngleAxisd aa(q);
                std::cout << "  Rotation: " << (aa.angle() * 180.0 / M_PI) << " deg around ["
                          << aa.axis().transpose() << "]\n";
            }
        }

        if (opts.translationNoise > 0.0f)
        {
            perturbPoseTranslation(poseB, opts.translationNoise, rngs.translation);

            if (opts.verbose)
            {
                std::cout << "  Translation: [" << poseB[4] << ", "
                          << poseB[5] << ", " << poseB[6] << "]\n";
            }
        }
    }

    // Set up ICP parameters
    CeresICPOptions ceresOpts = commonOptionsToCeresOptions(opts);
    OuterParams outerParams = commonOptionsToOuterParams(opts);

    GeometryWeighting weighting;
    weighting.enable_weight = opts.enableIncidenceWeight;
    weighting.enable_gate = opts.enableGrazingGate;
    weighting.tau = opts.incidenceTau;

    Vector3 rayDir(0.0, 0.0, -1.0);

    // Display configuration
    std::cout << "\n=== Two-Pose ICP Configuration ===\n";
    std::cout << "Outer loop:\n";
    std::cout << "\tMax iterations: " << outerParams.maxIterations << "\n";
    std::cout << "\tConvergence tolerance: " << outerParams.convergenceTol << "\n";
    std::cout << "\tMax correspondence distance: " << outerParams.maxDist << "\n";
    std::cout << "\nInner loop (Ceres):\n";
    std::cout << "\tMax iterations: " << ceresOpts.maxIterations << "\n";
    std::cout << "\tSolver: " << (ceresOpts.useLM ? "LM" : "GN") << "\n";
    std::cout << "\nGeometry weighting:\n";
    std::cout << "\tIncidence weighting: " << (weighting.enable_weight ? "enabled" : "disabled") << "\n";
    std::cout << "\tGrazing angle gate: " << (weighting.enable_gate ? "enabled" : "disabled") << "\n";
    std::cout << "\tTau threshold: " << weighting.tau << "\n";

    // Run two-pose Ceres ICP solver
    std::cout << "\nRunning Two-Pose Ceres ICP";
    if (opts.fixFirstPose)
    {
        std::cout << " (first pose fixed)";
    }
    std::cout << "...\n";

    auto result = solveICPCeresTwoPose<RayJacobianSimplified>(
        gridA, gridB, poseA, poseB, rayDir, weighting, ceresOpts, outerParams,
        opts.fixFirstPose);

    // Display results
    std::cout << "\n=== Two-Pose ICP Results ===\n";
    std::cout << "\tConverged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "\tOuter iterations: " << result.outer_iterations << "\n";
    std::cout << "\tTotal inner iterations: " << result.total_inner_iterations << "\n";
    std::cout << "\tFinal RMS: " << std::scientific << std::setprecision(6)
              << result.rms << "\n";

    if (opts.verbose)
    {
        std::cout << "\n\tFinal pose A:\n";
        std::cout << "\t\tq = [" << result.poseA[0] << ", " << result.poseA[1] << ", "
                  << result.poseA[2] << ", " << result.poseA[3] << "]\n";
        std::cout << "\t\tt = [" << result.poseA[4] << ", " << result.poseA[5] << ", "
                  << result.poseA[6] << "]\n";

        std::cout << "\n\tFinal pose B:\n";
        std::cout << "\t\tq = [" << result.poseB[0] << ", " << result.poseB[1] << ", "
                  << result.poseB[2] << ", " << result.poseB[3] << "]\n";
        std::cout << "\t\tt = [" << result.poseB[4] << ", " << result.poseB[5] << ", "
                  << result.poseB[6] << "]\n";

        // Display relative pose
        Pose7 relPose = computeRelativePose(result.poseA, result.poseB);
        Quaternion qRel(relPose[3], relPose[0], relPose[1], relPose[2]);
        Eigen::AngleAxisd aaRel(qRel);
        std::cout << "\n\tRelative pose (A * B^-1):\n";
        std::cout << "\t\tRotation: " << (aaRel.angle() * 180.0 / M_PI)
                  << " deg around [" << aaRel.axis().transpose() << "]\n";
        std::cout << "\t\tTranslation: [" << relPose[4] << ", " << relPose[5] << ", "
                  << relPose[6] << "]\n";
    }

    return 0;
}
