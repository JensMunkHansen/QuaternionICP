// Standard C++ headers
#include <iomanip>
#include <iostream>

// Internal headers
#include <ICP/CommonOptions.h>
#include <ICP/EigenUtils.h>
#include <ICP/Grid.h>
#include <ICP/GridLoader.h>
#include <ICP/ICPCeresTwoPose.h>
#include <ICP/ICPParams.h>
#include <ICP/MultiViewICP.h>
#include <ICP/SE3.h>
#include <ICP/TestGridUtils.h>

using namespace ICP;

int main(int argc, char** argv)
{
    CommonOptions opts;
    if (!parseArgs(argc, argv, opts, "MultiICP - Multi-view ray-projection ICP"))
    {
        return 1;
    }

    // Check which mode we're in
    if (!opts.gridFolder.empty())
    {
        // ========================================
        // Multi-grid mode: load from folder
        // ========================================
        std::cout << "Loading grids from: " << opts.gridFolder << "\n";

        std::vector<Grid> grids = loadGrids(opts.gridFolder, opts.gridIndices);

        if (grids.size() < 2)
        {
            std::cerr << "Error: Need at least 2 grids for multi-view ICP\n";
            return 1;
        }

        std::cout << "Loaded " << grids.size() << " grids\n";

        // Initialize poses from grid files or identity
        std::vector<Pose7> initialPoses(grids.size());
        for (size_t i = 0; i < grids.size(); i++)
        {
            if (opts.useGridPoses)
            {
                initialPoses[i] = isometryToPose7(grids[i].pose);
            }
            else
            {
                initialPoses[i] = identityPose();
            }
        }

        if (opts.verbose && opts.useGridPoses)
        {
            std::cout << "\nInitial poses from grid files:\n";
            for (size_t i = 0; i < grids.size(); i++)
            {
                std::cout << "  Grid " << i << ": t=[" << initialPoses[i][4] << ", "
                          << initialPoses[i][5] << ", " << initialPoses[i][6] << "]\n";
            }
        }

        // Set up multi-view parameters
        MultiViewICPParams params;
        params.rayDir = Vector3(0.0, 0.0, -1.0);
        params.maxDistance = 100.0f;
        params.minMatch = 50;
        params.weighting.enable_weight = opts.enableIncidenceWeight;
        params.weighting.enable_gate = opts.enableGrazingGate;
        params.weighting.tau = opts.incidenceTau;
        params.maxOuterIterations = opts.outerIterations;
        params.convergenceTol = opts.rmsTol;
        params.ceresOptions = commonOptionsToCeresOptions(opts);
        params.fixFirstPose = opts.fixFirstPose;
        params.verbose = opts.verbose;

        // Display configuration
        std::cout << "\n=== Multi-View ICP Configuration ===\n";
        std::cout << "Grids: " << grids.size() << "\n";
        std::cout << "Max outer iterations: " << params.maxOuterIterations << "\n";
        std::cout << "Convergence tolerance: " << params.convergenceTol << "\n";
        std::cout << "Min correspondences per edge: " << params.minMatch << "\n";
        std::cout << "First pose fixed: " << (params.fixFirstPose ? "yes" : "no") << "\n";
        std::cout << "Linear solver: ";
        switch (params.ceresOptions.linearSolverType)
        {
            case ceres::DENSE_QR: std::cout << "DENSE_QR"; break;
            case ceres::DENSE_SCHUR: std::cout << "DENSE_SCHUR"; break;
            case ceres::SPARSE_SCHUR: std::cout << "SPARSE_SCHUR"; break;
            case ceres::ITERATIVE_SCHUR: std::cout << "ITERATIVE_SCHUR"; break;
            default: std::cout << "other"; break;
        }
        std::cout << "\n";

        // Run multi-view ICP
        std::cout << "\nRunning Multi-View ICP...\n";
        auto result = runMultiViewICP(grids, initialPoses, params);

        // Display results
        std::cout << "\n=== Multi-View ICP Results ===\n";
        std::cout << "Converged: " << (result.converged ? "yes" : "no") << "\n";
        std::cout << "Outer iterations: " << result.outerIterations << "\n";
        std::cout << "Total inner iterations: " << result.totalInnerIterations << "\n";
        std::cout << "Final RMS: " << std::scientific << std::setprecision(6) << result.rms << "\n";

        if (opts.verbose)
        {
            std::cout << "\nFinal poses:\n";
            for (size_t i = 0; i < result.poses.size(); i++)
            {
                Quaternion q(result.poses[i][3], result.poses[i][0],
                             result.poses[i][1], result.poses[i][2]);
                Eigen::AngleAxisd aa(q);
                std::cout << "  Grid " << i << ": rot=" << std::fixed << std::setprecision(2)
                          << (aa.angle() * 180.0 / M_PI) << " deg, t=["
                          << result.poses[i][4] << ", " << result.poses[i][5] << ", "
                          << result.poses[i][6] << "]\n";
            }
        }
    }
    else if (opts.useTestGrid || (!opts.sourceFile.empty() && !opts.targetFile.empty()))
    {
        // ========================================
        // Two-grid mode: test or source/target
        // ========================================
        Grid gridA, gridB;

        if (opts.useTestGrid)
        {
            auto grids = createTestGrids(opts, opts.verbose);
            gridA = grids.first;
            gridB = grids.second;
        }
        else
        {
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
            poseA = isometryToPose7(gridA.pose);
            poseB = isometryToPose7(gridB.pose);

            if (opts.verbose)
            {
                std::cout << "\nUsing grid poses:\n";
                std::cout << "  Pose A: t=[" << poseA[4] << ", " << poseA[5] << ", " << poseA[6] << "]\n";
                std::cout << "  Pose B: t=[" << poseB[4] << ", " << poseB[5] << ", " << poseB[6] << "]\n";
            }
        }
        else
        {
            poseA = identityPose();
            poseB = identityPose();
        }

        // Apply perturbations if requested
        if (opts.rotationNoise > 0.0f || opts.translationNoise > 0.0f)
        {
            PerturbationRNGs rngs;
            if (opts.rotationNoise > 0.0f)
                perturbPoseRotation(poseB, opts.rotationNoise, rngs.rotation);
            if (opts.translationNoise > 0.0f)
                perturbPoseTranslation(poseB, opts.translationNoise, rngs.translation);
        }

        // Set up parameters
        CeresICPOptions ceresOpts = commonOptionsToCeresOptions(opts);
        OuterParams outerParams = commonOptionsToOuterParams(opts);

        GeometryWeighting weighting;
        weighting.enable_weight = opts.enableIncidenceWeight;
        weighting.enable_gate = opts.enableGrazingGate;
        weighting.tau = opts.incidenceTau;

        Vector3 rayDir(0.0, 0.0, -1.0);

        // Display configuration
        std::cout << "\n=== Two-Pose ICP Configuration ===\n";
        std::cout << "Max outer iterations: " << outerParams.maxIterations << "\n";
        std::cout << "Convergence tolerance: " << outerParams.convergenceTol << "\n";

        // Run two-pose solver
        std::cout << "\nRunning Two-Pose Ceres ICP";
        if (opts.fixFirstPose) std::cout << " (first pose fixed)";
        std::cout << "...\n";

        auto result = solveICPCeresTwoPose<RayJacobianSimplified>(
            gridA, gridB, poseA, poseB, rayDir, weighting, ceresOpts, outerParams,
            opts.fixFirstPose);

        // Display results
        std::cout << "\n=== Two-Pose ICP Results ===\n";
        std::cout << "Converged: " << (result.converged ? "yes" : "no") << "\n";
        std::cout << "Outer iterations: " << result.outer_iterations << "\n";
        std::cout << "Total inner iterations: " << result.total_inner_iterations << "\n";
        std::cout << "Final RMS: " << std::scientific << std::setprecision(6) << result.rms << "\n";

        if (opts.verbose)
        {
            Pose7 relPose = computeRelativePose(result.poseA, result.poseB);
            Quaternion qRel(relPose[3], relPose[0], relPose[1], relPose[2]);
            Eigen::AngleAxisd aaRel(qRel);
            std::cout << "\nRelative pose: " << (aaRel.angle() * 180.0 / M_PI)
                      << " deg, t=[" << relPose[4] << ", " << relPose[5] << ", " << relPose[6] << "]\n";
        }
    }
    else
    {
        std::cerr << "Error: Specify --grid-folder, --test, or both --source and --target\n";
        return 1;
    }

    return 0;
}
