// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

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

        // grid.pose is already initialized from initialPose in loadFromExr
        // Reset to identity if not using grid poses
        if (!opts.useGridPoses)
        {
            for (auto& grid : grids)
                grid.pose = identityPose();
        }

        if (opts.verbose && opts.useGridPoses)
        {
            std::cout << "\nInitial poses from grid files:\n";
            for (size_t i = 0; i < grids.size(); i++)
            {
                std::cout << "  Grid " << i << ": t=[" << grids[i].pose[4] << ", "
                          << grids[i].pose[5] << ", " << grids[i].pose[6] << "]\n";
            }
        }

        // Save ground truth before perturbation (for results comparison)
        std::vector<Pose7> groundTruthPoses(grids.size());
        for (size_t i = 0; i < grids.size(); i++)
            groundTruthPoses[i] = grids[i].pose;

        // Perturb grid.pose directly (except first which is reference)
        bool hasPerturbation = (opts.rotationNoise > 0.0f || opts.translationNoise > 0.0f);
        if (hasPerturbation)
        {
            std::cout << "\nPerturbing poses (rot=" << opts.rotationNoise
                      << " deg, trans=" << opts.translationNoise << ")...\n";

            for (size_t i = 1; i < grids.size(); i++)
            {
                PerturbationRNGs rngs(static_cast<unsigned int>(i),
                                      static_cast<unsigned int>(i + 1000),
                                      static_cast<unsigned int>(i + 2000));
                if (opts.rotationNoise > 0.0f)
                    perturbPoseRotation(grids[i].pose, opts.rotationNoise, rngs.rotation);
                if (opts.translationNoise > 0.0f)
                    perturbPoseTranslation(grids[i].pose, opts.translationNoise, rngs.translation);
            }
        }

        // Save initial (possibly perturbed) poses for results table
        std::vector<Pose7> initialPoses(grids.size());
        for (size_t i = 0; i < grids.size(); i++)
            initialPoses[i] = grids[i].pose;

        // Set up parameters using canonical structs
        SessionParams session = commonOptionsToSessionParams(opts);
        OuterParams outer = commonOptionsToOuterParams(opts);
        InnerParams inner = commonOptionsToInnerParams(opts);

        // Multi-view uses ITERATIVE_SCHUR by default (preconditioner comes from opts)
        if (opts.linearSolver == CommonOptions::LinearSolver::DenseQR)
        {
            inner.linearSolverType = LinearSolverType::IterativeSchur;
        }

        // Display configuration
        CommonOptions displayOpts = opts;
        displayOpts.backend = CommonOptions::Backend::Ceres7;
        if (opts.linearSolver == CommonOptions::LinearSolver::DenseQR)
            displayOpts.linearSolver = CommonOptions::LinearSolver::IterativeSchur;
        printCommonConfig(displayOpts);
        std::cout << "\nMulti-view:\n";
        std::cout << "  Grids: " << grids.size() << "\n";
        std::cout << "  Min correspondences per edge: " << outer.minMatch << "\n";
        if (outer.maxCorrespondences > 0)
            std::cout << "  Max correspondences per edge: " << outer.maxCorrespondences << "\n";
        if (outer.maxNeighbors > 0)
            std::cout << "  Max neighbors per grid: " << outer.maxNeighbors << "\n";
        std::cout << "  Use grid poses: " << (session.useGridPoses ? "yes" : "no") << "\n";
        std::cout << "  First pose fixed: " << (session.fixFirstPose ? "yes" : "no") << "\n";

        // Run multi-view ICP (optimizes grids[i].pose in place)
        std::cout << "\nRunning Multi-View ICP...\n";
        auto result = runMultiViewICP(grids, session, outer, inner);

        // Display results
        std::cout << "\n=== Multi-View ICP Results ===\n";
        std::cout << "Converged: " << (result.converged ? "yes" : "no") << "\n";
        std::cout << "Outer iterations: " << result.outerIterations << "\n";
        std::cout << "Total inner iterations: " << result.totalInnerIterations << "\n";
        std::cout << "Final RMS: " << std::scientific << std::setprecision(6) << result.rms << "\n";

        if (opts.verbose)
        {
            std::cout << "\nFinal poses:\n";
            for (size_t i = 0; i < grids.size(); i++)
            {
                Quaternion q(grids[i].pose[3], grids[i].pose[0],
                             grids[i].pose[1], grids[i].pose[2]);
                Eigen::AngleAxisd aa(q);
                Eigen::Vector3d axis = aa.axis();
                std::cout << "  Grid " << i << ": rot=" << std::fixed << std::setprecision(2)
                          << (aa.angle() * 180.0 / M_PI) << " deg around ["
                          << axis[0] << ", " << axis[1] << ", " << axis[2] << "], t=["
                          << grids[i].pose[4] << ", " << grids[i].pose[5] << ", "
                          << grids[i].pose[6] << "]\n";
            }
        }

        // Show results table if perturbation was applied
        if (hasPerturbation)
        {
            // Extract final poses from grids for results table
            std::vector<Pose7> finalPoses(grids.size());
            for (size_t i = 0; i < grids.size(); i++)
                finalPoses[i] = grids[i].pose;
            printResultsTable(finalPoses, initialPoses, groundTruthPoses);
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
            poseA = isometryToPose7(gridA.initialPose);
            poseB = isometryToPose7(gridB.initialPose);

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
        InnerParams innerParams = commonOptionsToInnerParams(opts);
        OuterParams outerParams = commonOptionsToOuterParams(opts);

        // Display configuration
        std::cout << "\n=== Two-Pose ICP Configuration ===\n";
        std::cout << "Max outer iterations: " << outerParams.maxIterations << "\n";
        std::cout << "Convergence tolerance: " << outerParams.convergenceTol << "\n";

        // Run two-pose solver
        std::cout << "\nRunning Two-Pose Ceres ICP";
        if (opts.fixFirstPose) std::cout << " (first pose fixed)";
        std::cout << "...\n";

        auto result = solveICPCeresTwoPose<RayJacobianSimplified>(
            gridA, gridB, poseA, poseB, outerParams.rayDir, outerParams.weighting,
            innerParams, outerParams, opts.fixFirstPose);

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
