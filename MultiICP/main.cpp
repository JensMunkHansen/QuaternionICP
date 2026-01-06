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
#include <ICP/IntersectionBackend.h>
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

    // Apply intersection backend selection
    switch (opts.intersectionBackend)
    {
        case CommonOptions::IntersectionBackend::GridSearch:
            if (!isBackendAvailable(IntersectionBackendType::GridSearch))
            {
                std::cerr << "Error: GridSearch backend not available (compile with -DUSE_GRIDSEARCH=ON)\n";
                return 1;
            }
            setDefaultIntersectionBackend(IntersectionBackendType::GridSearch);
            break;
        case CommonOptions::IntersectionBackend::Embree:
            if (!isBackendAvailable(IntersectionBackendType::Embree))
            {
                std::cerr << "Error: Embree backend not available (compile with -DUSE_EMBREE=ON)\n";
                return 1;
            }
            setDefaultIntersectionBackend(IntersectionBackendType::Embree);
            break;
        default:
            break;  // Auto: use compile-time default
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
                std::cout << "  Grid " << i << ": " << poseToString(grids[i].pose) << "\n";
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
        // Default: use solveICPCeresTwoPose (validated against Python reference)
        // With --multi-view: use runMultiViewICP for testing/comparison
        // ========================================
        Grid source, target;

        if (opts.useTestGrid)
        {
            auto testGrids = createTestGrids(opts, opts.verbose);
            source = std::move(testGrids.first);
            target = std::move(testGrids.second);
        }
        else
        {
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

        // Initialize poses from grid initialPose or identity
        Pose7 poseA = opts.useGridPoses ? isometryToPose7(source.initialPose) : identityPose();
        Pose7 poseB = opts.useGridPoses ? isometryToPose7(target.initialPose) : identityPose();

        if (opts.verbose && opts.useGridPoses)
        {
            std::cout << "\nUsing grid poses:\n";
            std::cout << "  Grid 0: t=[" << poseA[4] << ", " << poseA[5] << ", " << poseA[6] << "]\n";
            std::cout << "  Grid 1: t=[" << poseB[4] << ", " << poseB[5] << ", " << poseB[6] << "]\n";
        }

        // Save ground truth before perturbation
        Pose7 groundTruthA = poseA;
        Pose7 groundTruthB = poseB;

        // Apply perturbations to match SingleICP convention:
        // In SingleICP, perturbation is applied to the relative pose srcToTgt
        // In two-pose with poseA fixed at identity:
        //   srcToTgt = poseB^{-1} * poseA = poseB^{-1}
        // So if we want srcToTgt = perturbed, we need poseB = perturbed^{-1}
        bool hasPerturbation = (opts.rotationNoise > 0.0f || opts.translationNoise > 0.0f);
        if (hasPerturbation)
        {
            // Create perturbed relative pose (like SingleICP's convention)
            // Use default seeds (same as SingleICP) for reproducibility
            Pose7 perturbedRelPose = identityPose();
            PerturbationRNGs rngs;  // Default seeds: 42, 43, 44
            if (opts.rotationNoise > 0.0f)
                perturbPoseRotation(perturbedRelPose, opts.rotationNoise, rngs.rotation);
            if (opts.translationNoise > 0.0f)
                perturbPoseTranslation(perturbedRelPose, opts.translationNoise, rngs.translation);

            // Set poseB = perturbedRelPose^{-1} so that srcToTgt = poseB^{-1} = perturbedRelPose
            Eigen::Isometry3d relIso = pose7ToIsometry(perturbedRelPose);
            poseB = isometryToPose7(relIso.inverse());
        }

        // Save initial poses for results table
        Pose7 initialA = poseA;
        Pose7 initialB = poseB;

        // Set up parameters
        OuterParams outer = commonOptionsToOuterParams(opts);
        InnerParams inner = commonOptionsToInnerParams(opts);

        GeometryWeighting weighting;
        weighting.enable_weight = opts.enableIncidenceWeight;
        weighting.enable_gate = opts.enableGrazingGate;
        weighting.tau = opts.incidenceTau;

        Vector3 rayDir(0.0, 0.0, -1.0);

        // Display configuration
        CommonOptions displayOpts = opts;
        displayOpts.backend = CommonOptions::Backend::Ceres7;
        printCommonConfig(displayOpts);
        std::cout << "\nTwo-grid mode:\n";
        std::cout << "  Solver: " << (opts.useMultiViewSolver ? "MultiViewICP" : "CeresTwoPose") << "\n";
        std::cout << "  First pose fixed: " << (opts.fixFirstPose ? "yes" : "no") << "\n";

        // Run ICP
        bool converged = false;
        int outerIterations = 0;
        int totalInnerIterations = 0;
        double finalRms = 0.0;
        Pose7 finalA, finalB;

        if (opts.useMultiViewSolver)
        {
            // Use runMultiViewICP for testing/comparison
            std::cout << "\nRunning Multi-View ICP (two grids)...\n";

            std::vector<Grid> grids(2);
            grids[0] = std::move(source);
            grids[1] = std::move(target);
            grids[0].pose = poseA;
            grids[1].pose = poseB;

            SessionParams session = commonOptionsToSessionParams(opts);
            outer.minMatch = 1;
            outer.maxCorrespondences = 0;

            auto result = runMultiViewICP(grids, session, outer, inner);

            converged = result.converged;
            outerIterations = result.outerIterations;
            totalInnerIterations = result.totalInnerIterations;
            finalRms = result.rms;
            finalA = grids[0].pose;
            finalB = grids[1].pose;
        }
        else
        {
            // Use validated solveICPCeresTwoPose
            std::cout << "\nRunning Ceres Two-Pose ICP...\n";

            auto result = solveICPCeresTwoPose(
                source, target, poseA, poseB, rayDir, weighting, inner, outer, opts.fixFirstPose);

            converged = result.converged;
            outerIterations = result.outer_iterations;
            totalInnerIterations = result.total_inner_iterations;
            finalRms = result.rms;
            finalA = result.poseA;
            finalB = result.poseB;
        }

        // Display results
        std::cout << "\n=== ICP Results ===\n";
        std::cout << "Converged: " << (converged ? "yes" : "no") << "\n";
        std::cout << "Outer iterations: " << outerIterations << "\n";
        std::cout << "Total inner iterations: " << totalInnerIterations << "\n";
        std::cout << "Final RMS: " << std::scientific << std::setprecision(6) << finalRms << "\n";

        if (opts.verbose)
        {
            std::cout << "\nFinal poses:\n";
            std::cout << "  Grid 0: " << poseToString(finalA) << "\n";
            std::cout << "  Grid 1: " << poseToString(finalB) << "\n";
        }

        // Show results table if perturbation was applied
        if (hasPerturbation)
        {
            std::vector<Pose7> finalPoses = {finalA, finalB};
            std::vector<Pose7> initialPoses = {initialA, initialB};
            std::vector<Pose7> groundTruthPoses = {groundTruthA, groundTruthB};
            printResultsTable(finalPoses, initialPoses, groundTruthPoses);
        }
    }
    else
    {
        std::cerr << "Error: Specify --grid-folder, --test, or both --source and --target\n";
        return 1;
    }

    return 0;
}
