#include <ICP/ICPSimple.h>
#include <ICP/GridFactory.h>
#include <ICP/EigenUtils.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>

using namespace ICP;
using Catch::Matchers::WithinAbs;

/**
 * Create InnerParams matching Python reference implementation.
 * Pure Gauss-Newton, no line search, no damping.
 * Uses factory method to ensure all parameters are explicit.
 */
InnerParams pythonReferenceParams()
{
    // All parameters explicit via factory method
    return InnerParams::gaussNewton(
        /*maxIterations=*/12,
        /*translationThreshold=*/1e-9,  // Very small to match Python (uses stepTol only)
        /*rotationThreshold=*/1e-9,     // Very small to match Python
        /*damping=*/0.0,                // Pure Gauss-Newton
        /*lineSearchEnabled=*/false,    // No line search
        /*verbose=*/false
    );
}

TEST_CASE("solveInner with fixed correspondences", "[icp][inner][python]")
{
    // Match Python: nx=45, ny=45, spacing=2.0/44
    float spacing = 2.0f / 44.0f;
    Grid source = createHeightfieldGrid(45, 45, spacing, spacing);
    Grid target = source;

    // Match Python: no weighting
    GeometryWeighting weighting;
    weighting.enable_weight = false;
    weighting.enable_gate = false;

    // Use explicit Python-matching parameters
    InnerParams pyParams = pythonReferenceParams();

    auto v0 = source.getVertex(0);
    auto vLast = source.getVertex(45*45 - 1);
    float diag = std::sqrt(std::pow(vLast.x() - v0.x(), 2) + std::pow(vLast.y() - v0.y(), 2));
    WARN("C++ grid: min=(" << v0.x() << ", " << v0.y() << "), max=(" << vLast.x() << ", " << vLast.y() << ")");
    WARN("C++ XY diagonal: " << diag);
    WARN("Total vertices: " << source.nRows() * source.nCols());
    WARN("Triangle vertex indices: " << source.getTriangleVertexIndices().size());

    Vector3 rayDir(0, 0, -1);
    float maxDist = 100.0f;

    SECTION("Identity pose")
    {
        Pose7 pose = identityPose();
        Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
        auto corrs = computeBidirectionalCorrs(source, target, rayDir.cast<float>(), srcToTgt, maxDist);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        // Check which source vertices are missing forward correspondences
        std::vector<bool> hasForwardHit(source.nRows() * source.nCols(), false);
        for (const auto& corr : corrs.forward) {
            hasForwardHit[corr.srcVertexIdx] = true;
        }
        int missingCount = 0;
        std::vector<int> missingIndices;
        for (int i = 0; i < hasForwardHit.size(); ++i) {
            if (!hasForwardHit[i]) {
                missingCount++;
                if (missingIndices.size() < 10) missingIndices.push_back(i);
            }
        }
        WARN("Missing forward hits: " << missingCount << " out of " << hasForwardHit.size());
        if (!missingIndices.empty()) {
            std::cout << "\tFirst missing grid positions (row,col): ";
            for (int idx : missingIndices) {
                int row = idx / source.nCols();
                int col = idx % source.nCols();
                std::cout << "(" << row << "," << col << ") ";
            }
            std::cout << "\n";
        }

        auto result = solveInner<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, rayDir, weighting, pyParams);

        WARN("RMS: " << result.rms);
        WARN("Iterations: " << result.iterations);
        WARN("Valid count: " << result.valid_count);

        CHECK(result.rms < 1e-8);
    }

    SECTION("Translation offset tx=0.05")
    {
        Pose7 pose = translationPose(0.05, 0.0, 0.0);
        Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
        srcToTgt.translation() = Eigen::Vector3d(0.05, 0, 0);
        auto corrs = computeBidirectionalCorrs(source, target, rayDir.cast<float>(), srcToTgt, maxDist);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        // Run single iterations to print per-iteration RMS
        // Use Python reference params but with 1 iteration per call
        InnerParams singleIter = pythonReferenceParams();
        singleIter.maxIterations = 1;
        singleIter.translationThreshold = 0;  // Don't stop early
        singleIter.rotationThreshold = 0;

        for (int i = 0; i < 12; ++i)
        {
            auto result = solveInner<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, rayDir, weighting, singleIter);
            WARN("inner " << i << ": rms=" << result.rms << ", valid=" << result.valid_count);
            pose = result.pose;
            if (result.rms < 1e-9) break;
        }
    }

    SECTION("Rotation offset")
    {
        Vector3 axisAngle(0.05, 0.02, -0.03);
        Pose7 pose = rotationPose(axisAngle);

        Quaternion q(pose[3], pose[0], pose[1], pose[2]);
        Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
        srcToTgt.linear() = q.toRotationMatrix();
        auto corrs = computeBidirectionalCorrs(source, target, rayDir.cast<float>(), srcToTgt, maxDist);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        // Run single iterations to print per-iteration RMS
        // Use Python reference params but with 1 iteration per call
        InnerParams singleIter = pythonReferenceParams();
        singleIter.maxIterations = 1;
        singleIter.translationThreshold = 0;  // Don't stop early
        singleIter.rotationThreshold = 0;

        for (int i = 0; i < 12; ++i)
        {
            auto result = solveInner<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, rayDir, weighting, singleIter);
            WARN("inner " << i << ": rms=" << result.rms << ", valid=" << result.valid_count);
            pose = result.pose;
            if (result.rms < 1e-9) break;
        }
    }
}

TEST_CASE("solveICP outer loop", "[icp][outer][python]")
{
    // Match Python: nx=45, ny=45, spacing=2.0/44
    float spacing = 2.0f / 44.0f;
    Grid source = createHeightfieldGrid(45, 45, spacing, spacing);
    Grid target = source;

    // Match Python: no weighting
    GeometryWeighting weighting;
    weighting.enable_weight = false;
    weighting.enable_gate = false;

    Vector3 rayDir(0, 0, -1);

    // Explicit Python reference parameters for inner loop
    InnerParams innerParams = pythonReferenceParams();
    innerParams.verbose = true;

    // Explicit outer loop parameters
    OuterParams outerParams;
    outerParams.maxIterations = 6;
    outerParams.convergenceTol = 0.0;  // Disable early stopping to run all iterations
    outerParams.maxDist = 100.0f;
    outerParams.verbose = true;

    SECTION("Small perturbation")
    {
        WARN("\n=== Test 1: Small perturbation ===");

        // Match Python: axis_angle = [0.01, 0.005, -0.008], t = [0.01, -0.005, 0.02]
        Vector3 axisAngle(0.01, 0.005, -0.008);
        Pose7 initialPose = rotationPose(axisAngle);
        initialPose[4] = 0.01;   // tx
        initialPose[5] = -0.005; // ty
        initialPose[6] = 0.02;   // tz

        WARN("Initial pose: q=[" << initialPose[0] << ", " << initialPose[1] << ", "
             << initialPose[2] << ", " << initialPose[3] << "], t=["
             << initialPose[4] << ", " << initialPose[5] << ", " << initialPose[6] << "]");

        auto result = solveICP<RayJacobianSimplified>(
            source, target, initialPose, rayDir, weighting, innerParams, outerParams);

        WARN("Final pose: [" << result.pose[0] << ", " << result.pose[1] << ", "
             << result.pose[2] << ", " << result.pose[3] << ", "
             << result.pose[4] << ", " << result.pose[5] << ", " << result.pose[6] << "]");
        WARN("Outer iterations: " << result.outer_iterations);
        WARN("Total inner iterations: " << result.total_inner_iterations);
        WARN("Final RMS: " << result.rms);

        // Should converge close to identity
        CHECK(std::abs(result.pose[0]) < 1e-6);  // qx
        CHECK(std::abs(result.pose[1]) < 1e-6);  // qy
        CHECK(std::abs(result.pose[2]) < 1e-6);  // qz
        CHECK(std::abs(result.pose[3] - 1.0) < 1e-6);  // qw
        CHECK(std::abs(result.pose[4]) < 1e-6);  // tx
        CHECK(std::abs(result.pose[5]) < 1e-6);  // ty
        CHECK(std::abs(result.pose[6]) < 1e-6);  // tz
    }

    SECTION("Larger perturbation")
    {
        WARN("\n=== Test 2: Larger perturbation ===");

        // Match Python: axis_angle = [0.05, 0.02, -0.03], t = [0.05, 0.02, 0.08]
        Vector3 axisAngle(0.05, 0.02, -0.03);
        Pose7 initialPose = rotationPose(axisAngle);
        initialPose[4] = 0.05;  // tx
        initialPose[5] = 0.02;  // ty
        initialPose[6] = 0.08;  // tz

        WARN("Initial pose: q=[" << initialPose[0] << ", " << initialPose[1] << ", "
             << initialPose[2] << ", " << initialPose[3] << "], t=["
             << initialPose[4] << ", " << initialPose[5] << ", " << initialPose[6] << "]");

        auto result = solveICP<RayJacobianSimplified>(
            source, target, initialPose, rayDir, weighting, innerParams, outerParams);

        WARN("Final pose: [" << result.pose[0] << ", " << result.pose[1] << ", "
             << result.pose[2] << ", " << result.pose[3] << ", "
             << result.pose[4] << ", " << result.pose[5] << ", " << result.pose[6] << "]");
        WARN("Outer iterations: " << result.outer_iterations);
        WARN("Total inner iterations: " << result.total_inner_iterations);
        WARN("Final RMS: " << result.rms);

        // Should converge close to identity
        CHECK(std::abs(result.pose[0]) < 1e-6);
        CHECK(std::abs(result.pose[1]) < 1e-6);
        CHECK(std::abs(result.pose[2]) < 1e-6);
        CHECK(std::abs(result.pose[3] - 1.0) < 1e-6);
        CHECK(std::abs(result.pose[4]) < 1e-6);
        CHECK(std::abs(result.pose[5]) < 1e-6);
        CHECK(std::abs(result.pose[6]) < 1e-6);
    }

    SECTION("Translation only")
    {
        WARN("\n=== Test 3: Translation only ===");

        // Match Python: q = [0, 0, 0, 1], t = [0.03, -0.02, 0.05]
        Pose7 initialPose = identityPose();
        initialPose[4] = 0.03;   // tx
        initialPose[5] = -0.02;  // ty
        initialPose[6] = 0.05;   // tz

        WARN("Initial pose: q=[" << initialPose[0] << ", " << initialPose[1] << ", "
             << initialPose[2] << ", " << initialPose[3] << "], t=["
             << initialPose[4] << ", " << initialPose[5] << ", " << initialPose[6] << "]");

        // Check correspondences at converged pose (identity)
        Eigen::Isometry3d identityTransform = Eigen::Isometry3d::Identity();
        auto convergedCorrs = computeBidirectionalCorrs(source, target, rayDir.cast<float>(), identityTransform, outerParams.maxDist);

        // Also check at the initial perturbed pose
        Eigen::Isometry3d initialTransform = pose7ToIsometry(initialPose);
        auto initialCorrs = computeBidirectionalCorrs(source, target, rayDir.cast<float>(), initialTransform, outerParams.maxDist);
        std::cout << "\tInitial transform translation: [" << initialTransform.translation().transpose() << "]\n";
        WARN("At initial perturbed pose: forward=" << initialCorrs.forward.size());

        std::vector<bool> hasHit(source.nRows() * source.nCols(), false);
        for (const auto& corr : convergedCorrs.forward) {
            hasHit[corr.srcVertexIdx] = true;
        }
        int missing = 0;
        std::vector<int> missingIdx;
        for (int i = 0; i < hasHit.size(); ++i) {
            if (!hasHit[i]) {
                missing++;
                if (missingIdx.size() < 20) missingIdx.push_back(i);
            }
        }
        WARN("At identity: forward=" << convergedCorrs.forward.size() << ", missing=" << missing);
        if (!missingIdx.empty()) {
            std::cout << "\tMissing grid positions (row,col): ";
            for (int idx : missingIdx) {
                std::cout << "(" << idx/source.nCols() << "," << idx%source.nCols() << ") ";
            }
            std::cout << "\n";
        }

        auto result = solveICP<RayJacobianSimplified>(
            source, target, initialPose, rayDir, weighting, innerParams, outerParams);

        WARN("Final pose: [" << result.pose[0] << ", " << result.pose[1] << ", "
             << result.pose[2] << ", " << result.pose[3] << ", "
             << result.pose[4] << ", " << result.pose[5] << ", " << result.pose[6] << "]");
        WARN("Outer iterations: " << result.outer_iterations);
        WARN("Total inner iterations: " << result.total_inner_iterations);
        WARN("Final RMS: " << result.rms);

        // Check correspondences at the converged pose
        Eigen::Isometry3d finalTransform = pose7ToIsometry(result.pose);
        auto finalCorrs = computeBidirectionalCorrs(source, target, rayDir.cast<float>(), finalTransform, outerParams.maxDist);
        std::vector<bool> hasFinalHit(source.nRows() * source.nCols(), false);
        for (const auto& corr : finalCorrs.forward) {
            hasFinalHit[corr.srcVertexIdx] = true;
        }
        int finalMissing = 0;
        std::vector<int> finalMissingIdx;
        for (int i = 0; i < hasFinalHit.size(); ++i) {
            if (!hasFinalHit[i]) {
                finalMissing++;
                if (finalMissingIdx.size() < 30) finalMissingIdx.push_back(i);
            }
        }
        WARN("At final converged pose: forward=" << finalCorrs.forward.size() << ", missing=" << finalMissing);
        if (!finalMissingIdx.empty()) {
            std::cout << "\tMissing at final pose (row,col): ";
            for (int idx : finalMissingIdx) {
                std::cout << "(" << idx/source.nCols() << "," << idx%source.nCols() << ") ";
            }
            std::cout << "\n";
        }

        // Should converge close to identity
        CHECK(std::abs(result.pose[0]) < 1e-6);
        CHECK(std::abs(result.pose[1]) < 1e-6);
        CHECK(std::abs(result.pose[2]) < 1e-6);
        CHECK(std::abs(result.pose[3] - 1.0) < 1e-6);
        CHECK(std::abs(result.pose[4]) < 1e-6);
        CHECK(std::abs(result.pose[5]) < 1e-6);
        CHECK(std::abs(result.pose[6]) < 1e-6);
    }
}

