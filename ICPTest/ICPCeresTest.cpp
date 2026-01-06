// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

// Catch2 headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Test utilities
#include "TestUtils.h"

using namespace ICP;
using namespace TestUtils;
using Catch::Matchers::WithinAbs;

TEST_CASE("Custom manifold vs Sophus manifold", "[icp][inner][ceres][manifold]")
{
    SolverTestFixture fix;

    // Create initial pose with both rotation and translation offset
    Pose7 initialPose = SolverTestFixture::createPose({ 0.05, 0.02, -0.03 }, 0.05, 0.02, 0.08);
    auto corrs = fix.computeCorrs(initialPose);

    WARN("Forward correspondences: " << corrs.forward.size());
    WARN("Reverse correspondences: " << corrs.reverse.size());

    // Solve with Sophus manifold
    auto sophusParams = Presets::ceresGN(20, false);
    sophusParams.useCustomManifold = false;

    auto sophusResult = solveInnerCeres<RayJacobianSimplified>(
        corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, sophusParams);

    WARN("Sophus manifold: iterations=" << sophusResult.iterations
                                         << ", rms=" << std::scientific << sophusResult.rms);

    // Solve with our custom manifold (scale=1.0, should be equivalent)
    auto customParams = Presets::ceresGN(20, false);
    customParams.useCustomManifold = true;
    customParams.rotationScale = 1.0;

    auto customResult = solveInnerCeres<RayJacobianSimplified>(
        corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, customParams);

    WARN("Custom manifold (scale=1.0): iterations=" << customResult.iterations
                                                     << ", rms=" << std::scientific << customResult.rms);

    // Compare final poses
    double quat_diff = std::min(
        (customResult.pose.head<4>() - sophusResult.pose.head<4>()).norm(),
        (customResult.pose.head<4>() + sophusResult.pose.head<4>()).norm());
    double trans_diff = (customResult.pose.tail<3>() - sophusResult.pose.tail<3>()).norm();

    WARN("Custom vs Sophus: quat_diff=" << quat_diff << ", trans_diff=" << trans_diff);

    CHECK(quat_diff < 1e-6);
    CHECK(trans_diff < 1e-6);

    // RMS should match
    CHECK(std::abs(customResult.rms - sophusResult.rms) < 1e-10);
}

TEST_CASE("rotationScale invariance with Ceres", "[icp][inner][ceres][scaling]")
{
    SolverTestFixture fix;

    // Create initial pose with both rotation and translation offset
    Pose7 initialPose = SolverTestFixture::createPose({ 0.05, 0.02, -0.03 }, 0.05, 0.02, 0.08);
    auto corrs = fix.computeCorrs(initialPose);

    WARN("Forward correspondences: " << corrs.forward.size());
    WARN("Reverse correspondences: " << corrs.reverse.size());

    // Test Ceres with different rotation scales using our SE3ScaledManifold
    std::vector<double> scales = { 1.0, 0.001, 0.01, 0.1, 0.5, 2.0, 10.0, 100.0, 1000.0 };
    std::vector<Pose7> ceresResults;
    std::vector<int> ceresIterations;

    for (double scale : scales)
    {
        auto params = Presets::ceresGN(20, false);
        params.rotationScale = scale;

        auto result = solveInnerCeres<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, params);

        ceresResults.push_back(result.pose);
        ceresIterations.push_back(result.iterations);

        WARN("Ceres scale " << scale << ": iterations=" << result.iterations
                            << ", rms=" << std::scientific << result.rms);
    }

    // Verify all Ceres results converge to the same pose (invariance)
    Pose7 ceresRef = ceresResults[0];
    for (size_t i = 1; i < ceresResults.size(); ++i)
    {
        double quat_diff = std::min(
            (ceresResults[i].head<4>() - ceresRef.head<4>()).norm(),
            (ceresResults[i].head<4>() + ceresRef.head<4>()).norm());
        CHECK(quat_diff < 1e-5);

        double trans_diff = (ceresResults[i].tail<3>() - ceresRef.tail<3>()).norm();
        CHECK(trans_diff < 1e-5);

        WARN("Scale " << scales[i] << " vs 1.0: quat_diff=" << quat_diff
                      << ", trans_diff=" << trans_diff);
    }

    // Compare with hand-rolled solver at scale=1.0
    auto handRolledParams = Presets::gaussNewton(20, false);
    handRolledParams.rotationScale = 1.0;

    auto handRolledResult = solveInner<RayJacobianSimplified>(
        corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, handRolledParams);

    WARN("\nHand-rolled (scale=1.0): iterations=" << handRolledResult.iterations
                                                   << ", rms=" << handRolledResult.rms);

    // Compare Ceres vs hand-rolled final pose
    double quat_diff = std::min(
        (ceresRef.head<4>() - handRolledResult.pose.head<4>()).norm(),
        (ceresRef.head<4>() + handRolledResult.pose.head<4>()).norm());
    double trans_diff = (ceresRef.tail<3>() - handRolledResult.pose.tail<3>()).norm();

    WARN("Ceres vs Hand-rolled: quat_diff=" << quat_diff << ", trans_diff=" << trans_diff);

    CHECK(quat_diff < 1e-5);
    CHECK(trans_diff < 1e-5);

    WARN("\n=== Ceres iteration summary ===");
    for (size_t i = 0; i < scales.size(); ++i)
    {
        WARN("Scale " << scales[i] << ": " << ceresIterations[i] << " iterations");
    }
}

TEST_CASE("solveInnerCeres with fixed correspondences", "[icp][inner][ceres]")
{
    SolverTestFixture fix;
    auto ceresOpts = Presets::ceresGN(12, false);

    SECTION("Identity pose")
    {
        Pose7 pose = identityPose();
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        auto result = solveInnerCeres<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, ceresOpts);

        WARN("Ceres RMS: " << result.rms);
        WARN("Ceres Iterations: " << result.iterations);

        CHECK(result.rms < 1e-8);
    }

    SECTION("Translation offset tx=0.05")
    {
        Pose7 pose = translationPose(0.05, 0.0, 0.0);
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        auto result = solveInnerCeres<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, ceresOpts);

        WARN("Ceres RMS: " << result.rms);
        WARN("Ceres Iterations: " << result.iterations);
    }

    SECTION("Rotation offset")
    {
        Pose7 pose = SolverTestFixture::createPose({ 0.05, 0.02, -0.03 });
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        auto result = solveInnerCeres<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, ceresOpts);

        WARN("Ceres RMS: " << result.rms);
        WARN("Ceres Iterations: " << result.iterations);
    }
}
