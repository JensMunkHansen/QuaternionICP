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

TEST_CASE("solveInner with fixed correspondences", "[icp][inner][python]")
{
    SolverTestFixture fix;
    auto params = Presets::gaussNewton(12, false);

    SECTION("Identity pose")
    {
        Pose7 pose = identityPose();
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        auto result = solveInner<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);

        WARN("RMS: " << result.rms);
        WARN("Iterations: " << result.iterations);

        CHECK(result.rms < 1e-8);
    }

    SECTION("Translation offset tx=0.05")
    {
        Pose7 pose = translationPose(0.05, 0.0, 0.0);
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        auto result = solveInner<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);

        WARN("RMS: " << result.rms);
        WARN("Iterations: " << result.iterations);
    }

    SECTION("Rotation offset")
    {
        Pose7 pose = SolverTestFixture::createPose({ 0.05, 0.02, -0.03 });
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        auto result = solveInner<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);

        WARN("RMS: " << result.rms);
        WARN("Iterations: " << result.iterations);
    }
}

TEST_CASE("solveICP outer loop", "[icp][outer][python]")
{
    SolverTestFixture fix;

    auto innerParams = Presets::gaussNewton(12, true);
    OuterParams outerParams;
    outerParams.maxIterations = 6;
    outerParams.convergenceTol = 0.0;
    outerParams.maxDist = 100.0f;
    outerParams.verbose = true;

    SECTION("Small perturbation")
    {
        WARN("\n=== Test: Small perturbation ===");

        Pose7 initialPose = SolverTestFixture::createPose({ 0.01, 0.005, -0.008 }, 0.01, -0.005, 0.02);

        auto result = solveICP<RayJacobianSimplified>(
            fix.source, fix.target, initialPose, fix.rayDir, fix.weighting, innerParams, outerParams);

        WARN("Outer iterations: " << result.outer_iterations);
        WARN("Total inner iterations: " << result.total_inner_iterations);
        WARN("Final RMS: " << result.rms);

        CHECK(std::abs(result.pose[0]) < 1e-6);
        CHECK(std::abs(result.pose[1]) < 1e-6);
        CHECK(std::abs(result.pose[2]) < 1e-6);
        CHECK(std::abs(result.pose[3] - 1.0) < 1e-6);
        CHECK(std::abs(result.pose[4]) < 1e-6);
        CHECK(std::abs(result.pose[5]) < 1e-6);
        CHECK(std::abs(result.pose[6]) < 1e-6);
    }

    SECTION("Larger perturbation")
    {
        WARN("\n=== Test: Larger perturbation ===");

        Pose7 initialPose = SolverTestFixture::createPose({ 0.05, 0.02, -0.03 }, 0.05, 0.02, 0.08);

        auto result = solveICP<RayJacobianSimplified>(
            fix.source, fix.target, initialPose, fix.rayDir, fix.weighting, innerParams, outerParams);

        WARN("Outer iterations: " << result.outer_iterations);
        WARN("Total inner iterations: " << result.total_inner_iterations);
        WARN("Final RMS: " << result.rms);

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
        WARN("\n=== Test: Translation only ===");

        Pose7 initialPose = translationPose(0.03, -0.02, 0.05);

        auto result = solveICP<RayJacobianSimplified>(
            fix.source, fix.target, initialPose, fix.rayDir, fix.weighting, innerParams, outerParams);

        WARN("Outer iterations: " << result.outer_iterations);
        WARN("Total inner iterations: " << result.total_inner_iterations);
        WARN("Final RMS: " << result.rms);

        CHECK(std::abs(result.pose[0]) < 1e-6);
        CHECK(std::abs(result.pose[1]) < 1e-6);
        CHECK(std::abs(result.pose[2]) < 1e-6);
        CHECK(std::abs(result.pose[3] - 1.0) < 1e-6);
        CHECK(std::abs(result.pose[4]) < 1e-6);
        CHECK(std::abs(result.pose[5]) < 1e-6);
        CHECK(std::abs(result.pose[6]) < 1e-6);
    }
}
