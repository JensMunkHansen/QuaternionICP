// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * @file ICPSimpleLeftTest.cpp
 * @brief Compare left-perturbation (tangent) solver with ambient (right) solver.
 *
 * This test validates that both solvers converge to the same solution,
 * demonstrating equivalence of left and right perturbation approaches.
 */

// Catch2 headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Test utilities
#include "TestUtils.h"

// Left-perturbation solver
#include <ICP/ICPSolverLeft.h>

using namespace ICP;
using namespace TestUtils;
using Catch::Matchers::WithinAbs;

/**
 * Compare inner solver results from ambient (right) and left-perturbation solvers.
 * Both should converge to the same final pose.
 */
TEST_CASE("Left vs Right inner solver: same result", "[icp][left][inner]")
{
    SolverTestFixture fix;
    auto params = Presets::gaussNewton(20, false);

    SECTION("Small rotation")
    {
        Pose7 initialPose = SolverTestFixture::createPose({ 0.02, 0.01, -0.015 });
        auto corrs = fix.computeCorrs(initialPose);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        // Solve with ambient (right-perturbation)
        auto resultRight = solveInner<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, params);

        // Solve with left-perturbation
        auto resultLeft = solveInnerGNLeft<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, params);

        WARN("Right: iters=" << resultRight.iterations << ", rms=" << resultRight.rms);
        WARN("Left:  iters=" << resultLeft.iterations << ", rms=" << resultLeft.rms);

        // Check final poses match
        double quat_diff = std::min(
            (resultLeft.pose.head<4>() - resultRight.pose.head<4>()).norm(),
            (resultLeft.pose.head<4>() + resultRight.pose.head<4>()).norm());
        double trans_diff = (resultLeft.pose.tail<3>() - resultRight.pose.tail<3>()).norm();

        WARN("Quaternion difference: " << quat_diff);
        WARN("Translation difference: " << trans_diff);

        CHECK(quat_diff < 1e-6);
        CHECK(trans_diff < 1e-6);
        CHECK_THAT(resultLeft.rms, WithinAbs(resultRight.rms, 1e-8));
    }

    SECTION("Translation offset")
    {
        Pose7 initialPose = translationPose(0.03, -0.02, 0.01);
        auto corrs = fix.computeCorrs(initialPose);

        auto resultRight = solveInner<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, params);

        auto resultLeft = solveInnerGNLeft<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, params);

        WARN("Right: iters=" << resultRight.iterations << ", rms=" << resultRight.rms);
        WARN("Left:  iters=" << resultLeft.iterations << ", rms=" << resultLeft.rms);

        double quat_diff = std::min(
            (resultLeft.pose.head<4>() - resultRight.pose.head<4>()).norm(),
            (resultLeft.pose.head<4>() + resultRight.pose.head<4>()).norm());
        double trans_diff = (resultLeft.pose.tail<3>() - resultRight.pose.tail<3>()).norm();

        WARN("Quaternion difference: " << quat_diff);
        WARN("Translation difference: " << trans_diff);

        CHECK(quat_diff < 1e-6);
        CHECK(trans_diff < 1e-6);
        CHECK_THAT(resultLeft.rms, WithinAbs(resultRight.rms, 1e-8));
    }

    SECTION("Combined rotation and translation")
    {
        Pose7 initialPose = SolverTestFixture::createPose({ 0.03, 0.02, -0.01 }, 0.02, -0.01, 0.03);
        auto corrs = fix.computeCorrs(initialPose);

        auto resultRight = solveInner<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, params);

        auto resultLeft = solveInnerGNLeft<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, params);

        WARN("Right: iters=" << resultRight.iterations << ", rms=" << resultRight.rms);
        WARN("Left:  iters=" << resultLeft.iterations << ", rms=" << resultLeft.rms);

        double quat_diff = std::min(
            (resultLeft.pose.head<4>() - resultRight.pose.head<4>()).norm(),
            (resultLeft.pose.head<4>() + resultRight.pose.head<4>()).norm());
        double trans_diff = (resultLeft.pose.tail<3>() - resultRight.pose.tail<3>()).norm();

        WARN("Quaternion difference: " << quat_diff);
        WARN("Translation difference: " << trans_diff);

        CHECK(quat_diff < 1e-6);
        CHECK(trans_diff < 1e-6);
        CHECK_THAT(resultLeft.rms, WithinAbs(resultRight.rms, 1e-8));
    }
}

/**
 * Compare full ICP solver (with outer loop) between ambient and left-perturbation.
 */
TEST_CASE("Left vs Right full ICP: same result", "[icp][left][outer]")
{
    SolverTestFixture fix;

    auto innerParams = Presets::gaussNewton(12, false);
    OuterParams outerParams;
    outerParams.maxIterations = 6;
    outerParams.convergenceTol = 0.0;
    outerParams.maxDist = 100.0f;
    outerParams.verbose = false;

    SECTION("Small perturbation")
    {
        Pose7 initialPose = SolverTestFixture::createPose({ 0.01, 0.005, -0.008 }, 0.01, -0.005, 0.02);

        auto resultRight = solveICP<RayJacobianSimplified>(
            fix.source, fix.target, initialPose, fix.rayDir, fix.weighting, innerParams, outerParams);

        auto resultLeft = solveICPLeft<RayJacobianSimplified>(
            fix.source, fix.target, initialPose, fix.rayDir, fix.weighting, innerParams, outerParams);

        WARN("Right: outer=" << resultRight.outer_iterations
                             << ", total_inner=" << resultRight.total_inner_iterations
                             << ", rms=" << resultRight.rms);
        WARN("Left:  outer=" << resultLeft.outer_iterations
                             << ", total_inner=" << resultLeft.total_inner_iterations
                             << ", rms=" << resultLeft.rms);

        // Check final poses match (both should converge to identity)
        double quat_diff = std::min(
            (resultLeft.pose.head<4>() - resultRight.pose.head<4>()).norm(),
            (resultLeft.pose.head<4>() + resultRight.pose.head<4>()).norm());
        double trans_diff = (resultLeft.pose.tail<3>() - resultRight.pose.tail<3>()).norm();

        WARN("Quaternion difference: " << quat_diff);
        WARN("Translation difference: " << trans_diff);

        CHECK(quat_diff < 1e-5);
        CHECK(trans_diff < 1e-5);

        // Both should converge to identity pose
        CHECK(std::abs(resultLeft.pose[3] - 1.0) < 1e-5);  // qw ≈ 1
        CHECK(resultLeft.pose.head<3>().norm() < 1e-5);     // qx,qy,qz ≈ 0
        CHECK(resultLeft.pose.tail<3>().norm() < 1e-5);     // t ≈ 0
    }

    SECTION("Larger perturbation")
    {
        Pose7 initialPose = SolverTestFixture::createPose({ 0.05, 0.02, -0.03 }, 0.05, 0.02, 0.08);

        auto resultRight = solveICP<RayJacobianSimplified>(
            fix.source, fix.target, initialPose, fix.rayDir, fix.weighting, innerParams, outerParams);

        auto resultLeft = solveICPLeft<RayJacobianSimplified>(
            fix.source, fix.target, initialPose, fix.rayDir, fix.weighting, innerParams, outerParams);

        WARN("Right: outer=" << resultRight.outer_iterations
                             << ", total_inner=" << resultRight.total_inner_iterations
                             << ", rms=" << resultRight.rms);
        WARN("Left:  outer=" << resultLeft.outer_iterations
                             << ", total_inner=" << resultLeft.total_inner_iterations
                             << ", rms=" << resultLeft.rms);

        double quat_diff = std::min(
            (resultLeft.pose.head<4>() - resultRight.pose.head<4>()).norm(),
            (resultLeft.pose.head<4>() + resultRight.pose.head<4>()).norm());
        double trans_diff = (resultLeft.pose.tail<3>() - resultRight.pose.tail<3>()).norm();

        WARN("Quaternion difference: " << quat_diff);
        WARN("Translation difference: " << trans_diff);

        CHECK(quat_diff < 1e-5);
        CHECK(trans_diff < 1e-5);

        // Both should converge to identity pose
        CHECK(std::abs(resultLeft.pose[3] - 1.0) < 1e-5);
        CHECK(resultLeft.pose.head<3>().norm() < 1e-5);
        CHECK(resultLeft.pose.tail<3>().norm() < 1e-5);
    }
}

/**
 * Verify that the left solver produces correct residual at identity.
 */
TEST_CASE("Left solver: identity pose has zero residual", "[icp][left][identity]")
{
    SolverTestFixture fix;
    auto params = Presets::gaussNewton(5, false);

    Pose7 pose = identityPose();
    auto corrs = fix.computeCorrs(pose);

    auto result = solveInnerGNLeft<RayJacobianSimplified>(
        corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);

    WARN("RMS at identity: " << result.rms);
    WARN("Iterations: " << result.iterations);

    CHECK(result.rms < 1e-8);
}

/**
 * Iteration count comparison - left and right may differ but should be similar.
 */
TEST_CASE("Left vs Right: iteration count comparison", "[icp][left][iterations]")
{
    SolverTestFixture fix;
    auto params = Presets::gaussNewton(50, false);

    Pose7 initialPose = SolverTestFixture::createPose({ 0.05, 0.03, -0.04 }, 0.04, -0.02, 0.06);
    auto corrs = fix.computeCorrs(initialPose);

    auto resultRight = solveInner<RayJacobianSimplified>(
        corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, params);

    auto resultLeft = solveInnerGNLeft<RayJacobianSimplified>(
        corrs.forward, corrs.reverse, initialPose, fix.rayDir, fix.weighting, params);

    WARN("Right iterations: " << resultRight.iterations);
    WARN("Left iterations:  " << resultLeft.iterations);
    WARN("Iteration difference: " << std::abs(resultRight.iterations - resultLeft.iterations));

    // Both should converge
    CHECK(resultRight.converged);
    CHECK(resultLeft.converged);

    // Final RMS should match closely
    CHECK_THAT(resultLeft.rms, WithinAbs(resultRight.rms, 1e-8));
}
