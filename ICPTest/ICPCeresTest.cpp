// Catch2 headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Test utilities
#include "TestUtils.h"

using namespace ICP;
using namespace TestUtils;
using Catch::Matchers::WithinAbs;

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
