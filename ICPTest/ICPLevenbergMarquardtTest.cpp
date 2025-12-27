// Catch2 headers
#include <catch2/catch_test_macros.hpp>

// Test utilities
#include "TestUtils.h"

using namespace ICP;
using namespace TestUtils;

TEST_CASE("Levenberg-Marquardt vs Gauss-Newton", "[icp][inner][lm]")
{
    SolverTestFixture fix;
    std::vector<SolverResult> results;

    SECTION("Large rotation perturbation")
    {
        Pose7 pose = SolverTestFixture::createPose({ 0.15, 0.08, -0.12 });
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward: " << corrs.forward.size() << ", Reverse: " << corrs.reverse.size());

        // GN
        {
            auto params = Presets::gaussNewton(12);
            WARN("\n=== Gauss-Newton ===");
            auto r = solveInnerGN<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "GN", r.iterations, r.rms });
        }

        // LM fixed
        {
            auto params = Presets::lmFixed(1e-6, 12);
            WARN("\n=== LM (fixed lambda) ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "LM (fixed)", r.iterations, r.rms });
        }

        // LM adaptive
        {
            auto params = Presets::lmAdaptive(1e-3, 12);
            WARN("\n=== LM (adaptive) ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "LM (adaptive)", r.iterations, r.rms });
        }

        printResults(results);
    }

    SECTION("Large translation + rotation")
    {
        Pose7 pose = SolverTestFixture::createPose({ 0.10, 0.05, -0.08 }, 0.08, -0.06, 0.04);
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward: " << corrs.forward.size() << ", Reverse: " << corrs.reverse.size());

        // GN
        {
            auto params = Presets::gaussNewton(12);
            WARN("\n=== Gauss-Newton ===");
            auto r = solveInner<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "GN", r.iterations, r.rms });
        }

        // LM adaptive
        {
            auto params = Presets::lmAdaptive(1e-3, 12);
            WARN("\n=== LM (adaptive) ===");
            auto r = solveInner<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "LM (adaptive)", r.iterations, r.rms });
        }

        printResults(results);
    }

    SECTION("Line search comparison")
    {
        Pose7 pose = SolverTestFixture::createPose({ 0.4, 0.35, -0.4 }, 0.20, -0.18, 0.15);
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward: " << corrs.forward.size() << ", Reverse: " << corrs.reverse.size());

        // GN
        {
            auto params = Presets::gaussNewton();
            WARN("\n=== GN ===");
            auto r = solveInnerGN<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "GN", r.iterations, r.rms });
            CHECK(r.rms < 0.1);
        }

        // GN + line search
        {
            auto params = Presets::gaussNewtonWithLineSearch();
            WARN("\n=== GN + LS ===");
            auto r = solveInnerGN<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "GN + LS", r.iterations, r.rms });
            CHECK(r.rms < 0.1);
        }

        // LM
        {
            auto params = Presets::lmAdaptive();
            WARN("\n=== LM ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "LM", r.iterations, r.rms });
            CHECK(r.rms < 0.1);
        }

        // LM + line search
        {
            auto params = Presets::lmAdaptiveWithLineSearch();
            WARN("\n=== LM + LS ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "LM + LS", r.iterations, r.rms });
            CHECK(r.rms < 0.1);
        }

        printResults(results);
    }
}

TEST_CASE("Hand-rolled vs Ceres", "[icp][inner][lm][ceres]")
{
    SolverTestFixture fix;
    std::vector<SolverResult> results;

    SECTION("Large perturbation comparison")
    {
        Pose7 pose = SolverTestFixture::createPose({ 0.15, 0.08, -0.12 }, 0.10, -0.08, 0.06);
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward: " << corrs.forward.size() << ", Reverse: " << corrs.reverse.size());

        // Hand-rolled LM (Simple strategy)
        {
            auto params = Presets::lmAdaptive();
            WARN("\n=== Hand-rolled LM (Simple) ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "LM Simple", r.iterations, r.rms });
            CHECK(r.rms < 0.05);
        }

        // Hand-rolled LM (GainRatio strategy)
        {
            auto params = Presets::lmGainRatio();
            WARN("\n=== Hand-rolled LM (GainRatio) ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back({ "LM GainRatio", r.iterations, r.rms });
            CHECK(r.rms < 0.05);
        }

        // Ceres LM
        {
            auto opts = Presets::ceresLM();
            WARN("\n=== Ceres LM ===");
            auto r = solveInnerCeres<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, opts);
            results.push_back({ "Ceres LM", r.iterations, r.rms });
            CHECK(r.rms < 0.05);
        }

        // Ceres GN
        {
            auto opts = Presets::ceresGN();
            WARN("\n=== Ceres GN ===");
            auto r = solveInnerCeres<RayJacobianSimplified>(
              corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, opts);
            results.push_back({ "Ceres GN", r.iterations, r.rms });
            CHECK(r.rms < 0.05);
        }

        printResults(results);
    }
}
