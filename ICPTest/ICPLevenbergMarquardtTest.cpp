#include <ICP/ICPSimple.h>
#include <ICP/ICPCeres.h>
#include <ICP/GridFactory.h>
#include <ICP/EigenUtils.h>
#include <ICP/Correspondences.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace ICP;

// ============================================================================
// Test fixture with common setup
// ============================================================================

struct SolverTestFixture
{
    Grid source;
    Grid target;
    GeometryWeighting weighting;
    Vector3 rayDir{0, 0, -1};
    float maxDist = 100.0f;

    SolverTestFixture()
    {
        float spacing = 2.0f / 44.0f;
        source = createHeightfieldGrid(45, 45, spacing, spacing);
        target = source;

        weighting.enable_weight = false;
        weighting.enable_gate = false;
    }

    // Compute correspondences for a given pose
    BidirectionalCorrs computeCorrs(const Pose7& pose) const
    {
        Quaternion q(pose[3], pose[0], pose[1], pose[2]);
        Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
        srcToTgt.linear() = q.toRotationMatrix();
        srcToTgt.translation() = pose.tail<3>();
        return computeBidirectionalCorrs(source, target, rayDir.cast<float>(), srcToTgt, maxDist);
    }

    // Create pose from axis-angle and translation
    static Pose7 createPose(const Vector3& axisAngle, double tx = 0, double ty = 0, double tz = 0)
    {
        Pose7 pose = rotationPose(axisAngle);
        pose[4] = tx;
        pose[5] = ty;
        pose[6] = tz;
        return pose;
    }
};

// ============================================================================
// Solver configuration presets
// ============================================================================

namespace SolverPresets
{
    inline InnerParams gaussNewton(int maxIter = 20, bool verbose = true)
    {
        InnerParams p;
        p.solverType = SolverType::GaussNewton;
        p.maxIterations = maxIter;
        p.translationThreshold = 1e-9;
        p.rotationThreshold = 1e-9;
        p.damping = 0.0;
        p.verbose = verbose;
        return p;
    }

    inline InnerParams gaussNewtonWithLineSearch(int maxIter = 20, bool verbose = true)
    {
        InnerParams p = gaussNewton(maxIter, verbose);
        p.lineSearch.enabled = true;
        p.lineSearch.maxIterations = 10;
        p.lineSearch.alpha = 1.0;
        p.lineSearch.beta = 0.5;
        return p;
    }

    inline InnerParams lmFixed(double lambda = 1e-6, int maxIter = 20, bool verbose = true)
    {
        InnerParams p;
        p.solverType = SolverType::LevenbergMarquardt;
        p.maxIterations = maxIter;
        p.translationThreshold = 1e-9;
        p.rotationThreshold = 1e-9;
        p.verbose = verbose;
        p.lm.lambda = lambda;
        p.lm.fixedLambda = true;
        return p;
    }

    inline InnerParams lmAdaptive(double lambda = 1e-3, int maxIter = 20, bool verbose = true)
    {
        InnerParams p;
        p.solverType = SolverType::LevenbergMarquardt;
        p.maxIterations = maxIter;
        p.translationThreshold = 1e-9;
        p.rotationThreshold = 1e-9;
        p.verbose = verbose;
        p.lm.lambda = lambda;
        p.lm.fixedLambda = false;
        p.lm.lambdaUp = 10.0;
        p.lm.lambdaDown = 0.1;
        p.lm.lambdaMin = 1e-10;
        p.lm.lambdaMax = 1e10;
        return p;
    }

    inline InnerParams lmAdaptiveWithLineSearch(double lambda = 1e-3, int maxIter = 20, bool verbose = true)
    {
        InnerParams p = lmAdaptive(lambda, maxIter, verbose);
        p.lineSearch.enabled = true;
        p.lineSearch.maxIterations = 10;
        p.lineSearch.alpha = 1.0;
        p.lineSearch.beta = 0.5;
        return p;
    }

    inline InnerParams lmGainRatio(double lambda = 1e-3, int maxIter = 20, bool verbose = true)
    {
        InnerParams p;
        p.solverType = SolverType::LevenbergMarquardt;
        p.maxIterations = maxIter;
        p.translationThreshold = 1e-9;
        p.rotationThreshold = 1e-9;
        p.verbose = verbose;
        p.lm.strategy = LMStrategy::GainRatio;
        p.lm.lambda = lambda;
        p.lm.lambdaMin = 1e-10;
        p.lm.lambdaMax = 1e10;
        p.lm.minRelativeDecrease = 1e-3;  // Ceres default
        return p;
    }

    inline CeresICPOptions ceresLM(double lambda = 1e-3, int maxIter = 20, bool verbose = true)
    {
        CeresICPOptions opts;
        opts.maxIterations = maxIter;
        opts.functionTolerance = 1e-9;
        opts.gradientTolerance = 1e-9;
        opts.parameterTolerance = 1e-9;
        opts.useLM = true;
        opts.initialTrustRegionRadius = 1.0 / lambda;
        opts.maxTrustRegionRadius = 1e10;
        opts.verbose = verbose;
        opts.silent = !verbose;
        return opts;
    }

    inline CeresICPOptions ceresGN(int maxIter = 20, bool verbose = true)
    {
        CeresICPOptions opts;
        opts.maxIterations = maxIter;
        opts.functionTolerance = 1e-9;
        opts.gradientTolerance = 1e-9;
        opts.parameterTolerance = 1e-9;
        opts.useLM = false;
        opts.verbose = verbose;
        opts.silent = !verbose;
        return opts;
    }
}

// ============================================================================
// Result collection and reporting
// ============================================================================

struct SolverResult
{
    std::string name;
    int iterations;
    double rms;
};

inline void printResults(const std::vector<SolverResult>& results)
{
    WARN("\n=== Summary ===");
    for (const auto& r : results)
    {
        WARN(std::left << std::setw(20) << r.name << ": iters=" << r.iterations
             << ", rms=" << std::scientific << std::setprecision(6) << r.rms);
    }
}

// ============================================================================
// Tests
// ============================================================================

TEST_CASE("Levenberg-Marquardt vs Gauss-Newton", "[icp][inner][lm]")
{
    SolverTestFixture fix;
    std::vector<SolverResult> results;

    SECTION("Large rotation perturbation")
    {
        Pose7 pose = SolverTestFixture::createPose({0.15, 0.08, -0.12});
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward: " << corrs.forward.size() << ", Reverse: " << corrs.reverse.size());

        // GN
        {
            auto params = SolverPresets::gaussNewton(12);
            WARN("\n=== Gauss-Newton ===");
            auto r = solveInnerGN<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"GN", r.iterations, r.rms});
        }

        // LM fixed
        {
            auto params = SolverPresets::lmFixed(1e-6, 12);
            WARN("\n=== LM (fixed lambda) ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"LM (fixed)", r.iterations, r.rms});
        }

        // LM adaptive
        {
            auto params = SolverPresets::lmAdaptive(1e-3, 12);
            WARN("\n=== LM (adaptive) ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"LM (adaptive)", r.iterations, r.rms});
        }

        printResults(results);
    }

    SECTION("Large translation + rotation")
    {
        Pose7 pose = SolverTestFixture::createPose({0.10, 0.05, -0.08}, 0.08, -0.06, 0.04);
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward: " << corrs.forward.size() << ", Reverse: " << corrs.reverse.size());

        // GN
        {
            auto params = SolverPresets::gaussNewton(12);
            WARN("\n=== Gauss-Newton ===");
            auto r = solveInner<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"GN", r.iterations, r.rms});
        }

        // LM adaptive
        {
            auto params = SolverPresets::lmAdaptive(1e-3, 12);
            WARN("\n=== LM (adaptive) ===");
            auto r = solveInner<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"LM (adaptive)", r.iterations, r.rms});
        }

        printResults(results);
    }

    SECTION("Line search comparison")
    {
        // Very large perturbation (~45 degrees)
        Pose7 pose = SolverTestFixture::createPose({0.4, 0.35, -0.4}, 0.20, -0.18, 0.15);
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward: " << corrs.forward.size() << ", Reverse: " << corrs.reverse.size());

        // GN
        {
            auto params = SolverPresets::gaussNewton();
            WARN("\n=== GN ===");
            auto r = solveInnerGN<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"GN", r.iterations, r.rms});
            CHECK(r.rms < 0.1);
        }

        // GN + line search
        {
            auto params = SolverPresets::gaussNewtonWithLineSearch();
            WARN("\n=== GN + LS ===");
            auto r = solveInnerGN<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"GN + LS", r.iterations, r.rms});
            CHECK(r.rms < 0.1);
        }

        // LM
        {
            auto params = SolverPresets::lmAdaptive();
            WARN("\n=== LM ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"LM", r.iterations, r.rms});
            CHECK(r.rms < 0.1);
        }

        // LM + line search
        {
            auto params = SolverPresets::lmAdaptiveWithLineSearch();
            WARN("\n=== LM + LS ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"LM + LS", r.iterations, r.rms});
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
        Pose7 pose = SolverTestFixture::createPose({0.15, 0.08, -0.12}, 0.10, -0.08, 0.06);
        auto corrs = fix.computeCorrs(pose);

        WARN("Forward: " << corrs.forward.size() << ", Reverse: " << corrs.reverse.size());

        // Hand-rolled LM (Simple strategy)
        {
            auto params = SolverPresets::lmAdaptive();
            WARN("\n=== Hand-rolled LM (Simple) ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"LM Simple", r.iterations, r.rms});
            CHECK(r.rms < 0.05);
        }

        // Hand-rolled LM (GainRatio strategy)
        {
            auto params = SolverPresets::lmGainRatio();
            WARN("\n=== Hand-rolled LM (GainRatio) ===");
            auto r = solveInnerLM<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, params);
            results.push_back(SolverResult{"LM GainRatio", r.iterations, r.rms});
            CHECK(r.rms < 0.05);
        }

        // Ceres LM
        {
            auto opts = SolverPresets::ceresLM();
            WARN("\n=== Ceres LM ===");
            auto r = solveInnerCeres<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, opts);
            results.push_back(SolverResult{"Ceres LM", r.iterations, r.rms});
            CHECK(r.rms < 0.05);
        }

        // Ceres GN
        {
            auto opts = SolverPresets::ceresGN();
            WARN("\n=== Ceres GN ===");
            auto r = solveInnerCeres<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, pose, fix.rayDir, fix.weighting, opts);
            results.push_back(SolverResult{"Ceres GN", r.iterations, r.rms});
            CHECK(r.rms < 0.05);
        }

        printResults(results);
    }
}
