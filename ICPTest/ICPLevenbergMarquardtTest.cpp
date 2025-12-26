#include <ICP/ICPSimple.h>
#include <ICP/GridFactory.h>
#include <ICP/EigenUtils.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>

using namespace ICP;
using Catch::Matchers::WithinAbs;

TEST_CASE("Levenberg-Marquardt vs Gauss-Newton", "[icp][inner][lm]")
{
    // Create test grids
    float spacing = 2.0f / 44.0f;
    Grid source = createHeightfieldGrid(45, 45, spacing, spacing);
    Grid target = source;

    GeometryWeighting weighting;
    weighting.enable_weight = false;
    weighting.enable_gate = false;

    Vector3 rayDir(0, 0, -1);
    float maxDist = 100.0f;

    SECTION("Large rotation perturbation - LM should converge better")
    {
        // Large rotation that makes linearization poor
        Vector3 axisAngle(0.15, 0.08, -0.12);  // Larger than normal test
        Pose7 initialPose = rotationPose(axisAngle);

        Quaternion q(initialPose[3], initialPose[0], initialPose[1], initialPose[2]);
        Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
        srcToTgt.linear() = q.toRotationMatrix();
        auto corrs = computeBidirectionalCorrs(source, target, rayDir.cast<float>(), srcToTgt, maxDist);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        // Test 1: Gauss-Newton (no damping)
        {
            InnerParams gnParams;
            gnParams.solverType = SolverType::GaussNewton;
            gnParams.maxIterations = 12;
            gnParams.stepTol = 1e-9;
            gnParams.damping = 0.0;
            gnParams.verbose = true;

            WARN("\n=== Gauss-Newton (damping=0) ===");
            auto result = solveInnerGN<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, initialPose, rayDir, weighting, gnParams);

            WARN("GN: iterations=" << result.iterations << ", rms=" << result.rms);
        }

        // Test 2: Gauss-Newton with small damping
        {
            InnerParams gnParams;
            gnParams.solverType = SolverType::GaussNewton;
            gnParams.maxIterations = 12;
            gnParams.stepTol = 1e-9;
            gnParams.damping = 1e-6;
            gnParams.verbose = true;

            WARN("\n=== Gauss-Newton (damping=1e-6) ===");
            auto result = solveInnerGN<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, initialPose, rayDir, weighting, gnParams);

            WARN("GN: iterations=" << result.iterations << ", rms=" << result.rms);
        }

        // Test 3: LM with fixed lambda
        {
            InnerParams lmParams;
            lmParams.solverType = SolverType::LevenbergMarquardt;
            lmParams.maxIterations = 12;
            lmParams.stepTol = 1e-9;
            lmParams.verbose = true;
            lmParams.lm.lambda = 1e-6;
            lmParams.lm.fixedLambda = true;  // Fixed lambda mode

            WARN("\n=== LM with fixed lambda (1e-6) ===");
            auto result = solveInnerLM<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, initialPose, rayDir, weighting, lmParams);

            WARN("LM (fixed): iterations=" << result.iterations << ", rms=" << result.rms);
        }

        // Test 4: Adaptive Levenberg-Marquardt
        {
            InnerParams lmParams;
            lmParams.solverType = SolverType::LevenbergMarquardt;
            lmParams.maxIterations = 12;
            lmParams.stepTol = 1e-9;
            lmParams.verbose = true;
            lmParams.lm.lambda = 1e-3;        // Initial lambda
            lmParams.lm.fixedLambda = false;  // Adaptive mode
            lmParams.lm.lambdaUp = 10.0;      // Increase factor
            lmParams.lm.lambdaDown = 0.1;     // Decrease factor
            lmParams.lm.lambdaMin = 1e-10;
            lmParams.lm.lambdaMax = 1e10;

            WARN("\n=== Adaptive Levenberg-Marquardt ===");
            auto result = solveInnerLM<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, initialPose, rayDir, weighting, lmParams);

            WARN("LM (adaptive): iterations=" << result.iterations << ", rms=" << result.rms);
        }
    }

    SECTION("Large translation + rotation - challenging case")
    {
        // Combined large translation and rotation
        Vector3 axisAngle(0.10, 0.05, -0.08);
        Pose7 initialPose = rotationPose(axisAngle);
        initialPose[4] = 0.08;  // tx
        initialPose[5] = -0.06; // ty
        initialPose[6] = 0.04;  // tz

        Quaternion q(initialPose[3], initialPose[0], initialPose[1], initialPose[2]);
        Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
        srcToTgt.linear() = q.toRotationMatrix();
        srcToTgt.translation() = initialPose.tail<3>();
        auto corrs = computeBidirectionalCorrs(source, target, rayDir.cast<float>(), srcToTgt, maxDist);

        WARN("\nForward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        // Gauss-Newton
        {
            InnerParams gnParams;
            gnParams.solverType = SolverType::GaussNewton;
            gnParams.maxIterations = 12;
            gnParams.stepTol = 1e-9;
            gnParams.damping = 0.0;
            gnParams.verbose = true;

            WARN("\n=== Gauss-Newton ===");
            auto result = solveInner<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, initialPose, rayDir, weighting, gnParams);

            WARN("GN: iterations=" << result.iterations << ", rms=" << result.rms);
        }

        // LM with fixed lambda
        {
            InnerParams lmParams;
            lmParams.solverType = SolverType::LevenbergMarquardt;
            lmParams.maxIterations = 12;
            lmParams.stepTol = 1e-9;
            lmParams.verbose = true;
            lmParams.lm.lambda = 1e-6;
            lmParams.lm.fixedLambda = true;

            WARN("\n=== LM with fixed lambda ===");
            auto result = solveInner<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, initialPose, rayDir, weighting, lmParams);

            WARN("LM (fixed): iterations=" << result.iterations << ", rms=" << result.rms);
        }

        // Adaptive LM
        {
            InnerParams lmParams;
            lmParams.solverType = SolverType::LevenbergMarquardt;
            lmParams.maxIterations = 12;
            lmParams.stepTol = 1e-9;
            lmParams.verbose = true;
            lmParams.lm.lambda = 1e-3;
            lmParams.lm.fixedLambda = false;
            lmParams.lm.lambdaUp = 10.0;
            lmParams.lm.lambdaDown = 0.1;
            lmParams.lm.lambdaMin = 1e-10;
            lmParams.lm.lambdaMax = 1e10;

            WARN("\n=== Adaptive Levenberg-Marquardt ===");
            auto result = solveInner<RayJacobianSimplified>(
                corrs.forward, corrs.reverse, initialPose, rayDir, weighting, lmParams);

            WARN("LM (adaptive): iterations=" << result.iterations << ", rms=" << result.rms);
        }
    }
}
