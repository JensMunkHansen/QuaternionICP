/**
 * Tests for two-pose Ceres ICP solver.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <ICP/EigenUtils.h>
#include <ICP/GridFactory.h>
#include <ICP/ICPCeresTwoPose.h>
#include <ICP/ICPParams.h>

using namespace ICP;
using Catch::Matchers::WithinAbs;

TEST_CASE("Two-pose Ceres solver compiles and runs", "[ceres][twopose]")
{
    // Create two identical grids
    Grid gridA = createTwoHemispheresGrid(32, 32);
    Grid gridB = createTwoHemispheresGrid(32, 32);

    // Start with identity poses
    Pose7 poseA = identityPose();
    Pose7 poseB = identityPose();

    // Perturb pose B slightly
    PerturbationRNGs rngs;
    perturbPoseRotation(poseB, 2.0, rngs.rotation);      // 2 degrees
    perturbPoseTranslation(poseB, 0.05, rngs.translation); // 0.05 units

    Vector3 rayDir(0.0, 0.0, -1.0);
    GeometryWeighting weighting;
    CeresICPOptions ceresOpts;
    ceresOpts.verbose = false;
    ceresOpts.silent = true;
    OuterParams outerParams;
    outerParams.maxIterations = 5;

    // Run two-pose solver with A fixed
    auto result = solveICPCeresTwoPose<RayJacobianSimplified>(
        gridA, gridB, poseA, poseB, rayDir, weighting, ceresOpts, outerParams, true);

    REQUIRE(result.outer_iterations > 0);
    REQUIRE(result.total_inner_iterations > 0);

    // Pose A should remain identity since it was fixed
    REQUIRE_THAT(result.poseA[0], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(result.poseA[1], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(result.poseA[2], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(result.poseA[3], WithinAbs(1.0, 1e-10));

    INFO("Final RMS: " << result.rms);
    INFO("Outer iterations: " << result.outer_iterations);
    INFO("Total inner iterations: " << result.total_inner_iterations);
}

TEST_CASE("Two-pose solver converges to identity for identical grids", "[ceres][twopose]")
{
    // Create two identical grids
    Grid gridA = createTwoHemispheresGrid(32, 32);
    Grid gridB = createTwoHemispheresGrid(32, 32);

    // Start with identity for A, perturbed for B
    Pose7 poseA = identityPose();
    Pose7 poseB = identityPose();

    PerturbationRNGs rngs;
    perturbPoseRotation(poseB, 1.0, rngs.rotation);
    perturbPoseTranslation(poseB, 0.02, rngs.translation);

    Vector3 rayDir(0.0, 0.0, -1.0);
    GeometryWeighting weighting;
    CeresICPOptions ceresOpts;
    ceresOpts.verbose = false;
    ceresOpts.silent = true;
    OuterParams outerParams;
    outerParams.maxIterations = 10;

    // Fix pose A, optimize pose B
    auto result = solveICPCeresTwoPose<RayJacobianSimplified>(
        gridA, gridB, poseA, poseB, rayDir, weighting, ceresOpts, outerParams, true);

    // After convergence, relative pose should be near identity
    Pose7 relPose = computeRelativePose(result.poseA, result.poseB);

    // Quaternion should be near identity: [0, 0, 0, 1]
    double qNorm = std::sqrt(relPose[0]*relPose[0] + relPose[1]*relPose[1] +
                             relPose[2]*relPose[2]);
    REQUIRE(qNorm < 0.01);  // Small rotation

    // Translation should be near zero
    double tNorm = std::sqrt(relPose[4]*relPose[4] + relPose[5]*relPose[5] +
                             relPose[6]*relPose[6]);
    REQUIRE(tNorm < 0.01);  // Small translation

    INFO("Relative rotation (xyz norm): " << qNorm);
    INFO("Relative translation norm: " << tNorm);
    INFO("Final RMS: " << result.rms);
}

TEST_CASE("Two-pose solver with both poses free", "[ceres][twopose]")
{
    // Create two identical grids
    Grid gridA = createTwoHemispheresGrid(32, 32);
    Grid gridB = createTwoHemispheresGrid(32, 32);

    // Start with identity for both, perturb in opposite directions
    Pose7 poseA = identityPose();
    Pose7 poseB = identityPose();

    PerturbationRNGs rngsA(100, 101, 102);
    PerturbationRNGs rngsB(200, 201, 202);
    perturbPoseRotation(poseA, 0.5, rngsA.rotation);
    perturbPoseTranslation(poseA, 0.01, rngsA.translation);
    perturbPoseRotation(poseB, 0.5, rngsB.rotation);
    perturbPoseTranslation(poseB, 0.01, rngsB.translation);

    // Record initial relative pose
    Pose7 initialRel = computeRelativePose(poseA, poseB);

    Vector3 rayDir(0.0, 0.0, -1.0);
    GeometryWeighting weighting;
    CeresICPOptions ceresOpts;
    ceresOpts.verbose = false;
    ceresOpts.silent = true;
    OuterParams outerParams;
    outerParams.maxIterations = 10;

    // Both poses free
    auto result = solveICPCeresTwoPose<RayJacobianSimplified>(
        gridA, gridB, poseA, poseB, rayDir, weighting, ceresOpts, outerParams, false);

    // Final relative pose should be near identity
    Pose7 finalRel = computeRelativePose(result.poseA, result.poseB);

    double finalQNorm = std::sqrt(finalRel[0]*finalRel[0] + finalRel[1]*finalRel[1] +
                                  finalRel[2]*finalRel[2]);
    double finalTNorm = std::sqrt(finalRel[4]*finalRel[4] + finalRel[5]*finalRel[5] +
                                  finalRel[6]*finalRel[6]);

    REQUIRE(finalQNorm < 0.01);
    REQUIRE(finalTNorm < 0.01);

    INFO("Initial relative rotation (xyz norm): " << std::sqrt(initialRel[0]*initialRel[0] + initialRel[1]*initialRel[1] + initialRel[2]*initialRel[2]));
    INFO("Final relative rotation (xyz norm): " << finalQNorm);
    INFO("Final relative translation norm: " << finalTNorm);
    INFO("Final RMS: " << result.rms);
    INFO("Converged: " << result.converged);
}
