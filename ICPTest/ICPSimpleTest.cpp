#include <ICP/ICPSimple.h>
#include <ICP/GridFactory.h>
#include <ICP/EigenUtils.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>

using namespace ICP;
using Catch::Matchers::WithinAbs;

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

    auto v0 = source.getVertex(0);
    auto vLast = source.getVertex(45*45 - 1);
    float diag = std::sqrt(std::pow(vLast.x() - v0.x(), 2) + std::pow(vLast.y() - v0.y(), 2));
    WARN("C++ grid: min=(" << v0.x() << ", " << v0.y() << "), max=(" << vLast.x() << ", " << vLast.y() << ")");
    WARN("C++ XY diagonal: " << diag);

    Vector3 rayDir(0, 0, -1);
    float maxDist = 100.0f;

    SECTION("Identity pose")
    {
        Pose7 pose = identityPose();
        Eigen::Isometry3d srcToTgt = Eigen::Isometry3d::Identity();
        auto corrs = computeBidirectionalCorrs(source, target, rayDir.cast<float>(), srcToTgt, maxDist);

        WARN("Forward correspondences: " << corrs.forward.size());
        WARN("Reverse correspondences: " << corrs.reverse.size());

        auto result = solveInner<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, rayDir, weighting);

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
        InnerParams singleIter;
        singleIter.maxIterations = 1;
        singleIter.stepTol = 0;  // Don't stop early

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
        InnerParams singleIter;
        singleIter.maxIterations = 1;
        singleIter.stepTol = 0;  // Don't stop early

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

