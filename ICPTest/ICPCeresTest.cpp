#include <ICP/ICPCeres.h>
#include <ICP/GridFactory.h>
#include <ICP/EigenUtils.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>

using namespace ICP;
using Catch::Matchers::WithinAbs;

TEST_CASE("solveInnerCeres with fixed correspondences", "[icp][inner][ceres]")
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
    WARN("Total vertices: " << source.nRows() * source.nCols());
    WARN("Triangle vertex indices: " << source.getTriangleVertexIndices().size());

    Vector3 rayDir(0, 0, -1);
    float maxDist = 100.0f;

    // Configure Ceres options
    CeresICPOptions ceresOpts;
    ceresOpts.maxIterations = 12;
    ceresOpts.functionTolerance = 1e-9;
    ceresOpts.gradientTolerance = 1e-9;
    ceresOpts.parameterTolerance = 1e-9;
    ceresOpts.useLM = false;  // Use GN approximation
    ceresOpts.verbose = false;
    ceresOpts.silent = true;

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

        auto result = solveInnerCeres<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, rayDir, weighting, ceresOpts);

        WARN("Ceres RMS: " << result.rms);
        WARN("Ceres Iterations: " << result.iterations);
        WARN("Ceres Valid count: " << result.valid_count);
        WARN("Ceres Termination: " << result.summary.termination_type);
        WARN("Final pose: [qx=" << result.pose[0] << ", qy=" << result.pose[1]
             << ", qz=" << result.pose[2] << ", qw=" << result.pose[3]
             << ", tx=" << result.pose[4] << ", ty=" << result.pose[5]
             << ", tz=" << result.pose[6] << "]");

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

        auto result = solveInnerCeres<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, rayDir, weighting, ceresOpts);

        WARN("Ceres RMS: " << result.rms);
        WARN("Ceres Iterations: " << result.iterations);
        WARN("Ceres Valid count: " << result.valid_count);
        WARN("Ceres Termination: " << result.summary.termination_type);
        WARN("Final pose: [qx=" << result.pose[0] << ", qy=" << result.pose[1]
             << ", qz=" << result.pose[2] << ", qw=" << result.pose[3]
             << ", tx=" << result.pose[4] << ", ty=" << result.pose[5]
             << ", tz=" << result.pose[6] << "]");
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

        auto result = solveInnerCeres<RayJacobianSimplified>(
            corrs.forward, corrs.reverse, pose, rayDir, weighting, ceresOpts);

        WARN("Ceres RMS: " << result.rms);
        WARN("Ceres Iterations: " << result.iterations);
        WARN("Ceres Valid count: " << result.valid_count);
        WARN("Ceres Termination: " << result.summary.termination_type);
        WARN("Final pose: [qx=" << result.pose[0] << ", qy=" << result.pose[1]
             << ", qz=" << result.pose[2] << ", qw=" << result.pose[3]
             << ", tx=" << result.pose[4] << ", ty=" << result.pose[5]
             << ", tz=" << result.pose[6] << "]");
    }
}
