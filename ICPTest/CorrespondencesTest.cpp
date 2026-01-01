// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

// Standard C++ headers
#include <cmath>

// Catch2 headers
#include <catch2/catch_test_macros.hpp>

// Internal headers
#include <ICP/Config.h>
#include <ICP/Correspondences.h>
#include <ICP/GridFactory.h>

using namespace ICP;

// Allow small variation in correspondence count on CI due to Embree version differences
#if ICP_CI_BUILD
#define CHECK_CORR_COUNT(actual, expected) CHECK((actual) >= (expected) - 20)
#else
#define CHECK_CORR_COUNT(actual, expected) CHECK((actual) == (expected))
#endif

TEST_CASE("Correspondence count with two hemispheres", "[correspondences]")
{
    Grid source = createTwoHemispheresGrid();
    Grid target = source;

    Eigen::Vector3f rayDir(0, 0, -1);
    float maxDist = 100.0f;

    // Identity transform (use initialPose for Isometry3d operations)
    Eigen::Isometry3d srcToTgt = target.initialPose.inverse() * source.initialPose;
    int baseline = static_cast<int>(computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist).size());
    REQUIRE(baseline > 1000);

    SECTION("Identity transform - many correspondences")
    {
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK_CORR_COUNT(corrs.size(), baseline);
    }

    SECTION("Z translation preserves hits")
    {
        target.initialPose.translation() = Eigen::Vector3d(0, 0, 1.0);
        srcToTgt = target.initialPose.inverse() * source.initialPose;
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK_CORR_COUNT(corrs.size(), baseline);

        target.initialPose.translation() = Eigen::Vector3d(0, 0, -1.0);
        srcToTgt = target.initialPose.inverse() * source.initialPose;
        corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK_CORR_COUNT(corrs.size(), baseline);
    }

    SECTION("X/Y translation reduces hits")
    {
        target.initialPose.translation() = Eigen::Vector3d(2.0, 0, 0);
        srcToTgt = target.initialPose.inverse() * source.initialPose;
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() < baseline);

        target.initialPose.translation() = Eigen::Vector3d(0, 2.0, 0);
        srcToTgt = target.initialPose.inverse() * source.initialPose;
        corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() < baseline);
    }

    SECTION("Z rotation preserves most hits")
    {
        double angle = 10.0 * M_PI / 180.0;
        target.initialPose.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        srcToTgt = target.initialPose.inverse() * source.initialPose;
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() >= baseline * 0.8);
    }

    SECTION("X/Y rotation reduces hits")
    {
        double angle = 5.0 * M_PI / 180.0;

        target.initialPose.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
        srcToTgt = target.initialPose.inverse() * source.initialPose;
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() < baseline);

        target.initialPose.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()).toRotationMatrix();
        srcToTgt = target.initialPose.inverse() * source.initialPose;
        corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() < baseline);
    }

    SECTION("Larger rotation reduces hits progressively")
    {
        target.initialPose.linear() = Eigen::AngleAxisd(5.0 * M_PI / 180.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
        srcToTgt = target.initialPose.inverse() * source.initialPose;
        int hits5 = static_cast<int>(computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist).size());

        target.initialPose.linear() = Eigen::AngleAxisd(20.0 * M_PI / 180.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
        srcToTgt = target.initialPose.inverse() * source.initialPose;
        int hits20 = static_cast<int>(computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist).size());

        CHECK(hits5 < baseline);
        CHECK(hits20 < hits5);
    }
}
