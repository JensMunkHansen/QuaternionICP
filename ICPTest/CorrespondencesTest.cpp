// Standard C++ headers
#include <cmath>

// Catch2 headers
#include <catch2/catch_test_macros.hpp>

// Internal headers
#include <ICP/Correspondences.h>
#include <ICP/GridFactory.h>

using namespace ICP;

TEST_CASE("Correspondence count with two hemispheres", "[correspondences]")
{
    Grid source = createTwoHemispheresGrid();
    Grid target = source;

    Eigen::Vector3f rayDir(0, 0, -1);
    float maxDist = 100.0f;

    // Identity transform
    Eigen::Isometry3d srcToTgt = target.pose.inverse() * source.pose;
    int baseline = static_cast<int>(computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist).size());
    REQUIRE(baseline > 1000);

    SECTION("Identity transform - many correspondences")
    {
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() == baseline);
    }

    SECTION("Z translation preserves hits")
    {
        target.pose.translation() = Eigen::Vector3d(0, 0, 1.0);
        srcToTgt = target.pose.inverse() * source.pose;
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() == baseline);

        target.pose.translation() = Eigen::Vector3d(0, 0, -1.0);
        srcToTgt = target.pose.inverse() * source.pose;
        corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() == baseline);
    }

    SECTION("X/Y translation reduces hits")
    {
        target.pose.translation() = Eigen::Vector3d(2.0, 0, 0);
        srcToTgt = target.pose.inverse() * source.pose;
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() < baseline);

        target.pose.translation() = Eigen::Vector3d(0, 2.0, 0);
        srcToTgt = target.pose.inverse() * source.pose;
        corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() < baseline);
    }

    SECTION("Z rotation preserves most hits")
    {
        double angle = 10.0 * M_PI / 180.0;
        target.pose.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        srcToTgt = target.pose.inverse() * source.pose;
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() >= baseline * 0.8);
    }

    SECTION("X/Y rotation reduces hits")
    {
        double angle = 5.0 * M_PI / 180.0;

        target.pose.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
        srcToTgt = target.pose.inverse() * source.pose;
        auto corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() < baseline);

        target.pose.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()).toRotationMatrix();
        srcToTgt = target.pose.inverse() * source.pose;
        corrs = computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist);
        CHECK(corrs.size() < baseline);
    }

    SECTION("Larger rotation reduces hits progressively")
    {
        target.pose.linear() = Eigen::AngleAxisd(5.0 * M_PI / 180.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
        srcToTgt = target.pose.inverse() * source.pose;
        int hits5 = static_cast<int>(computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist).size());

        target.pose.linear() = Eigen::AngleAxisd(20.0 * M_PI / 180.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
        srcToTgt = target.pose.inverse() * source.pose;
        int hits20 = static_cast<int>(computeRayCorrespondences(source, target, rayDir, srcToTgt, maxDist).size());

        CHECK(hits5 < baseline);
        CHECK(hits20 < hits5);
    }
}
