#include <ICP/SE3.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>

using namespace ICP;
using Catch::Matchers::WithinAbs;

// Tolerance for floating point comparisons
constexpr double TOL = 1e-10;

TEST_CASE("skew matrix", "[se3]")
{
    Eigen::Vector3d w(1.0, 2.0, 3.0);
    Eigen::Matrix3d S = skew(w);

    // Expected from Python:
    // [[ 0. -3.  2.]
    //  [ 3.  0. -1.]
    //  [-2.  1.  0.]]
    CHECK_THAT(S(0, 0), WithinAbs(0.0, TOL));
    CHECK_THAT(S(0, 1), WithinAbs(-3.0, TOL));
    CHECK_THAT(S(0, 2), WithinAbs(2.0, TOL));
    CHECK_THAT(S(1, 0), WithinAbs(3.0, TOL));
    CHECK_THAT(S(1, 1), WithinAbs(0.0, TOL));
    CHECK_THAT(S(1, 2), WithinAbs(-1.0, TOL));
    CHECK_THAT(S(2, 0), WithinAbs(-2.0, TOL));
    CHECK_THAT(S(2, 1), WithinAbs(1.0, TOL));
    CHECK_THAT(S(2, 2), WithinAbs(0.0, TOL));

    // Also verify skew-symmetric property: S = -S^T
    CHECK((S + S.transpose()).norm() < TOL);

    // And cross product property: S * v = w x v
    Eigen::Vector3d v(4.0, 5.0, 6.0);
    Eigen::Vector3d cross_result = S * v;
    Eigen::Vector3d expected_cross = w.cross(v);
    CHECK((cross_result - expected_cross).norm() < TOL);
}
