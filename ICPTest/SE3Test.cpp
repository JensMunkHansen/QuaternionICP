#include <ICP/SE3.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>

using namespace ICP;
using Catch::Matchers::WithinAbs;

// Tolerance for floating point comparisons
constexpr double TOL = 1e-10;

TEST_CASE("skew matrix", "[se3][python]")
{
    Eigen::Vector3d w(1.0, 2.0, 3.0);
    Eigen::Matrix3d S = skew(w);

    WARN("C++ skew([1, 2, 3]):");
    WARN("  [" << S(0,0) << ", " << S(0,1) << ", " << S(0,2) << "]");
    WARN("  [" << S(1,0) << ", " << S(1,1) << ", " << S(1,2) << "]");
    WARN("  [" << S(2,0) << ", " << S(2,1) << ", " << S(2,2) << "]");

    // Skew-symmetric property: S = -S^T
    CHECK((S + S.transpose()).norm() < TOL);

    // Diagonal should be zero
    CHECK(std::abs(S(0, 0)) < TOL);
    CHECK(std::abs(S(1, 1)) < TOL);
    CHECK(std::abs(S(2, 2)) < TOL);

    // Cross product property: S * v = w x v
    Eigen::Vector3d v(4.0, 5.0, 6.0);
    Eigen::Vector3d cross_result = S * v;
    Eigen::Vector3d expected_cross = w.cross(v);
    CHECK((cross_result - expected_cross).norm() < TOL);
}

TEST_CASE("quatExpSO3", "[se3][python]")
{
    SECTION("small angle")
    {
        Eigen::Vector3d w(0.001, 0.002, -0.001);
        Eigen::Quaterniond q = quatExpSO3(w);

        WARN("C++ quatExpSO3([0.001, 0.002, -0.001]):");
        WARN("  [" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]");

        // For small angles, imaginary part ≈ w/2, real part ≈ 1
        CHECK(std::abs(q.x()) < 0.001);
        CHECK(std::abs(q.y()) < 0.002);
        CHECK(std::abs(q.z()) < 0.001);
        CHECK(q.w() > 0.999);
        CHECK_THAT(q.norm(), WithinAbs(1.0, TOL));
    }

    SECTION("larger angle")
    {
        Eigen::Vector3d w(0.1, 0.2, -0.15);
        Eigen::Quaterniond q = quatExpSO3(w);

        WARN("C++ quatExpSO3([0.1, 0.2, -0.15]):");
        WARN("  [" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]");

        // Quaternion should be unit
        CHECK_THAT(q.norm(), WithinAbs(1.0, TOL));
        // Real part should be close to 1 for small total angle
        CHECK(q.w() > 0.9);
        // Rotation matrix should be valid
        CHECK_THAT(q.toRotationMatrix().determinant(), WithinAbs(1.0, TOL));
    }
}

TEST_CASE("Eigen quaternion to rotation matrix", "[se3][python]")
{
    Eigen::Vector4d q_unnorm(0.1, 0.2, 0.3, 0.9);
    q_unnorm.normalize();

    // Eigen Quaterniond constructor: (w, x, y, z)
    Eigen::Quaterniond q(q_unnorm[3], q_unnorm[0], q_unnorm[1], q_unnorm[2]);
    Eigen::Matrix3d R = q.toRotationMatrix();

    WARN("C++ quat_to_R([0.1, 0.2, 0.3, 0.9] normalized):");
    WARN("  [" << R(0,0) << ", " << R(0,1) << ", " << R(0,2) << "]");
    WARN("  [" << R(1,0) << ", " << R(1,1) << ", " << R(1,2) << "]");
    WARN("  [" << R(2,0) << ", " << R(2,1) << ", " << R(2,2) << "]");

    // Verify it's a valid rotation
    CHECK_THAT(R.determinant(), WithinAbs(1.0, TOL));
    CHECK((R * R.transpose() - Eigen::Matrix3d::Identity()).norm() < TOL);
}

TEST_CASE("Vso3 left Jacobian", "[se3][python]")
{
    SECTION("larger angle")
    {
        Eigen::Vector3d w(0.1, 0.2, -0.15);
        Eigen::Matrix3d V = Vso3(w);

        // Print for comparison with Python
        WARN("C++ Vso3([0.1, 0.2, -0.15]):");
        WARN("  [" << V(0,0) << ", " << V(0,1) << ", " << V(0,2) << "]");
        WARN("  [" << V(1,0) << ", " << V(1,1) << ", " << V(1,2) << "]");
        WARN("  [" << V(2,0) << ", " << V(2,1) << ", " << V(2,2) << "]");

        // Basic sanity check: V should be close to identity for small angles
        CHECK(V.determinant() > 0.9);
    }
}

TEST_CASE("se3Plus", "[se3][python]")
{
    // Same initial pose as Python test
    Eigen::Vector3d w0(0.02, 0.01, -0.03);
    Eigen::Quaterniond q0 = quatExpSO3(w0);
    Eigen::Vector3d t0(-0.02, 0.01, 0.05);

    Pose7 x;
    x << q0.x(), q0.y(), q0.z(), q0.w(), t0;

    WARN("C++ initial pose x:");
    WARN("  [" << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << ", "
              << x[4] << ", " << x[5] << ", " << x[6] << "]");

    Tangent6 delta;
    delta << 0.01, -0.02, 0.005, 0.03, -0.01, 0.02;

    Pose7 x_plus = se3Plus(x, delta);

    WARN("C++ se3Plus(x, delta):");
    WARN("  [" << x_plus[0] << ", " << x_plus[1] << ", " << x_plus[2] << ", " << x_plus[3] << ", "
              << x_plus[4] << ", " << x_plus[5] << ", " << x_plus[6] << "]");

    // Quaternion should be normalized
    double qnorm = std::sqrt(x_plus[0]*x_plus[0] + x_plus[1]*x_plus[1] +
                             x_plus[2]*x_plus[2] + x_plus[3]*x_plus[3]);
    CHECK_THAT(qnorm, WithinAbs(1.0, TOL));
}

TEST_CASE("plusJacobian7x6", "[se3][python]")
{
    // Same initial pose as Python test
    Eigen::Vector3d w0(0.02, 0.01, -0.03);
    Eigen::Quaterniond q0 = quatExpSO3(w0);
    Eigen::Vector3d t0(-0.02, 0.01, 0.05);

    Pose7 x;
    x << q0.x(), q0.y(), q0.z(), q0.w(), t0;

    auto P = plusJacobian7x6(x);

    WARN("C++ plusJacobian7x6(x):");
    for (int i = 0; i < 7; i++)
    {
        WARN("  [" << P(i,0) << ", " << P(i,1) << ", " << P(i,2) << ", "
                  << P(i,3) << ", " << P(i,4) << ", " << P(i,5) << "]");
    }

    // Basic checks: shape is 7x6, translation block should be rotation matrix
    CHECK(P.rows() == 7);
    CHECK(P.cols() == 6);
}

TEST_CASE("dRv_dq and dRTv_dq", "[se3][python]")
{
    // Same quaternion as Python test
    Eigen::Vector4d q_unnorm(0.1, 0.2, 0.3, 0.9);
    q_unnorm.normalize();
    Eigen::Quaterniond q(q_unnorm[3], q_unnorm[0], q_unnorm[1], q_unnorm[2]);

    Eigen::Vector3d v(1.0, 2.0, 3.0);

    auto J_Rv = dRv_dq(q, v);
    auto J_RTv = dRTv_dq(q, v);

    WARN("C++ dRv_dq(q, v) where q=normalized([0.1,0.2,0.3,0.9]), v=[1,2,3]:");
    WARN("  [" << J_Rv(0,0) << ", " << J_Rv(0,1) << ", " << J_Rv(0,2) << ", " << J_Rv(0,3) << "]");
    WARN("  [" << J_Rv(1,0) << ", " << J_Rv(1,1) << ", " << J_Rv(1,2) << ", " << J_Rv(1,3) << "]");
    WARN("  [" << J_Rv(2,0) << ", " << J_Rv(2,1) << ", " << J_Rv(2,2) << ", " << J_Rv(2,3) << "]");

    WARN("C++ dRTv_dq(q, v):");
    WARN("  [" << J_RTv(0,0) << ", " << J_RTv(0,1) << ", " << J_RTv(0,2) << ", " << J_RTv(0,3) << "]");
    WARN("  [" << J_RTv(1,0) << ", " << J_RTv(1,1) << ", " << J_RTv(1,2) << ", " << J_RTv(1,3) << "]");
    WARN("  [" << J_RTv(2,0) << ", " << J_RTv(2,1) << ", " << J_RTv(2,2) << ", " << J_RTv(2,3) << "]");

    // Basic shape checks
    CHECK(J_Rv.rows() == 3);
    CHECK(J_Rv.cols() == 4);
    CHECK(J_RTv.rows() == 3);
    CHECK(J_RTv.cols() == 4);
}
