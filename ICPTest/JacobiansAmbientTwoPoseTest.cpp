/**
 * Test for two-pose ambient jacobians.
 *
 * Tests ForwardRayCostTwoPose and ReverseRayCostTwoPose cost functions
 * which take two 7D poses and produce <1, 7, 7> cost functions for Ceres.
 *
 * NOTE: These tests only verify basic functionality and print output for
 * Python comparison. FD validation is NOT done for simplified jacobians
 * as they intentionally ignore the db/dq term (see JacobiansAmbientTest.cpp).
 */

// Standard C++ headers
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// Catch2 headers
#include <catch2/catch_test_macros.hpp>

// Internal headers
#include <ICP/EigenUtils.h>
#include <ICP/JacobiansAmbientTwoPose.h>
#include <ICP/SE3.h>

using namespace ICP;

namespace
{

/// Fixed test inputs for Python comparison - with non-trivial ray direction
struct PythonTwoPoseInputs
{
    // Geometry for forward (A -> B)
    Vector3 pA{0.1, 0.2, 0.3};
    Vector3 qB{0.12, 0.18, 0.35};
    Vector3 nB{0.0, 0.0, 1.0};
    // Non-trivial ray direction: normalized [0.1, 0.2, -0.9]
    Vector3 dA0{0.107832773203438, 0.215665546406877, -0.970494958830946};

    // Pose A: quat_exp_so3([0.3, 0.2, 0.1]) in xyzw format
    double xA[7] = {0.149126529974578, 0.099417686649719, 0.049708843324859,
                    0.982550982155259, 0.01, -0.01, 0.02};

    // Pose B: quat_exp_so3([0.1, -0.2, 0.3]) in xyzw format
    double xB[7] = {0.049708843324859, -0.099417686649719, 0.149126529974578,
                    0.982550982155259, 0.02, 0.00, -0.03};
};

/// Random test geometry and two poses
struct TwoPoseTestSetup
{
    Vector3 pA, qB, nB, dA0;  // Forward: A -> B
    Vector3 pB, qA, nA, dB0;  // Reverse: B -> A
    double xA[7];
    double xB[7];
    GeometryWeighting weighting;

    explicit TwoPoseTestSetup(std::mt19937& rng, double geomScale = 0.5, double poseScale = 0.1)
    {
        // Geometry for forward (A -> B)
        pA = randomVector3(rng, geomScale);
        qB = pA + randomVector3(rng, 0.1);
        nB = randomUnitVector<3>(rng);
        dA0 = -nB + randomVector3(rng, 0.2);
        dA0.normalize();

        // Geometry for reverse (B -> A)
        pB = randomVector3(rng, geomScale);
        qA = pB + randomVector3(rng, 0.1);
        nA = randomUnitVector<3>(rng);
        dB0 = -nA + randomVector3(rng, 0.2);
        dB0.normalize();

        // Pose A
        Quaternion qApose = randomQuaternion(rng);
        Vector3 tA = randomVector3(rng, poseScale);
        xA[0] = qApose.x(); xA[1] = qApose.y(); xA[2] = qApose.z(); xA[3] = qApose.w();
        xA[4] = tA.x(); xA[5] = tA.y(); xA[6] = tA.z();

        // Pose B
        Quaternion qBpose = randomQuaternion(rng);
        Vector3 tB = randomVector3(rng, poseScale);
        xB[0] = qBpose.x(); xB[1] = qBpose.y(); xB[2] = qBpose.z(); xB[3] = qBpose.w();
        xB[4] = tB.x(); xB[5] = tB.y(); xB[6] = tB.z();

        weighting.enable_weight = false;
        weighting.enable_gate = false;
    }
};

} // namespace

TEST_CASE("ForwardRayCostTwoPose basic evaluation", "[jacobians][two-pose][forward]")
{
    std::mt19937 rng(42);
    TwoPoseTestSetup setup(rng);

    ForwardRayCostTwoPoseSimplified cost(setup.pA, setup.qB, setup.nB, setup.dA0, setup.weighting);

    // Evaluate residual and jacobians
    double const* parameters[2] = {setup.xA, setup.xB};
    double residual;
    double J7A[7], J7B[7];
    double* jacobians[2] = {J7A, J7B};

    bool ok = cost.Evaluate(parameters, &residual, jacobians);
    REQUIRE(ok);

    WARN("=== ForwardRayCostTwoPose ===");
    WARN("Residual: " << std::setprecision(9) << residual);
    WARN("J7A (dq): [" << J7A[0] << ", " << J7A[1] << ", " << J7A[2] << ", " << J7A[3] << "]");
    WARN("J7A (dt): [" << J7A[4] << ", " << J7A[5] << ", " << J7A[6] << "]");
    WARN("J7B (dq): [" << J7B[0] << ", " << J7B[1] << ", " << J7B[2] << ", " << J7B[3] << "]");
    WARN("J7B (dt): [" << J7B[4] << ", " << J7B[5] << ", " << J7B[6] << "]");

    // Basic sanity: jacobians should not all be zero
    double sumA = 0, sumB = 0;
    for (int i = 0; i < 7; ++i)
    {
        sumA += std::abs(J7A[i]);
        sumB += std::abs(J7B[i]);
    }
    CHECK(sumA > 1e-10);
    CHECK(sumB > 1e-10);
}

TEST_CASE("ReverseRayCostTwoPose basic evaluation", "[jacobians][two-pose][reverse]")
{
    std::mt19937 rng(43);
    TwoPoseTestSetup setup(rng);

    ReverseRayCostTwoPoseSimplified cost(setup.pB, setup.qA, setup.nA, setup.dB0, setup.weighting);

    // Evaluate residual and jacobians
    double const* parameters[2] = {setup.xA, setup.xB};
    double residual;
    double J7A[7], J7B[7];
    double* jacobians[2] = {J7A, J7B};

    bool ok = cost.Evaluate(parameters, &residual, jacobians);
    REQUIRE(ok);

    WARN("=== ReverseRayCostTwoPose ===");
    WARN("Residual: " << std::setprecision(9) << residual);
    WARN("J7A (dq): [" << J7A[0] << ", " << J7A[1] << ", " << J7A[2] << ", " << J7A[3] << "]");
    WARN("J7A (dt): [" << J7A[4] << ", " << J7A[5] << ", " << J7A[6] << "]");
    WARN("J7B (dq): [" << J7B[0] << ", " << J7B[1] << ", " << J7B[2] << ", " << J7B[3] << "]");
    WARN("J7B (dt): [" << J7B[4] << ", " << J7B[5] << ", " << J7B[6] << "]");

    // Basic sanity: jacobians should not all be zero
    double sumA = 0, sumB = 0;
    for (int i = 0; i < 7; ++i)
    {
        sumA += std::abs(J7A[i]);
        sumB += std::abs(J7B[i]);
    }
    CHECK(sumA > 1e-10);
    CHECK(sumB > 1e-10);
}

TEST_CASE("Two-pose residual-only evaluation", "[jacobians][two-pose]")
{
    std::mt19937 rng(44);
    TwoPoseTestSetup setup(rng);

    SECTION("Forward without jacobians")
    {
        ForwardRayCostTwoPoseSimplified cost(setup.pA, setup.qB, setup.nB, setup.dA0, setup.weighting);
        double const* parameters[2] = {setup.xA, setup.xB};
        double residual;
        bool ok = cost.Evaluate(parameters, &residual, nullptr);
        REQUIRE(ok);
        CHECK(std::isfinite(residual));
    }

    SECTION("Reverse without jacobians")
    {
        ReverseRayCostTwoPoseSimplified cost(setup.pB, setup.qA, setup.nA, setup.dB0, setup.weighting);
        double const* parameters[2] = {setup.xA, setup.xB};
        double residual;
        bool ok = cost.Evaluate(parameters, &residual, nullptr);
        REQUIRE(ok);
        CHECK(std::isfinite(residual));
    }
}

TEST_CASE("Two-pose partial jacobian evaluation", "[jacobians][two-pose]")
{
    std::mt19937 rng(45);
    TwoPoseTestSetup setup(rng);

    ForwardRayCostTwoPoseSimplified cost(setup.pA, setup.qB, setup.nB, setup.dA0, setup.weighting);

    SECTION("Only pose A jacobian")
    {
        double const* parameters[2] = {setup.xA, setup.xB};
        double residual;
        double J7A[7];
        double* jacobians[2] = {J7A, nullptr};
        bool ok = cost.Evaluate(parameters, &residual, jacobians);
        REQUIRE(ok);
    }

    SECTION("Only pose B jacobian")
    {
        double const* parameters[2] = {setup.xA, setup.xB};
        double residual;
        double J7B[7];
        double* jacobians[2] = {nullptr, J7B};
        bool ok = cost.Evaluate(parameters, &residual, jacobians);
        REQUIRE(ok);
    }
}

TEST_CASE("Two-pose intermediate values debug", "[jacobians][two-pose][debug]")
{
    PythonTwoPoseInputs inputs;

    Quaternion qA(inputs.xA[3], inputs.xA[0], inputs.xA[1], inputs.xA[2]);
    qA.normalize();
    Quaternion qB(inputs.xB[3], inputs.xB[0], inputs.xB[1], inputs.xB[2]);
    qB.normalize();

    Matrix3 RB = qB.toRotationMatrix();

    Matrix3x4 dRApA_dqA = dRv_dq(qA, inputs.pA);
    Matrix3x4 dx_dqA = RB.transpose() * dRApA_dqA;

    WARN("=== Debug Intermediate Values ===");
    WARN("qA = [" << qA.x() << ", " << qA.y() << ", " << qA.z() << ", " << qA.w() << "]");
    WARN("dRApA_dqA col 3 (dw): [" << std::setprecision(15) << dRApA_dqA(0,3) << ", "
                                   << dRApA_dqA(1,3) << ", " << dRApA_dqA(2,3) << "]");
    WARN("dx_dqA col 3 (dw): [" << std::setprecision(15) << dx_dqA(0,3) << ", "
                                << dx_dqA(1,3) << ", " << dx_dqA(2,3) << "]");

    Vector3 nB(0.0, 0.0, 1.0);
    Eigen::RowVector4d da_dqA = nB.transpose() * dx_dqA;
    WARN("da_dqA = [" << da_dqA(0) << ", " << da_dqA(1) << ", " << da_dqA(2) << ", " << da_dqA(3) << "]");
    WARN("da_dqA[3] = " << std::scientific << da_dqA(3));
}

TEST_CASE("Two-pose jacobians for Python comparison", "[jacobians][two-pose][python]")
{
    PythonTwoPoseInputs inputs;
    GeometryWeighting weighting;
    weighting.enable_weight = false;
    weighting.enable_gate = false;

    WARN("=== Two-Pose Test Inputs ===");
    WARN("pA = [" << inputs.pA.x() << ", " << inputs.pA.y() << ", " << inputs.pA.z() << "]");
    WARN("qB = [" << inputs.qB.x() << ", " << inputs.qB.y() << ", " << inputs.qB.z() << "]");
    WARN("nB = [" << inputs.nB.x() << ", " << inputs.nB.y() << ", " << inputs.nB.z() << "]");
    WARN("dA0 = [" << inputs.dA0.x() << ", " << inputs.dA0.y() << ", " << inputs.dA0.z() << "]");
    WARN("xA (pose A) = [" << inputs.xA[0] << ", " << inputs.xA[1] << ", " << inputs.xA[2] << ", "
                          << inputs.xA[3] << ", " << inputs.xA[4] << ", " << inputs.xA[5] << ", " << inputs.xA[6] << "]");
    WARN("xB (pose B) = [" << inputs.xB[0] << ", " << inputs.xB[1] << ", " << inputs.xB[2] << ", "
                          << inputs.xB[3] << ", " << inputs.xB[4] << ", " << inputs.xB[5] << ", " << inputs.xB[6] << "]");

    SECTION("ForwardRayCostTwoPose Simplified")
    {
        ForwardRayCostTwoPoseSimplified cost(inputs.pA, inputs.qB, inputs.nB, inputs.dA0, weighting);

        double const* parameters[2] = {inputs.xA, inputs.xB};
        double residual;
        double J7A[7], J7B[7];
        double* jacobians[2] = {J7A, J7B};

        bool ok = cost.Evaluate(parameters, &residual, jacobians);
        CHECK(ok);

        WARN("=== ForwardRayCostTwoPose Simplified ===");
        WARN("residual = " << std::setprecision(15) << std::scientific << residual);
        WARN("J7A = [" << std::setprecision(9) << J7A[0] << ", " << J7A[1] << ", " << J7A[2] << ", "
                      << J7A[3] << ", " << J7A[4] << ", " << J7A[5] << ", " << J7A[6] << "]");
        WARN("  J7A[0:4] (dqA) = [" << J7A[0] << ", " << J7A[1] << ", " << J7A[2] << ", " << J7A[3] << "]");
        WARN("  J7A[4:7] (dtA) = [" << J7A[4] << ", " << J7A[5] << ", " << J7A[6] << "]");
        WARN("J7B = [" << std::setprecision(9) << J7B[0] << ", " << J7B[1] << ", " << J7B[2] << ", "
                      << J7B[3] << ", " << J7B[4] << ", " << J7B[5] << ", " << J7B[6] << "]");
        WARN("  J7B[0:4] (dqB) = [" << J7B[0] << ", " << J7B[1] << ", " << J7B[2] << ", " << J7B[3] << "]");
        WARN("  J7B[4:7] (dtB) = [" << J7B[4] << ", " << J7B[5] << ", " << J7B[6] << "]");

        // Print values for manual comparison with Python
        // Compare these with: python test_two_pose_jacobians.py
    }
}
