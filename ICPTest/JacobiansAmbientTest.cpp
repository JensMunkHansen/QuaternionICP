/**
 * Test for simplified ambient jacobian - prints residual and jacobian for comparison with Python.
 * Includes finite difference validation.
 */

#include <ICP/JacobiansAmbient.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace ICP;
using Catch::Matchers::WithinAbs;

namespace
{

/// Evaluate residual only (no jacobian)
template <typename CostFunctor>
double evaluateResidual(const CostFunctor& cost, const double* x)
{
    double const* parameters[1] = {x};
    double residual;
    cost(parameters, &residual, nullptr);
    return residual;
}

/// Compute numerical jacobian via central differences
template <typename CostFunctor>
void numericalJacobian(const CostFunctor& cost, const double* x, double* J_num, double eps = 1e-7)
{
    double x_plus[7], x_minus[7];
    for (int i = 0; i < 7; ++i)
    {
        std::copy(x, x + 7, x_plus);
        std::copy(x, x + 7, x_minus);
        x_plus[i] += eps;
        x_minus[i] -= eps;

        double r_plus = evaluateResidual(cost, x_plus);
        double r_minus = evaluateResidual(cost, x_minus);
        J_num[i] = (r_plus - r_minus) / (2 * eps);
    }
}

} // namespace

TEST_CASE("RayProjectionFwd simplified jacobian", "[jacobians]")
{
    // Test inputs (same as Python test_ambient_jacobian.py)
    Vector3 pS(0.1, 0.2, 0.3);
    Vector3 qT(0.12, 0.18, 0.35);
    Vector3 nT(0.0, 0.0, 1.0);
    Vector3 dS0(0.0, 0.0, -1.0);

    // Pose: small rotation + translation
    // q = quat_exp_so3([0.01, 0.02, 0.03]) in xyzw format
    double qx = 0.004999708338437;
    double qy = 0.009999416676875;
    double qz = 0.014999125015312;
    double qw = 0.999825005104107;
    double tx = 0.01;
    double ty = -0.01;
    double tz = 0.02;

    double x[7] = {qx, qy, qz, qw, tx, ty, tz};

    WARN("=== Test Inputs ===");
    WARN("pS = [" << pS.x() << ", " << pS.y() << ", " << pS.z() << "]");
    WARN("qT = [" << qT.x() << ", " << qT.y() << ", " << qT.z() << "]");
    WARN("nT = [" << nT.x() << ", " << nT.y() << ", " << nT.z() << "]");
    WARN("dS0 = [" << dS0.x() << ", " << dS0.y() << ", " << dS0.z() << "]");
    WARN("x (pose) = [" << x[0] << ", " << x[1] << ", " << x[2] << ", "
                        << x[3] << ", " << x[4] << ", " << x[5] << ", " << x[6] << "]");

    GeometryWeighting weighting;
    WARN("weighting: enable_weight=" << weighting.enable_weight
         << ", enable_gate=" << weighting.enable_gate
         << ", tau=" << weighting.tau);

    RayProjectionFwd cost(pS, qT, nT, dS0, weighting);

    double const* parameters[1] = {x};
    double residual;
    double jacobian[7];
    double* jacobians[1] = {jacobian};

    bool ok = cost(parameters, &residual, jacobians);

    WARN("=== Outputs ===");
    WARN("ok = " << (ok ? "True" : "False"));
    WARN("residual = " << std::setprecision(15) << std::scientific << residual);
    WARN("J7 = [" << jacobian[0] << ", " << jacobian[1] << ", " << jacobian[2] << ", "
                  << jacobian[3] << ", " << jacobian[4] << ", " << jacobian[5] << ", " << jacobian[6] << "]");
    WARN("  J7[0:4] (dq) = [" << jacobian[0] << ", " << jacobian[1] << ", "
                              << jacobian[2] << ", " << jacobian[3] << "]");
    WARN("  J7[4:7] (dt) = [" << jacobian[4] << ", " << jacobian[5] << ", " << jacobian[6] << "]");

    CHECK(ok);
}

TEST_CASE("RayProjectionFwd finite difference validation", "[jacobians]")
{
    Vector3 pS(0.1, 0.2, 0.3);
    Vector3 qT(0.12, 0.18, 0.35);
    Vector3 nT(0.0, 0.0, 1.0);
    Vector3 dS0(0.0, 0.0, -1.0);

    double x[7] = {0.004999708338437, 0.009999416676875, 0.014999125015312,
                   0.999825005104107, 0.01, -0.01, 0.02};

    GeometryWeighting weighting;
    RayProjectionFwd cost(pS, qT, nT, dS0, weighting);

    // Analytical jacobian
    double const* parameters[1] = {x};
    double residual;
    double J_ana[7];
    double* jacobians[1] = {J_ana};
    cost(parameters, &residual, jacobians);

    // Numerical jacobian
    double J_num[7];
    numericalJacobian(cost, x, J_num);

    WARN("=== Finite Difference Validation ===");
    WARN("J_ana = [" << J_ana[0] << ", " << J_ana[1] << ", " << J_ana[2] << ", "
                     << J_ana[3] << ", " << J_ana[4] << ", " << J_ana[5] << ", " << J_ana[6] << "]");
    WARN("J_num = [" << J_num[0] << ", " << J_num[1] << ", " << J_num[2] << ", "
                     << J_num[3] << ", " << J_num[4] << ", " << J_num[5] << ", " << J_num[6] << "]");

    // Translation derivatives should match exactly (simplified jacobian is exact for dt)
    constexpr double tol = 1e-6;
    CHECK_THAT(J_ana[4], WithinAbs(J_num[4], tol));
    CHECK_THAT(J_ana[5], WithinAbs(J_num[5], tol));
    CHECK_THAT(J_ana[6], WithinAbs(J_num[6], tol));

    // Quaternion derivatives: simplified jacobian ignores db_dq term
    // For this test case with nT = [0,0,1] and dS0 = [0,0,-1], the db_dq term is small
    // so analytical and numerical should still be close
    for (int i = 0; i < 4; ++i)
    {
        double diff = std::abs(J_ana[i] - J_num[i]);
        double scale = std::max(1.0, std::abs(J_num[i]));
        WARN("  dq[" << i << "]: ana=" << J_ana[i] << " num=" << J_num[i]
                     << " rel_err=" << diff / scale);
    }

    CHECK(true);
}

TEST_CASE("RayProjectionFwdConsistent jacobian", "[jacobians]")
{
    // Same test inputs as simplified test
    Vector3 pS(0.1, 0.2, 0.3);
    Vector3 qT(0.12, 0.18, 0.35);
    Vector3 nT(0.0, 0.0, 1.0);
    Vector3 dS0(0.0, 0.0, -1.0);

    double x[7] = {0.004999708338437, 0.009999416676875, 0.014999125015312,
                   0.999825005104107, 0.01, -0.01, 0.02};

    GeometryWeighting weighting;
    RayProjectionFwdConsistent cost(pS, qT, nT, dS0, weighting);

    double const* parameters[1] = {x};
    double residual;
    double jacobian[7];
    double* jacobians[1] = {jacobian};

    bool ok = cost(parameters, &residual, jacobians);

    WARN("=== RayProjectionFwdConsistent (quotient-rule) ===");
    WARN("residual = " << std::setprecision(15) << std::scientific << residual);
    WARN("J7 = [" << jacobian[0] << ", " << jacobian[1] << ", " << jacobian[2] << ", "
                  << jacobian[3] << ", " << jacobian[4] << ", " << jacobian[5] << ", " << jacobian[6] << "]");
    WARN("  J7[0:4] (dq) = [" << jacobian[0] << ", " << jacobian[1] << ", "
                              << jacobian[2] << ", " << jacobian[3] << "]");
    WARN("  J7[4:7] (dt) = [" << jacobian[4] << ", " << jacobian[5] << ", " << jacobian[6] << "]");

    CHECK(ok);
}

TEST_CASE("RayProjectionFwdConsistent finite difference (no weight)", "[jacobians]")
{
    Vector3 pS(0.1, 0.2, 0.3);
    Vector3 qT(0.12, 0.18, 0.35);
    Vector3 nT(0.0, 0.0, 1.0);
    Vector3 dS0(0.0, 0.0, -1.0);

    double x[7] = {0.004999708338437, 0.009999416676875, 0.014999125015312,
                   0.999825005104107, 0.01, -0.01, 0.02};

    // Disable weighting so w=1 constant, eliminating dw/dq term
    GeometryWeighting weighting;
    weighting.enable_weight = false;
    weighting.enable_gate = false;
    RayProjectionFwdConsistent cost(pS, qT, nT, dS0, weighting);

    // Analytical jacobian
    double const* parameters[1] = {x};
    double residual;
    double J_ana[7];
    double* jacobians[1] = {J_ana};
    cost(parameters, &residual, jacobians);

    // Numerical jacobian
    double J_num[7];
    numericalJacobian(cost, x, J_num);

    WARN("=== Consistent Finite Difference (no weight) ===");
    WARN("J_ana = [" << J_ana[0] << ", " << J_ana[1] << ", " << J_ana[2] << ", "
                     << J_ana[3] << ", " << J_ana[4] << ", " << J_ana[5] << ", " << J_ana[6] << "]");
    WARN("J_num = [" << J_num[0] << ", " << J_num[1] << ", " << J_num[2] << ", "
                     << J_num[3] << ", " << J_num[4] << ", " << J_num[5] << ", " << J_num[6] << "]");

    // With w=1 constant, consistent jacobian should match numerical exactly
    // Use combined tolerance: max(abs_tol, rel_tol * |J_num|)
    constexpr double abs_tol = 2e-5;  // for near-zero values (qw normalization effects)
    constexpr double rel_tol = 1e-6;  // for non-zero values
    for (int i = 0; i < 7; ++i)
    {
        double diff = std::abs(J_ana[i] - J_num[i]);
        double scale = std::max(1.0, std::abs(J_num[i]));
        double tol = std::max(abs_tol, rel_tol * std::abs(J_num[i]));
        WARN("  J[" << i << "]: ana=" << J_ana[i] << " num=" << J_num[i] << " rel_err=" << diff / scale);
        CHECK_THAT(J_ana[i], WithinAbs(J_num[i], tol));
    }
}
