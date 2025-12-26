/**
 * Test for simplified ambient jacobian - prints residual and jacobian for comparison with Python.
 */

#include <ICP/JacobiansAmbient.h>

#include <catch2/catch_test_macros.hpp>

#include <iomanip>
#include <iostream>

using namespace ICP;

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
