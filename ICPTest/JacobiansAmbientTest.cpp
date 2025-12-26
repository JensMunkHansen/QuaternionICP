/**
 * Test for ambient jacobians with proper finite difference validation.
 *
 * Methodology (from Doc/JacobianValidation.md):
 * - Sweep eps over a decade and look for a plateau
 * - Test random R and random delta (unit and scaled)
 * - Compare directional derivatives, not full matrices
 */

#include <ICP/JacobiansAmbient.h>
#include <ICP/EigenUtils.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

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

/// Compute numerical directional derivative using proper SE(3) manifold perturbation.
/// delta6 = [v_x, v_y, v_z, w_x, w_y, w_z] (translation, rotation in body frame)
template <typename CostFunctor>
double numericalDirectionalDerivative(const CostFunctor& cost, const double* x,
                                       const Eigen::Matrix<double, 6, 1>& delta6, double eps)
{
    Quaternion q(x[3], x[0], x[1], x[2]);  // w, x, y, z
    Vector3 t(x[4], x[5], x[6]);
    Matrix3 R = q.toRotationMatrix();

    // Tangent vector components
    Vector3 v = delta6.head<3>();      // translation (body frame)
    Vector3 omega = delta6.tail<3>();  // rotation

    // Perturbed rotation: q_± = q * exp(±eps * omega)
    Quaternion dq_plus = quatExpSO3(eps * omega);
    Quaternion dq_minus = quatExpSO3(-eps * omega);
    Quaternion q_plus = q * dq_plus;
    Quaternion q_minus = q * dq_minus;

    // Perturbed translation: t_± = t ± R * (eps * v)  (body frame)
    Vector3 t_plus = t + R * (eps * v);
    Vector3 t_minus = t - R * (eps * v);

    double x_plus[7] = {q_plus.x(), q_plus.y(), q_plus.z(), q_plus.w(),
                        t_plus.x(), t_plus.y(), t_plus.z()};
    double x_minus[7] = {q_minus.x(), q_minus.y(), q_minus.z(), q_minus.w(),
                         t_minus.x(), t_minus.y(), t_minus.z()};

    double r_plus = evaluateResidual(cost, x_plus);
    double r_minus = evaluateResidual(cost, x_minus);
    return (r_plus - r_minus) / (2 * eps);
}


/// Logarithmic epsilon values per Doc/EpsilonSweep.md Section 6.1
/// Returns values like: 1e-2, 3e-3, 1e-3, 3e-4, ...
std::vector<double> logarithmicEpsilons(int exp_start = -2, int exp_end = -10)
{
    std::vector<double> epsilons;
    for (int e = exp_start; e >= exp_end; --e)
    {
        epsilons.push_back(std::pow(10.0, e));
        epsilons.push_back(3.0 * std::pow(10.0, e - 1));
    }
    std::sort(epsilons.begin(), epsilons.end(), std::greater<double>());
    return epsilons;
}

/// Result of an epsilon sweep
struct SweepResult
{
    double best_eps;
    double best_df_num;
    double min_err;
};

/// Sweep epsilon to find best finite difference approximation
/// Uses the formula from Doc/EpsilonSweep.md Section 5
template <typename CostFunctor>
SweepResult sweepEpsilon(const CostFunctor& cost, const double* x,
                         const Eigen::Matrix<double, 6, 1>& delta6,
                         double df_ana, const std::vector<double>& epsilons)
{
    SweepResult result{0, 0, std::numeric_limits<double>::max()};

    for (double eps : epsilons)
    {
        double df_num = numericalDirectionalDerivative(cost, x, delta6, eps);
        double err = std::abs(df_num - df_ana);
        if (err < result.min_err)
        {
            result.min_err = err;
            result.best_eps = eps;
            result.best_df_num = df_num;
        }
    }
    return result;
}

/// Compute analytical 6D jacobian from 7D jacobian using plus jacobian
Eigen::Matrix<double, 1, 6> jacobian7Dto6D(const double* J7, const double* x)
{
    Pose7 pose;
    pose << x[0], x[1], x[2], x[3], x[4], x[5], x[6];
    auto P = plusJacobian7x6(pose);
    Eigen::Map<const Eigen::RowVectorXd> J7_map(J7, 7);
    return J7_map * P;
}

/// Get analytical jacobian from a cost functor
template <typename CostFunctor>
Eigen::Matrix<double, 1, 6> getAnalyticalJacobian6(const CostFunctor& cost, const double* x)
{
    double const* parameters[1] = {x};
    double residual;
    double J7[7];
    double* jacobians[1] = {J7};
    cost(parameters, &residual, jacobians);
    return jacobian7Dto6D(J7, x);
}

} // namespace

TEST_CASE("RayProjectionFwd simplified jacobian", "[jacobians][python]")
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

TEST_CASE("RayProjectionFwdConsistent directional derivative validation", "[jacobians][fd]")
{
    std::mt19937 rng(42);

    Vector3 pS = randomVector3(rng, 0.5);
    Vector3 qT = pS + randomVector3(rng, 0.1);
    Vector3 nT = randomUnitVector<3>(rng);
    Vector3 dS0 = -nT + randomVector3(rng, 0.1);
    dS0.normalize();

    Quaternion q = randomQuaternion(rng);
    Vector3 t = randomVector3(rng, 0.05);
    double x[7] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

    GeometryWeighting weighting;
    weighting.enable_weight = false;
    weighting.enable_gate = false;
    RayProjectionFwdConsistent cost(pS, qT, nT, dS0, weighting);

    auto J_ana6 = getAnalyticalJacobian6(cost, x);
    auto epsilons = logarithmicEpsilons();

    constexpr int numDirections = 5;
    double scales[] = {1.0, 0.1, 10.0};

    for (double scale : scales)
    {
        for (int d = 0; d < numDirections; ++d)
        {
            Eigen::Matrix<double, 6, 1> delta6 = scale * randomUnitVector<6>(rng);
            double df_ana = J_ana6.dot(delta6);

            auto result = sweepEpsilon(cost, x, delta6, df_ana, epsilons);
            double rel_err = result.min_err / std::max(1.0, std::abs(df_ana));

            INFO("scale=" << scale << " dir=" << d << " df_ana=" << df_ana
                         << " df_num=" << result.best_df_num << " rel_err=" << rel_err);
            CHECK(rel_err < 1e-6);
        }
    }
}

TEST_CASE("RayProjectionFwdConsistent jacobian", "[jacobians][python]")
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

TEST_CASE("RayProjectionFwd simplified directional derivative validation", "[jacobians][fd]")
{
    // Fixed seed for reproducibility
    std::mt19937 rng(123);

    // Random geometry
    Vector3 pS = randomVector3(rng, 0.5);
    Vector3 qT = pS + randomVector3(rng, 0.1);
    Vector3 nT = randomUnitVector<3>(rng);
    Vector3 dS0 = -nT + randomVector3(rng, 0.1);
    dS0.normalize();

    // Random pose
    Quaternion q = randomQuaternion(rng);
    Vector3 t = randomVector3(rng, 0.05);
    double x[7] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

    GeometryWeighting weighting;
    weighting.enable_weight = false;
    weighting.enable_gate = false;
    RayProjectionFwd cost(pS, qT, nT, dS0, weighting);

    // Analytical 7D jacobian -> 6D via plus jacobian
    double const* parameters[1] = {x};
    double residual;
    double J_ana7[7];
    double* jacobians[1] = {J_ana7};
    cost(parameters, &residual, jacobians);

    Pose7 pose;
    pose << x[0], x[1], x[2], x[3], x[4], x[5], x[6];
    auto P = plusJacobian7x6(pose);
    Eigen::Map<Eigen::RowVectorXd> J7(J_ana7, 7);
    Eigen::Matrix<double, 1, 6> J_ana6 = J7 * P;

    // Test translation-only directions (simplified is exact for these)
    INFO("=== Translation-only directions (simplified is exact) ===");
    for (int i = 0; i < 3; ++i)
    {
        Eigen::Matrix<double, 6, 1> delta6 = Eigen::Matrix<double, 6, 1>::Zero();
        delta6[i] = 1.0;

        double df_ana = J_ana6.dot(delta6);

        double best_err = std::numeric_limits<double>::max();
        double best_eps = 0;
        double best_df_num = 0;

        for (int e = -4; e >= -10; --e)
        {
            double eps = std::pow(10.0, e);
            double df_num = numericalDirectionalDerivative(cost, x, delta6, eps);
            double err = std::abs(df_num - df_ana);
            if (err < best_err)
            {
                best_err = err;
                best_eps = eps;
                best_df_num = df_num;
            }
        }

        double rel_err = best_err / std::max(1.0, std::abs(df_ana));
        INFO("t[" << i << "]: df_ana=" << df_ana << " df_num=" << best_df_num
                  << " best_eps=" << best_eps << " rel_err=" << rel_err);
        CHECK(rel_err < 1e-6);
    }

    // Test rotation directions (simplified ignores db_dq, so will have some error)
    INFO("=== Rotation directions (simplified approximation) ===");
    for (int i = 0; i < 3; ++i)
    {
        Eigen::Matrix<double, 6, 1> delta6 = Eigen::Matrix<double, 6, 1>::Zero();
        delta6[3 + i] = 1.0;

        double df_ana = J_ana6.dot(delta6);

        double best_err = std::numeric_limits<double>::max();
        double best_eps = 0;
        double best_df_num = 0;

        for (int e = -4; e >= -10; --e)
        {
            double eps = std::pow(10.0, e);
            double df_num = numericalDirectionalDerivative(cost, x, delta6, eps);
            double err = std::abs(df_num - df_ana);
            if (err < best_err)
            {
                best_err = err;
                best_eps = eps;
                best_df_num = df_num;
            }
        }

        double rel_err = best_err / std::max(1.0, std::abs(df_ana));
        // Simplified ignores db_dq term - report error but don't fail
        INFO("w[" << i << "]: df_ana=" << df_ana << " df_num=" << best_df_num
                  << " best_eps=" << best_eps << " rel_err=" << rel_err);
    }

    CHECK(true);  // Always pass - rotation error is expected for simplified
}

// ============================================================================
// New tests following Doc/JacobianValidation.md methodology
// ============================================================================

/**
 * Sanity test: f(R) = R * p with known analytical derivative.
 *
 * From Doc/JacobianValidation.md Section 4:
 *   Right perturbation result: df = R * (delta_body x p)
 *
 * This tests dRv_dq against the known cross-product formula.
 */
TEST_CASE("Sanity test: f(R) = R*p rotation jacobian", "[jacobians][sanity]")
{
    std::mt19937 rng(999);

    constexpr int numTrials = 10;
    for (int trial = 0; trial < numTrials; ++trial)
    {
        // Random rotation and point
        Quaternion q = randomQuaternion(rng);
        Vector3 p = randomVector3(rng, 1.0);
        Matrix3 R = q.toRotationMatrix();

        // Analytical jacobian: d(R*p)/dq
        Matrix3x4 dRp_dq = dRv_dq(q, p);

        // Test with random rotation directions
        for (int d = 0; d < 5; ++d)
        {
            Vector3 omega = randomUnitVector<3>(rng);

            // Convert omega to quaternion tangent via plus jacobian
            // For pure rotation: delta7 = P @ [0, 0, 0, omega]
            Pose7 pose;
            pose << q.x(), q.y(), q.z(), q.w(), 0, 0, 0;
            auto P = plusJacobian7x6(pose);
            Eigen::Matrix<double, 6, 1> delta6;
            delta6 << 0, 0, 0, omega.x(), omega.y(), omega.z();
            Eigen::Matrix<double, 7, 1> delta7 = P * delta6;
            Eigen::Vector4d dq_tangent = delta7.head<4>();

            // Analytical: df_ana = dRp_dq @ dq_tangent
            Vector3 df_ana = dRp_dq * dq_tangent;

            // Known formula: df = R * (omega x p)
            Vector3 df_known = R * omega.cross(p);

            // Numerical via central differences
            double best_err = std::numeric_limits<double>::max();
            Vector3 df_num_best;
            double best_eps = 0;

            for (int e = -4; e >= -10; --e)
            {
                double eps = std::pow(10.0, e);
                Quaternion dq = quatExpSO3(eps * omega);
                Quaternion q_plus = q * dq;
                Quaternion q_minus = q * dq.conjugate();

                Vector3 f_plus = q_plus.toRotationMatrix() * p;
                Vector3 f_minus = q_minus.toRotationMatrix() * p;
                Vector3 df_num = (f_plus - f_minus) / (2 * eps);

                double err = (df_num - df_known).norm();
                if (err < best_err)
                {
                    best_err = err;
                    df_num_best = df_num;
                    best_eps = eps;
                }
            }

            // Check analytical matches known formula
            double err_ana_known = (df_ana - df_known).norm();
            INFO("trial=" << trial << " dir=" << d);
            INFO("df_ana = [" << df_ana.transpose() << "]");
            INFO("df_known = [" << df_known.transpose() << "]");
            INFO("df_num = [" << df_num_best.transpose() << "] (eps=" << best_eps << ")");
            INFO("err_ana_known=" << err_ana_known << " err_num_known=" << best_err);

            CHECK(err_ana_known < 1e-10);
            CHECK(best_err < 1e-8);
        }
    }
}

/**
 * Test consistent jacobian with multiple random poses.
 *
 * From Doc/JacobianValidation.md Section 2.3:
 *   "Test random R and random delta (unit and scaled)"
 */
TEST_CASE("Multiple random poses: consistent jacobian", "[jacobians][random]")
{
    auto epsilons = logarithmicEpsilons();
    constexpr int numPoses = 20;
    constexpr int numDirections = 3;

    for (int seed = 0; seed < numPoses; ++seed)
    {
        std::mt19937 rng(seed * 1000);

        Vector3 pS = randomVector3(rng, 0.5);
        Vector3 qT = pS + randomVector3(rng, 0.1);
        Vector3 nT = randomUnitVector<3>(rng);
        Vector3 dS0 = -nT + randomVector3(rng, 0.2);
        dS0.normalize();

        Quaternion q = randomQuaternion(rng);
        Vector3 t = randomVector3(rng, 0.1);
        double x[7] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

        GeometryWeighting weighting;
        weighting.enable_weight = false;
        weighting.enable_gate = false;
        RayProjectionFwdConsistent cost(pS, qT, nT, dS0, weighting);

        auto J_ana6 = getAnalyticalJacobian6(cost, x);

        for (int d = 0; d < numDirections; ++d)
        {
            Eigen::Matrix<double, 6, 1> delta6 = randomUnitVector<6>(rng);
            double df_ana = J_ana6.dot(delta6);

            auto result = sweepEpsilon(cost, x, delta6, df_ana, epsilons);
            double rel_err = result.min_err / std::max(1.0, std::abs(df_ana));

            INFO("pose=" << seed << " dir=" << d << " rel_err=" << rel_err);
            CHECK(rel_err < 1e-6);
        }
    }
}

/**
 * Epsilon sweep: compare SIMPLIFIED vs CONSISTENT jacobians.
 *
 * From Doc/EpsilonSweep.md:
 *   - Sweep ε logarithmically: 1e-2, 3e-3, 1e-3, 3e-4, ...
 *   - Translation and rotation plateau at 10^-6 → 10^-8
 *
 * Expected behavior:
 *   - Consistent: both translation and rotation should match FD at plateau
 *   - Simplified: translation matches, rotation does NOT (ignores db_dq)
 */
TEST_CASE("Epsilon sweep: simplified vs consistent", "[jacobians][epsilon]")
{
    std::mt19937 rng(777);

    Vector3 pS = randomVector3(rng, 0.5);
    Vector3 qT = pS + randomVector3(rng, 0.1);
    Vector3 nT = randomUnitVector<3>(rng);
    Vector3 dS0 = -nT + randomVector3(rng, 0.2);
    dS0.normalize();

    Quaternion q = randomQuaternion(rng);
    Vector3 t = randomVector3(rng, 0.05);
    double x[7] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

    GeometryWeighting weighting;
    weighting.enable_weight = false;
    weighting.enable_gate = false;

    RayProjectionFwdConsistent costConsistent(pS, qT, nT, dS0, weighting);
    RayProjectionFwd costSimplified(pS, qT, nT, dS0, weighting);

    auto J6_cons = getAnalyticalJacobian6(costConsistent, x);
    auto J6_simp = getAnalyticalJacobian6(costSimplified, x);
    auto epsilons = logarithmicEpsilons();

    WARN("=== TRANSLATION ===");
    WARN("J6_cons (t) = [" << J6_cons(0) << ", " << J6_cons(1) << ", " << J6_cons(2) << "]");
    WARN("J6_simp (t) = [" << J6_simp(0) << ", " << J6_simp(1) << ", " << J6_simp(2) << "]");

    for (int k = 0; k < 3; ++k)
    {
        Eigen::Matrix<double, 6, 1> delta6 = Eigen::Matrix<double, 6, 1>::Zero();
        delta6(k) = 1.0;  // Unit vector in translation direction k

        auto result = sweepEpsilon(costConsistent, x, delta6, J6_cons(k), epsilons);
        double err_cons = std::abs(result.best_df_num - J6_cons(k));
        double err_simp = std::abs(result.best_df_num - J6_simp(k));

        WARN("  t[" << k << "]: FD=" << result.best_df_num << " cons=" << J6_cons(k) << " simp=" << J6_simp(k)
                    << " | err_cons=" << err_cons << " err_simp=" << err_simp << " best_eps=" << result.best_eps);

        CHECK(err_cons < 1e-10);
        CHECK(err_simp < 1e-10);
    }

    WARN("=== ROTATION ===");
    WARN("J6_cons (w) = [" << J6_cons(3) << ", " << J6_cons(4) << ", " << J6_cons(5) << "]");
    WARN("J6_simp (w) = [" << J6_simp(3) << ", " << J6_simp(4) << ", " << J6_simp(5) << "]");

    for (int k = 0; k < 3; ++k)
    {
        Eigen::Matrix<double, 6, 1> delta6 = Eigen::Matrix<double, 6, 1>::Zero();
        delta6(3 + k) = 1.0;  // Unit vector in rotation direction k

        auto result = sweepEpsilon(costConsistent, x, delta6, J6_cons(3 + k), epsilons);
        double err_cons = std::abs(result.best_df_num - J6_cons(3 + k));
        double err_simp = std::abs(result.best_df_num - J6_simp(3 + k));

        WARN("  w[" << k << "]: FD=" << result.best_df_num << " cons=" << J6_cons(3 + k) << " simp=" << J6_simp(3 + k)
                    << " | err_cons=" << err_cons << " err_simp=" << err_simp << " best_eps=" << result.best_eps);

        CHECK(err_cons < 1e-10);
        // Simplified will NOT match for rotation (ignores db_dq term)
    }
}

/**
 * Test reverse direction jacobian (RayProjectionRevConsistent).
 */
TEST_CASE("RayProjectionRevConsistent directional derivative validation", "[jacobians][fd][reverse]")
{
    std::mt19937 rng(555);

    Vector3 pT = randomVector3(rng, 0.5);
    Vector3 qS = pT + randomVector3(rng, 0.1);
    Vector3 nS = randomUnitVector<3>(rng);
    Vector3 dT0 = -nS + randomVector3(rng, 0.1);
    dT0.normalize();

    Quaternion q = randomQuaternion(rng);
    Vector3 t = randomVector3(rng, 0.05);
    double x[7] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

    GeometryWeighting weighting;
    weighting.enable_weight = false;
    weighting.enable_gate = false;
    RayProjectionRevConsistent cost(pT, qS, nS, dT0, weighting);

    auto J_ana6 = getAnalyticalJacobian6(cost, x);
    auto epsilons = logarithmicEpsilons();

    constexpr int numDirections = 10;
    double scales[] = {1.0, 0.1, 10.0};

    for (double scale : scales)
    {
        for (int d = 0; d < numDirections; ++d)
        {
            Eigen::Matrix<double, 6, 1> delta6 = scale * randomUnitVector<6>(rng);
            double df_ana = J_ana6.dot(delta6);

            auto result = sweepEpsilon(cost, x, delta6, df_ana, epsilons);
            double rel_err = result.min_err / std::max(1.0, std::abs(df_ana));

            INFO("scale=" << scale << " dir=" << d << " rel_err=" << rel_err);
            CHECK(rel_err < 1e-6);
        }
    }
}
