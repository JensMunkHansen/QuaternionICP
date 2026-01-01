// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * Test for ambient jacobians with proper finite difference validation.
 *
 * Methodology (from Doc/JacobianValidation.md):
 * - Sweep eps over a decade and look for a plateau
 * - Test random R and random delta (unit and scaled)
 * - Compare directional derivatives, not full matrices
 */

// Standard C++ headers
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

// Catch2 headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Internal headers
#include <ICP/EigenUtils.h>
#include <ICP/JacobiansAmbient.h>

using namespace ICP;
using Catch::Matchers::WithinAbs;

namespace
{

// ============================================================================
// FD validation traits - test infrastructure only
// ============================================================================

template<typename Policy>
struct FDValidationTraits;

// ============================================================================
// Finite-Difference Validation Parameters (see Doc/*.md)
// ============================================================================
//
// Empirical values from epsilon sweep:
//
// TRANSLATION (linear in t, see Doc/ExpectedEpsilon.md):
//   - FD is exact for any eps (no truncation error)
//   - best_eps: 1e-2 (arbitrary, all work)
//   - tol: 1e-14 (machine precision)
//
// ROTATION (nonlinear in R, see Doc/ExpectedEpsilon.md):
//   - FD requires proper eps sweep
//   - best_eps: 3e-6 (plateau range)
//   - Consistent tol: 1e-10 (matches FD)
//   - Simplified: DO NOT validate against FD (see below)
//
// SIMPLIFIED ROTATION (see Doc/SimplifiedError.md):
//   The Simplified jacobian ignores the db/dq term: dr/dq = da/dq / b
//   The missing term (-a * db/dq / b²) can dominate for random geometry:
//   - Expected: ~1% error for typical ICP geometry (small residual a)
//   - Observed: up to 60% for random geometry (large |a * db/dq|)
//   This is mathematically correct - the approximation breaks down when
//   |a * db/dq| >> |da/dq * b|.
//
//   VALIDATION STRATEGY:
//   - Do NOT validate Simplified rotation against FD (tolerance is meaningless)
//   - Instead, "[diagnostic]" test validates: (Consistent - Simplified) == missing term
//   - This confirms both implementations are correct
// ============================================================================

template<>
struct FDValidationTraits<RayJacobianSimplified>
{
    static constexpr double translation_eps = 1e-2;   // Any eps works (linear)
    static constexpr double rotation_eps = 3e-6;      // Plateau at 3e-6
    static constexpr double translation_tol = 2e-14;  // Near machine precision
    // Rotation: not validated against FD - see "[diagnostic]" test
    static constexpr double rotation_tol = std::numeric_limits<double>::max();
};

template<>
struct FDValidationTraits<RayJacobianConsistent>
{
    static constexpr double translation_eps = 1e-2;   // Any eps works (linear)
    static constexpr double rotation_eps = 3e-6;      // Plateau at 3e-6
    static constexpr double translation_tol = 2e-14;  // Near machine precision
    static constexpr double rotation_tol = 1e-10;     // Matches FD (sweep: ~3e-11)
};

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

/// Result of jacobian validation
struct ValidationResult
{
    bool passed = true;
    double worst_err = 0.0;
    double worst_tol = 0.0;
    std::string worst_desc;

    void check(double err, double tol, const std::string& desc)
    {
        if (err >= tol && err > worst_err)
        {
            passed = false;
            worst_err = err;
            worst_tol = tol;
            worst_desc = desc;
        }
    }
};

/// Validate jacobian using FDValidationTraits for the cost's policy
template <typename CostFunctor>
void validateJacobianWithPolicy(const CostFunctor& cost, const double* x, std::mt19937& rng)
{
    using Policy = typename CostFunctor::policy_tag;
    using Traits = FDValidationTraits<Policy>;

    auto J_ana6 = getAnalyticalJacobian6(cost, x);
    ValidationResult result;

    // Test translation directions
    for (int k = 0; k < 3; ++k)
    {
        Eigen::Matrix<double, 6, 1> delta6 = Eigen::Matrix<double, 6, 1>::Zero();
        delta6(k) = 1.0;
        double df_ana = J_ana6.dot(delta6);
        double df_num = numericalDirectionalDerivative(cost, x, delta6, Traits::translation_eps);
        double err = std::abs(df_num - df_ana);

        std::ostringstream oss;
        oss << "Translation[" << k << "]: ana=" << df_ana << " num=" << df_num << " err=" << err;
        result.check(err, Traits::translation_tol, oss.str());
    }

    // Test rotation directions
    for (int k = 0; k < 3; ++k)
    {
        Eigen::Matrix<double, 6, 1> delta6 = Eigen::Matrix<double, 6, 1>::Zero();
        delta6(3 + k) = 1.0;
        double df_ana = J_ana6.dot(delta6);
        double df_num = numericalDirectionalDerivative(cost, x, delta6, Traits::rotation_eps);
        double err = std::abs(df_num - df_ana);

        std::ostringstream oss;
        oss << "Rotation[" << k << "]: ana=" << df_ana << " num=" << df_num << " err=" << err;
        result.check(err, Traits::rotation_tol, oss.str());
    }

    // Test random mixed directions
    for (int d = 0; d < 3; ++d)
    {
        Eigen::Matrix<double, 6, 1> delta6 = randomUnitVector<6>(rng);
        double df_ana = J_ana6.dot(delta6);
        double df_num = numericalDirectionalDerivative(cost, x, delta6, Traits::rotation_eps);
        double err = std::abs(df_num - df_ana);

        std::ostringstream oss;
        oss << "Mixed[" << d << "]: ana=" << df_ana << " num=" << df_num << " err=" << err;
        result.check(err, Traits::rotation_tol, oss.str());
    }

    INFO("Worst failure: " << result.worst_desc << " (tol=" << result.worst_tol << ")");
    CHECK(result.passed);
}

/// Validate jacobian with epsilon sweep over multiple scales and directions
template <typename CostFunctor>
void validateWithEpsilonSweep(const CostFunctor& cost, const double* x, std::mt19937& rng,
                               int numDirections = 5, double tolerance = 1e-6)
{
    auto J_ana6 = getAnalyticalJacobian6(cost, x);
    auto epsilons = logarithmicEpsilons();
    double scales[] = {1.0, 0.1, 10.0};

    ValidationResult validation;

    for (double scale : scales)
    {
        for (int d = 0; d < numDirections; ++d)
        {
            Eigen::Matrix<double, 6, 1> delta6 = scale * randomUnitVector<6>(rng);
            double df_ana = J_ana6.dot(delta6);

            auto result = sweepEpsilon(cost, x, delta6, df_ana, epsilons);
            double rel_err = result.min_err / std::max(1.0, std::abs(df_ana));

            std::ostringstream oss;
            oss << "scale=" << scale << " dir=" << d << " df_ana=" << df_ana
                << " df_num=" << result.best_df_num << " rel_err=" << rel_err;
            validation.check(rel_err, tolerance, oss.str());
        }
    }

    INFO("Worst failure: " << validation.worst_desc << " (tol=" << validation.worst_tol << ")");
    CHECK(validation.passed);
}

/// Print jacobian output for Python comparison
template <typename CostFunctor>
void printJacobianForPython(const CostFunctor& cost, const double* x)
{
    using Policy = typename CostFunctor::policy_tag;

    double const* parameters[1] = {x};
    double residual;
    double jacobian[7];
    double* jacobians[1] = {jacobian};

    bool ok = cost(parameters, &residual, jacobians);

    WARN("=== " << Policy::name << " jacobian ===");
    WARN("residual = " << std::setprecision(15) << std::scientific << residual);
    WARN("J7 = [" << std::setprecision(9) << jacobian[0] << ", " << jacobian[1] << ", " << jacobian[2] << ", "
                  << jacobian[3] << ", " << jacobian[4] << ", " << jacobian[5] << ", " << jacobian[6] << "]");
    WARN("  J7[0:4] (dq) = [" << jacobian[0] << ", " << jacobian[1] << ", "
                              << jacobian[2] << ", " << jacobian[3] << "]");
    WARN("  J7[4:7] (dt) = [" << jacobian[4] << ", " << jacobian[5] << ", " << jacobian[6] << "]");

    CHECK(ok);
}

/// Common test inputs for Python comparison tests
struct PythonTestInputs
{
    Vector3 pS{0.1, 0.2, 0.3};
    Vector3 qT{0.12, 0.18, 0.35};
    Vector3 nT{0.0, 0.0, 1.0};
    Vector3 dS0{0.0, 0.0, -1.0};

    // Pose: q = quat_exp_so3([0.01, 0.02, 0.03]) in xyzw format
    double x[7] = {0.004999708338437, 0.009999416676875, 0.014999125015312,
                   0.999825005104107, 0.01, -0.01, 0.02};
};

/// Random test geometry and pose for FD validation
struct RandomTestSetup
{
    Vector3 pS, qT, nT, dS0;
    double x[7];
    GeometryWeighting weighting;

    explicit RandomTestSetup(std::mt19937& rng, double geomScale = 0.5, double poseScale = 0.1)
    {
        pS = randomVector3(rng, geomScale);
        qT = pS + randomVector3(rng, 0.1);
        nT = randomUnitVector<3>(rng);
        dS0 = -nT + randomVector3(rng, 0.2);
        dS0.normalize();

        Quaternion q = randomQuaternion(rng);
        Vector3 t = randomVector3(rng, poseScale);
        x[0] = q.x(); x[1] = q.y(); x[2] = q.z(); x[3] = q.w();
        x[4] = t.x(); x[5] = t.y(); x[6] = t.z();

        weighting.enable_weight = false;
        weighting.enable_gate = false;
    }
};

} // namespace

TEST_CASE("Ray jacobians for Python comparison", "[jacobians][python]")
{
    PythonTestInputs inputs;
    GeometryWeighting weighting;

    WARN("=== Test Inputs ===");
    WARN("pS = [" << inputs.pS.x() << ", " << inputs.pS.y() << ", " << inputs.pS.z() << "]");
    WARN("qT = [" << inputs.qT.x() << ", " << inputs.qT.y() << ", " << inputs.qT.z() << "]");
    WARN("nT = [" << inputs.nT.x() << ", " << inputs.nT.y() << ", " << inputs.nT.z() << "]");
    WARN("dS0 = [" << inputs.dS0.x() << ", " << inputs.dS0.y() << ", " << inputs.dS0.z() << "]");
    WARN("x (pose) = [" << inputs.x[0] << ", " << inputs.x[1] << ", " << inputs.x[2] << ", "
                        << inputs.x[3] << ", " << inputs.x[4] << ", " << inputs.x[5] << ", " << inputs.x[6] << "]");
    WARN("weighting: enable_weight=" << weighting.enable_weight
         << ", enable_gate=" << weighting.enable_gate
         << ", tau=" << weighting.tau);

    SECTION("Simplified")
    {
        ForwardRayCostSimplified cost(inputs.pS, inputs.qT, inputs.nT, inputs.dS0, weighting);
        printJacobianForPython(cost, inputs.x);
    }

    SECTION("Consistent")
    {
        ForwardRayCostConsistent cost(inputs.pS, inputs.qT, inputs.nT, inputs.dS0, weighting);
        printJacobianForPython(cost, inputs.x);
    }
}

TEST_CASE("Simplified jacobian: translation exact, rotation approximate", "[jacobians][fd]")
{
    std::mt19937 rng(RandomSeeds::JACOBIAN_SIMPLIFIED_TEST);
    RandomTestSetup setup(rng, 0.5, 0.05);

    ForwardRayCostSimplified cost(setup.pS, setup.qT, setup.nT, setup.dS0, setup.weighting);

    auto J_ana6 = getAnalyticalJacobian6(cost, setup.x);
    auto epsilons = logarithmicEpsilons();

    ValidationResult validation;

    // Translation directions: simplified is exact
    for (int i = 0; i < 3; ++i)
    {
        Eigen::Matrix<double, 6, 1> delta6 = Eigen::Matrix<double, 6, 1>::Zero();
        delta6[i] = 1.0;
        double df_ana = J_ana6.dot(delta6);

        auto result = sweepEpsilon(cost, setup.x, delta6, df_ana, epsilons);
        double rel_err = result.min_err / std::max(1.0, std::abs(df_ana));

        std::ostringstream oss;
        oss << "t[" << i << "]: df_ana=" << df_ana << " df_num=" << result.best_df_num
            << " best_eps=" << result.best_eps << " rel_err=" << rel_err;
        validation.check(rel_err, 1e-6, oss.str());
    }

    // Rotation directions: simplified ignores db_dq, so has error (don't validate)

    INFO("Worst failure: " << validation.worst_desc << " (tol=" << validation.worst_tol << ")");
    CHECK(validation.passed);
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
    std::mt19937 rng(RandomSeeds::JACOBIAN_SANITY_TEST);

    ValidationResult validation;

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

            std::ostringstream oss;
            oss << "trial=" << trial << " dir=" << d << " err_ana_known=" << err_ana_known;
            validation.check(err_ana_known, 1e-10, oss.str());

            oss.str("");
            oss << "trial=" << trial << " dir=" << d << " err_num_known=" << best_err;
            validation.check(best_err, 1e-8, oss.str());
        }
    }

    INFO("Worst failure: " << validation.worst_desc << " (tol=" << validation.worst_tol << ")");
    CHECK(validation.passed);
}

/**
 * Test consistent jacobian with multiple random poses.
 *
 * From Doc/JacobianValidation.md Section 2.3:
 *   "Test random R and random delta (unit and scaled)"
 */
TEST_CASE("Multiple random poses: consistent jacobian", "[jacobians][random]")
{
    constexpr int numPoses = 20;

    ValidationResult validation;

    for (int seed = 0; seed < numPoses; ++seed)
    {
        std::mt19937 rng(RandomSeeds::JACOBIAN_MULTI_POSE_BASE + seed * 1000);
        RandomTestSetup setup(rng);

        ForwardRayCostConsistent cost(setup.pS, setup.qT, setup.nT, setup.dS0, setup.weighting);

        auto J_ana6 = getAnalyticalJacobian6(cost, setup.x);
        auto epsilons = logarithmicEpsilons();
        double scales[] = {1.0, 0.1, 10.0};

        for (double scale : scales)
        {
            for (int d = 0; d < 3; ++d)
            {
                Eigen::Matrix<double, 6, 1> delta6 = scale * randomUnitVector<6>(rng);
                double df_ana = J_ana6.dot(delta6);

                auto result = sweepEpsilon(cost, setup.x, delta6, df_ana, epsilons);
                double rel_err = result.min_err / std::max(1.0, std::abs(df_ana));

                std::ostringstream oss;
                oss << "seed=" << seed << " scale=" << scale << " dir=" << d << " rel_err=" << rel_err;
                validation.check(rel_err, 1e-6, oss.str());
            }
        }
    }

    INFO("Worst failure: " << validation.worst_desc << " (tol=" << validation.worst_tol << ")");
    CHECK(validation.passed);
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
    std::mt19937 rng(RandomSeeds::JACOBIAN_EPSILON_TEST);
    RandomTestSetup setup(rng, 0.5, 0.05);

    ForwardRayCostConsistent costConsistent(setup.pS, setup.qT, setup.nT, setup.dS0, setup.weighting);
    ForwardRayCostSimplified costSimplified(setup.pS, setup.qT, setup.nT, setup.dS0, setup.weighting);

    auto J6_cons = getAnalyticalJacobian6(costConsistent, setup.x);
    auto J6_simp = getAnalyticalJacobian6(costSimplified, setup.x);
    auto epsilons = logarithmicEpsilons();

    ValidationResult validation;

    WARN("=== TRANSLATION ===");
    WARN("J6_cons (t) = [" << J6_cons(0) << ", " << J6_cons(1) << ", " << J6_cons(2) << "]");
    WARN("J6_simp (t) = [" << J6_simp(0) << ", " << J6_simp(1) << ", " << J6_simp(2) << "]");

    for (int k = 0; k < 3; ++k)
    {
        Eigen::Matrix<double, 6, 1> delta6 = Eigen::Matrix<double, 6, 1>::Zero();
        delta6(k) = 1.0;  // Unit vector in translation direction k

        auto result = sweepEpsilon(costConsistent, setup.x, delta6, J6_cons(k), epsilons);
        double err_cons = std::abs(result.best_df_num - J6_cons(k));
        double err_simp = std::abs(result.best_df_num - J6_simp(k));

        WARN("  t[" << k << "]: FD=" << result.best_df_num << " cons=" << J6_cons(k) << " simp=" << J6_simp(k)
                    << " | err_cons=" << err_cons << " err_simp=" << err_simp << " best_eps=" << result.best_eps);

        std::ostringstream oss;
        oss << "t[" << k << "] cons err=" << err_cons;
        validation.check(err_cons, 1e-10, oss.str());

        oss.str("");
        oss << "t[" << k << "] simp err=" << err_simp;
        validation.check(err_simp, 1e-10, oss.str());
    }

    WARN("=== ROTATION ===");
    WARN("J6_cons (w) = [" << J6_cons(3) << ", " << J6_cons(4) << ", " << J6_cons(5) << "]");
    WARN("J6_simp (w) = [" << J6_simp(3) << ", " << J6_simp(4) << ", " << J6_simp(5) << "]");

    for (int k = 0; k < 3; ++k)
    {
        Eigen::Matrix<double, 6, 1> delta6 = Eigen::Matrix<double, 6, 1>::Zero();
        delta6(3 + k) = 1.0;  // Unit vector in rotation direction k

        auto result = sweepEpsilon(costConsistent, setup.x, delta6, J6_cons(3 + k), epsilons);
        double err_cons = std::abs(result.best_df_num - J6_cons(3 + k));
        double err_simp = std::abs(result.best_df_num - J6_simp(3 + k));

        WARN("  w[" << k << "]: FD=" << result.best_df_num << " cons=" << J6_cons(3 + k) << " simp=" << J6_simp(3 + k)
                    << " | err_cons=" << err_cons << " err_simp=" << err_simp << " best_eps=" << result.best_eps);

        std::ostringstream oss;
        oss << "w[" << k << "] cons err=" << err_cons;
        validation.check(err_cons, 1e-10, oss.str());
        // Simplified will NOT match for rotation (ignores db_dq term)
    }

    INFO("Worst failure: " << validation.worst_desc << " (tol=" << validation.worst_tol << ")");
    CHECK(validation.passed);
}

/**
 * Policy-based validation using JacobianTraits.
 *
 * Uses the policy_tag from each cost functor to automatically
 * select appropriate epsilon and tolerance values.
 */
TEST_CASE("Policy-based jacobian validation", "[jacobians][policy]")
{
    std::mt19937 rng(RandomSeeds::JACOBIAN_POLICY_TEST);
    RandomTestSetup setup(rng);

    SECTION("ForwardRayCost<RayJacobianSimplified>")
    {
        ForwardRayCostSimplified cost(setup.pS, setup.qT, setup.nT, setup.dS0, setup.weighting);
        static_assert(std::is_same_v<decltype(cost)::policy_tag, RayJacobianSimplified>);
        validateJacobianWithPolicy(cost, setup.x, rng);
    }

    SECTION("ForwardRayCost<RayJacobianConsistent>")
    {
        ForwardRayCostConsistent cost(setup.pS, setup.qT, setup.nT, setup.dS0, setup.weighting);
        static_assert(std::is_same_v<decltype(cost)::policy_tag, RayJacobianConsistent>);
        validateJacobianWithPolicy(cost, setup.x, rng);
    }

    SECTION("ReverseRayCost<RayJacobianSimplified>")
    {
        ReverseRayCostSimplified cost(setup.qT, setup.pS, setup.nT, setup.dS0, setup.weighting);
        static_assert(std::is_same_v<decltype(cost)::policy_tag, RayJacobianSimplified>);
        validateJacobianWithPolicy(cost, setup.x, rng);
    }

    SECTION("ReverseRayCost<RayJacobianConsistent>")
    {
        ReverseRayCostConsistent cost(setup.qT, setup.pS, setup.nT, setup.dS0, setup.weighting);
        static_assert(std::is_same_v<decltype(cost)::policy_tag, RayJacobianConsistent>);
        validateJacobianWithPolicy(cost, setup.x, rng);
    }
}

/**
 * Test reverse direction jacobian with epsilon sweep.
 */
TEST_CASE("Reverse consistent jacobian epsilon sweep", "[jacobians][fd][reverse]")
{
    std::mt19937 rng(RandomSeeds::JACOBIAN_REVERSE_TEST);
    RandomTestSetup setup(rng, 0.5, 0.05);

    ReverseRayCostConsistent cost(setup.pS, setup.qT, setup.nT, setup.dS0, setup.weighting);
    validateWithEpsilonSweep(cost, setup.x, rng, 10);
}

/**
 * Diagnostic: Verify Simplified error equals the missing db_dq term.
 *
 * With weight w = weighting.weight(b):
 *   Simplified:  dr/dq = w * da_dq / b
 *   Consistent:  dr/dq = w * (da_dq * b - a * db_dq) / b²
 *
 * Therefore:   Consistent - Simplified = -w * a * db_dq / b²
 *
 * This test verifies the difference matches the expected missing term.
 */
TEST_CASE("Simplified error equals missing db_dq term", "[jacobians][diagnostic]")
{
    std::mt19937 rng(RandomSeeds::JACOBIAN_EPSILON_TEST);
    RandomTestSetup setup(rng, 0.5, 0.05);

    // Extract pose
    Quaternion q(setup.x[3], setup.x[0], setup.x[1], setup.x[2]);
    q.normalize();
    Vector3 t(setup.x[4], setup.x[5], setup.x[6]);
    Matrix3 R = q.toRotationMatrix();

    // Compute intermediate values
    Vector3 xT = R * setup.pS + t;
    Vector3 d = R * setup.dS0;
    double a = setup.nT.dot(xT - setup.qT);
    double b = setup.nT.dot(d);

    // Get weight (must match what the cost functions use)
    GeometryWeighting weighting;
    double w = weighting.weight(b);

    // Compute db_dq = nT^T * dRd_dq
    Matrix3x4 dRd_dq = dRv_dq(q, setup.dS0);
    Eigen::RowVector4d db_dq = setup.nT.transpose() * dRd_dq;

    // Expected missing term: -w * a * db_dq / b²
    // The jacobians include w, so the difference must also include w
    Eigen::RowVector4d missing_term = -w * a * db_dq / (b * b);
    ForwardRayCostSimplified costSimp(setup.pS, setup.qT, setup.nT, setup.dS0, weighting);
    ForwardRayCostConsistent costCons(setup.pS, setup.qT, setup.nT, setup.dS0, weighting);

    double const* parameters[1] = {setup.x};
    double residual;
    double J7_simp[7], J7_cons[7];
    double* jac_simp[1] = {J7_simp};
    double* jac_cons[1] = {J7_cons};

    costSimp(parameters, &residual, jac_simp);
    costCons(parameters, &residual, jac_cons);

    double q_norm = std::sqrt(setup.x[0]*setup.x[0] + setup.x[1]*setup.x[1] +
                               setup.x[2]*setup.x[2] + setup.x[3]*setup.x[3]);

    WARN("=== Diagnostic: Simplified error analysis ===");
    WARN("q_norm (input) = " << std::setprecision(15) << q_norm);
    WARN("a (signed distance) = " << a);
    WARN("b (ray-normal dot)  = " << b);
    WARN("w (weight)          = " << w);
    WARN("a/b ratio = " << std::abs(a/b));

    WARN("=== Quaternion jacobian differences (rotation part only) ===");

    ValidationResult validation;

    for (int i = 0; i < 4; ++i)
    {
        double diff = J7_cons[i] - J7_simp[i];
        double expected = missing_term(i);
        double match_err = std::abs(diff - expected);
        double rel_err = std::abs(expected) > 1e-10 ? match_err / std::abs(expected) : match_err;

        WARN("  q[" << i << "]: cons=" << J7_cons[i] << " simp=" << J7_simp[i]
                    << " diff=" << diff << " expected=" << expected
                    << " match_err=" << match_err << " rel_err=" << rel_err);

        std::ostringstream oss;
        oss << "q[" << i << "]: rel_err=" << rel_err;
        validation.check(rel_err, 1e-12, oss.str());
    }

    // Translation jacobian should be identical
    for (int i = 4; i < 7; ++i)
    {
        double diff = std::abs(J7_cons[i] - J7_simp[i]);
        std::ostringstream oss;
        oss << "t[" << i-4 << "]: diff=" << diff;
        validation.check(diff, 1e-15, oss.str());
    }

    INFO("Worst failure: " << validation.worst_desc << " (tol=" << validation.worst_tol << ")");
    CHECK(validation.passed);
}
