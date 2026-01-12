// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

/**
 * Test for tangent (6D) Jacobians and their relationship to ambient (7D) Jacobians.
 *
 * Both the tangent Jacobians (JacobiansTangent.h) and the Simplified ambient
 * Jacobians (JacobiansAmbient.h) ignore the quotient rule - they don't account
 * for how the denominator (n^T d) depends on the rotation.
 *
 * The tangent Jacobians use LEFT perturbation (space/fixed frame):
 *   T_new = exp(delta_xi_hat) * T
 *
 * The ambient Jacobians use RIGHT perturbation (body/moving frame):
 *   T_new = T * exp(delta_xi_hat)
 *
 * From Doc/JacobianValidation.md, Section 3:
 *   J_right(R) = J_left(R) * R
 *   J_left(R) = J_right(R) * R^T
 *
 * For full SE(3), the relationship involves the adjoint:
 *   J_left = J_right * Ad_T^{-1}
 *
 * where Ad_T^{-1} = [R^T  -R^T [t]_x; 0  R^T] for T = (R, t).
 *
 * This test validates the relationship between tangent and projected ambient
 * Jacobians (both using the Simplified approximation).
 */

#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <ICP/EigenUtils.h>
#include <ICP/JacobiansAmbient.h>
#include <ICP/JacobiansTangent.h>
#include <ICP/SE3.h>

using namespace ICP;
using Catch::Matchers::WithinAbs;

namespace
{

// ============================================================================
// Test helpers
// ============================================================================

/// GeometryWeighting with weighting disabled for fair comparison
inline GeometryWeighting noWeighting()
{
    GeometryWeighting w;
    w.enable_weight = false;
    w.enable_gate = false;
    return w;
}

/// Random test geometry and pose
struct RandomTestSetup
{
    // Geometry
    Eigen::Vector3f srcPoint;   // p in source frame
    Eigen::Vector3f tgtPoint;   // q in target frame
    Eigen::Vector3f tgtNormal;  // n in target frame
    Eigen::Vector3f rayDirTarget;  // d in target frame (for tangent Jacobian)
    Eigen::Vector3f rayDirSource;  // dS0 in source frame (for ambient Jacobian)

    // Pose as (R, t)
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    // Pose as 7D ambient parameter [qx, qy, qz, qw, tx, ty, tz]
    double x[7];

    explicit RandomTestSetup(std::mt19937& rng, double geomScale = 0.5, double poseScale = 0.1)
    {
        srcPoint = randomVector3(rng, geomScale).cast<float>();
        tgtPoint = srcPoint.cast<double>().cast<float>() + randomVector3(rng, 0.1).cast<float>();
        tgtNormal = randomUnitVector<3>(rng).cast<float>();

        Quaternion q = randomQuaternion(rng);
        t = randomVector3(rng, poseScale);
        R = q.toRotationMatrix();

        // Ray direction in SOURCE frame (for ambient Jacobian)
        Eigen::Vector3d rayDir_src = -tgtNormal.cast<double>() + randomVector3(rng, 0.2);
        rayDir_src.normalize();
        rayDirSource = rayDir_src.cast<float>();

        // Ray direction in TARGET frame: d = R * dS0 (for tangent Jacobian)
        Eigen::Vector3d rayDir_tgt = R * rayDir_src;
        rayDirTarget = rayDir_tgt.cast<float>();

        x[0] = q.x();
        x[1] = q.y();
        x[2] = q.z();
        x[3] = q.w();
        x[4] = t.x();
        x[5] = t.y();
        x[6] = t.z();
    }
};

/// Compute tangent Jacobian using JacobiansTangent.h
/// Uses rayDirTarget (ray in target frame)
Eigen::Matrix<double, 1, 6> computeTangentJacobian(const RandomTestSetup& setup,
    const GeometryWeighting& weighting = noWeighting())
{
    Eigen::Matrix<double, 1, 6> J;
    computeForwardJacobian(setup.srcPoint, setup.tgtPoint, setup.tgtNormal,
                           setup.R, setup.t, setup.rayDirTarget, J, weighting);
    return J;
}

/// Compute tangent reverse Jacobian using JacobiansTangent.h
/// Uses rayDirSource (ray in source frame, for reverse direction)
Eigen::Matrix<double, 1, 6> computeTangentReverseJacobian(const RandomTestSetup& setup,
    const GeometryWeighting& weighting = noWeighting())
{
    Eigen::Matrix<double, 1, 6> J;
    computeReverseJacobian(setup.srcPoint, setup.tgtPoint, setup.tgtNormal,
                           setup.R, setup.t, setup.rayDirSource, J, weighting);
    return J;
}

/// Compute ambient Jacobian (Simplified) and project to 6D using SE3.h
/// Uses rayDirSource (dS0 in source frame, transformed internally as d = R * dS0)
Eigen::Matrix<double, 1, 6> computeAmbientJacobian6D(const RandomTestSetup& setup,
    const GeometryWeighting& weighting = noWeighting())
{
    ForwardRayCostSimplified cost(
        setup.srcPoint.cast<double>(),
        setup.tgtPoint.cast<double>(),
        setup.tgtNormal.cast<double>(),
        setup.rayDirSource.cast<double>(),
        weighting);

    double const* parameters[1] = {setup.x};
    double residual;
    double J7[7];
    double* jacobians[1] = {J7};

    cost.Evaluate(parameters, &residual, jacobians);

    // Project 7D -> 6D using the plus Jacobian (right perturbation)
    return jacobian7Dto6D(J7, setup.x);
}

/// Compute ambient reverse Jacobian (Simplified) and project to 6D
/// Uses rayDirTarget (dT0 in target frame, for reverse direction)
Eigen::Matrix<double, 1, 6> computeAmbientReverseJacobian6D(const RandomTestSetup& setup,
    const GeometryWeighting& weighting = noWeighting())
{
    ReverseRayCostSimplified cost(
        setup.srcPoint.cast<double>(),
        setup.tgtPoint.cast<double>(),
        setup.tgtNormal.cast<double>(),
        setup.rayDirTarget.cast<double>(),
        weighting);

    double const* parameters[1] = {setup.x};
    double residual;
    double J7[7];
    double* jacobians[1] = {J7};

    cost.Evaluate(parameters, &residual, jacobians);

    return jacobian7Dto6D(J7, setup.x);
}

/// Build the SE(3) inverse adjoint transformation matrix
/// Ad_T^{-1} = [R^T  -R^T [t]_x; 0  R^T]
/// This maps tangent vectors from body frame to space frame
Eigen::Matrix<double, 6, 6> buildInverseAdjoint(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
    Eigen::Matrix<double, 6, 6> AdInv = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix3d Rt = R.transpose();
    Eigen::Matrix3d t_skew = skew(t);

    // Upper-left: R^T (translation block)
    AdInv.block<3, 3>(0, 0) = Rt;

    // Upper-right: -R^T [t]_x (coupling from rotation to translation)
    AdInv.block<3, 3>(0, 3) = -Rt * t_skew;

    // Lower-right: R^T (rotation block)
    AdInv.block<3, 3>(3, 3) = Rt;

    return AdInv;
}

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

}  // namespace

// ============================================================================
// Tests
// ============================================================================

/**
 * Verify relationship between left (tangent) and right (ambient) Jacobians.
 *
 * From Doc/JacobianValidation.md:
 *   J_left = J_right * Ad_T^{-1}
 *
 * where Ad_T^{-1} = [R^T  -R^T[t]_x; 0  R^T] maps body-frame tangents to space-frame.
 *
 * Both Jacobians use the Simplified approximation (ignoring db/dphi), so they
 * should match exactly when transformed via the adjoint.
 */
TEST_CASE("Tangent vs Ambient: adjoint relationship", "[jacobians][tangent][ambient]")
{
    std::mt19937 rng(RandomSeeds::JACOBIAN_POLICY_TEST);

    constexpr int numTrials = 20;
    ValidationResult validation;

    for (int trial = 0; trial < numTrials; ++trial)
    {
        RandomTestSetup setup(rng);

        // Get tangent Jacobian (left perturbation, space frame)
        auto J_tangent = computeTangentJacobian(setup);

        // Get ambient Jacobian projected to 6D (right perturbation, body frame)
        auto J_ambient_6D = computeAmbientJacobian6D(setup);

        // Build inverse adjoint: Ad_T^{-1}
        auto AdInv = buildInverseAdjoint(setup.R, setup.t);

        // Predicted: J_tangent = J_ambient_6D * Ad_T^{-1}
        Eigen::Matrix<double, 1, 6> J_predicted = J_ambient_6D * AdInv;

        // Compare
        double err = (J_tangent - J_predicted).norm();
        double scale = std::max(J_tangent.norm(), 1.0);
        double rel_err = err / scale;

        std::ostringstream oss;
        oss << "trial=" << trial << " rel_err=" << rel_err;
        validation.check(rel_err, 1e-6, oss.str());  // Float precision tolerance

        if (trial == 0)
        {
            WARN("=== Sample Jacobian comparison (trial 0) ===");
            WARN("J_tangent    = [" << J_tangent << "]");
            WARN("J_ambient_6D = [" << J_ambient_6D << "]");
            WARN("J_predicted  = [" << J_predicted << "]");
            WARN("rel_err = " << rel_err);
        }
    }

    INFO("Worst failure: " << validation.worst_desc << " (tol=" << validation.worst_tol << ")");
    CHECK(validation.passed);
}

/**
 * Component-wise analysis of the adjoint relationship.
 *
 * The inverse adjoint Ad_T^{-1} has the structure:
 *   [R^T     -R^T[t]_x]
 *   [0       R^T      ]
 *
 * When computing J_left = J_right * Ad_T^{-1} (row vector × matrix):
 *   J_left[trans] = J_right[trans] * R^T
 *   J_left[rot]   = J_right[rot] * R^T - J_right[trans] * R^T * [t]_x
 */
TEST_CASE("Tangent vs Ambient: component analysis", "[jacobians][tangent][components]")
{
    std::mt19937 rng(42);
    RandomTestSetup setup(rng);

    auto J_tangent = computeTangentJacobian(setup);
    auto J_ambient_6D = computeAmbientJacobian6D(setup);
    Eigen::Matrix3d Rt = setup.R.transpose();
    Eigen::Matrix3d t_skew = skew(setup.t);

    // Extract components
    Eigen::RowVector3d J_tangent_trans = J_tangent.head<3>();
    Eigen::RowVector3d J_tangent_rot = J_tangent.tail<3>();
    Eigen::RowVector3d J_ambient_trans = J_ambient_6D.head<3>();
    Eigen::RowVector3d J_ambient_rot = J_ambient_6D.tail<3>();

    // Predicted relationships (row vector × matrix multiplication)
    Eigen::RowVector3d pred_trans = J_ambient_trans * Rt;
    Eigen::RowVector3d pred_rot = J_ambient_rot * Rt - J_ambient_trans * (Rt * t_skew);

    WARN("=== Component-wise comparison ===");
    WARN("J_tangent[trans]  = [" << J_tangent_trans << "]");
    WARN("pred[trans]       = [" << pred_trans << "]");
    WARN("error[trans]      = " << (J_tangent_trans - pred_trans).norm());

    WARN("J_tangent[rot]    = [" << J_tangent_rot << "]");
    WARN("pred[rot]         = [" << pred_rot << "]");
    WARN("error[rot]        = " << (J_tangent_rot - pred_rot).norm());

    CHECK((J_tangent_trans - pred_trans).norm() < 1e-6);  // Float precision tolerance
    CHECK((J_tangent_rot - pred_rot).norm() < 1e-6);    // Float precision tolerance
}

/**
 * Debug test to verify matching computation.
 */
TEST_CASE("Debug: verify matching computation", "[jacobians][tangent][debug]")
{
    std::mt19937 rng(42);
    RandomTestSetup setup(rng);

    WARN("=== Debug computation ===");
    WARN("srcPoint = [" << setup.srcPoint.transpose() << "]");
    WARN("tgtPoint = [" << setup.tgtPoint.transpose() << "]");
    WARN("tgtNormal = [" << setup.tgtNormal.transpose() << "]");
    WARN("rayDirSource = [" << setup.rayDirSource.transpose() << "]");
    WARN("rayDirTarget = [" << setup.rayDirTarget.transpose() << "]");
    WARN("R = \n" << setup.R);
    WARN("t = [" << setup.t.transpose() << "]");

    // Compute intermediate values manually
    Eigen::Vector3d p = setup.srcPoint.cast<double>();
    Eigen::Vector3d x_tangent = setup.R * p + setup.t;
    WARN("x (transformed point) = [" << x_tangent.transpose() << "]");

    Eigen::Vector3d n = setup.tgtNormal.cast<double>();
    Eigen::Vector3d q = setup.tgtPoint.cast<double>();

    // For tangent: use rayDirTarget directly
    Eigen::Vector3d d_tangent = setup.rayDirTarget.cast<double>();
    double b_tangent = n.dot(d_tangent);
    double a_tangent = n.dot(x_tangent - q);
    double r_tangent = a_tangent / b_tangent;

    WARN("Tangent: d = [" << d_tangent.transpose() << "], b = " << b_tangent << ", a = " << a_tangent << ", r = " << r_tangent);

    // For ambient: transform rayDirSource
    Eigen::Vector3d dS0 = setup.rayDirSource.cast<double>();
    Eigen::Vector3d d_ambient = setup.R * dS0;
    double b_ambient = n.dot(d_ambient);
    double a_ambient = n.dot(x_tangent - q);  // Same x
    double r_ambient = a_ambient / b_ambient;

    WARN("Ambient: dS0 = [" << dS0.transpose() << "]");
    WARN("Ambient: d = R*dS0 = [" << d_ambient.transpose() << "], b = " << b_ambient << ", a = " << a_ambient << ", r = " << r_ambient);

    // Verify d is the same (relax tolerance due to float precision)
    WARN("d_tangent - d_ambient = [" << (d_tangent - d_ambient).transpose() << "], norm = " << (d_tangent - d_ambient).norm());

    CHECK((d_tangent - d_ambient).norm() < 1e-6);
    CHECK(std::abs(r_tangent - r_ambient) < 1e-6);

    // Now check actual function calls (use noWeighting for fair comparison)
    Eigen::Matrix<double, 1, 6> J_tangent;
    double r_func_tangent = computeForwardJacobian(setup.srcPoint, setup.tgtPoint, setup.tgtNormal,
                                                   setup.R, setup.t, setup.rayDirTarget, J_tangent,
                                                   noWeighting());

    // Check what the ambient function sees
    Quaternion q_from_x(setup.x[3], setup.x[0], setup.x[1], setup.x[2]);
    q_from_x.normalize();
    Eigen::Matrix3d R_from_x = q_from_x.toRotationMatrix();
    Eigen::Vector3d t_from_x(setup.x[4], setup.x[5], setup.x[6]);

    WARN("Quaternion from setup: q = [" << setup.x[0] << ", " << setup.x[1] << ", " << setup.x[2] << ", " << setup.x[3] << "]");
    WARN("R_from_x = \n" << R_from_x);
    WARN("R - R_from_x = \n" << (setup.R - R_from_x));
    WARN("t_from_x = [" << t_from_x.transpose() << "]");
    WARN("t - t_from_x = [" << (setup.t - t_from_x).transpose() << "]");

    // Compute what ambient will get for d
    Eigen::Vector3d d_from_ambient = R_from_x * setup.rayDirSource.cast<double>();
    WARN("d from ambient (R_from_x * rayDirSource) = [" << d_from_ambient.transpose() << "]");
    WARN("d_tangent (rayDirTarget) = [" << d_tangent.transpose() << "]");
    WARN("difference = " << (d_tangent - d_from_ambient).norm());

    // Replicate what ForwardRayCostSimplified does internally
    Vector3 pS_d = setup.srcPoint.cast<double>();
    Vector3 qT_d = setup.tgtPoint.cast<double>();
    Vector3 nT_d = setup.tgtNormal.cast<double>();
    Vector3 dS0_d = setup.rayDirSource.cast<double>();

    // Extract pose from x[] exactly as the ambient function does
    Quaternion q_ambient(setup.x[3], setup.x[0], setup.x[1], setup.x[2]);
    q_ambient.normalize();
    Vector3 t_ambient(setup.x[4], setup.x[5], setup.x[6]);
    Matrix3 R_ambient = q_ambient.toRotationMatrix();

    Vector3 xT_ambient = R_ambient * pS_d + t_ambient;
    Vector3 d_ambient_calc = R_ambient * dS0_d;

    double a_ambient_calc = nT_d.dot(xT_ambient - qT_d);
    double b_ambient_calc = nT_d.dot(d_ambient_calc);
    double r_ambient_calc = a_ambient_calc / b_ambient_calc;

    WARN("=== Replicating ambient computation ===");
    WARN("xT_ambient = [" << xT_ambient.transpose() << "]");
    WARN("d_ambient_calc = [" << d_ambient_calc.transpose() << "]");
    WARN("a = " << a_ambient_calc << ", b = " << b_ambient_calc << ", r = " << r_ambient_calc);

    // Print the exact values being passed
    WARN("pS_d = [" << pS_d.transpose() << "]");
    WARN("qT_d = [" << qT_d.transpose() << "]");
    WARN("nT_d = [" << nT_d.transpose() << "]");
    WARN("dS0_d = [" << dS0_d.transpose() << "]");
    WARN("params x[] = [" << setup.x[0] << ", " << setup.x[1] << ", " << setup.x[2] << ", " << setup.x[3]
         << ", " << setup.x[4] << ", " << setup.x[5] << ", " << setup.x[6] << "]");

    // Create cost function with weighting DISABLED for fair comparison
    GeometryWeighting noWeight;
    noWeight.enable_weight = false;
    noWeight.enable_gate = false;
    ForwardRayCostSimplified cost(pS_d, qT_d, nT_d, dS0_d, noWeight);

    double const* params[1] = {setup.x};
    double r_func_ambient;
    cost.Evaluate(params, &r_func_ambient, nullptr);

    // Also test with explicit double array
    double params_explicit[7] = {setup.x[0], setup.x[1], setup.x[2], setup.x[3],
                                  setup.x[4], setup.x[5], setup.x[6]};
    double const* params2[1] = {params_explicit};
    double r_func_ambient2;
    cost.Evaluate(params2, &r_func_ambient2, nullptr);
    WARN("r_func_ambient with explicit array = " << r_func_ambient2);

    WARN("Function results: r_tangent = " << r_func_tangent << ", r_ambient = " << r_func_ambient);
    WARN("r_ambient_calc (manual) = " << r_ambient_calc);
    WARN("Difference (func - manual): " << std::abs(r_func_ambient - r_ambient_calc));
    WARN("Difference (tangent - ambient): " << std::abs(r_func_tangent - r_func_ambient));

    CHECK(std::abs(r_func_tangent - r_func_ambient) < 1e-6);
}

/**
 * Verify both Jacobian types produce the same residual.
 */
TEST_CASE("Tangent and Ambient produce same residual", "[jacobians][tangent][residual]")
{
    std::mt19937 rng(12345);
    constexpr int numTrials = 20;

    for (int trial = 0; trial < numTrials; ++trial)
    {
        RandomTestSetup setup(rng);

        // Tangent residual (uses rayDirTarget = d in target frame)
        Eigen::Matrix<double, 1, 6> J_tangent;
        double r_tangent = computeForwardJacobian(setup.srcPoint, setup.tgtPoint, setup.tgtNormal,
                                                  setup.R, setup.t, setup.rayDirTarget, J_tangent,
                                                  noWeighting());

        // Ambient residual (uses rayDirSource = dS0, transforms internally as d = R * dS0)
        ForwardRayCostSimplified cost(
            setup.srcPoint.cast<double>(),
            setup.tgtPoint.cast<double>(),
            setup.tgtNormal.cast<double>(),
            setup.rayDirSource.cast<double>(),
            noWeighting());

        double const* params[1] = {setup.x};
        double r_ambient;
        cost.Evaluate(params, &r_ambient, nullptr);

        CHECK_THAT(r_tangent, WithinAbs(r_ambient, 1e-6));  // Relax for float precision
    }
}

/**
 * Verify reverse Jacobians also satisfy the adjoint relationship.
 */
TEST_CASE("Tangent vs Ambient reverse: adjoint relationship", "[jacobians][tangent][reverse]")
{
    std::mt19937 rng(RandomSeeds::JACOBIAN_REVERSE_TEST);

    constexpr int numTrials = 20;
    ValidationResult validation;

    for (int trial = 0; trial < numTrials; ++trial)
    {
        RandomTestSetup setup(rng);

        // Get tangent reverse Jacobian
        auto J_tangent = computeTangentReverseJacobian(setup);

        // Get ambient reverse Jacobian projected to 6D
        auto J_ambient_6D = computeAmbientReverseJacobian6D(setup);

        // Build inverse adjoint
        auto AdInv = buildInverseAdjoint(setup.R, setup.t);

        // Predicted: J_tangent = J_ambient_6D * Ad_T^{-1}
        Eigen::Matrix<double, 1, 6> J_predicted = J_ambient_6D * AdInv;

        // Compare
        double err = (J_tangent - J_predicted).norm();
        double scale = std::max(J_tangent.norm(), 1.0);
        double rel_err = err / scale;

        std::ostringstream oss;
        oss << "trial=" << trial << " rel_err=" << rel_err;
        validation.check(rel_err, 1e-6, oss.str());  // Float precision tolerance
    }

    INFO("Worst failure: " << validation.worst_desc << " (tol=" << validation.worst_tol << ")");
    CHECK(validation.passed);
}

/**
 * Verify reverse residuals match.
 */
TEST_CASE("Tangent and Ambient reverse produce same residual", "[jacobians][tangent][reverse][residual]")
{
    std::mt19937 rng(54321);
    constexpr int numTrials = 20;

    for (int trial = 0; trial < numTrials; ++trial)
    {
        RandomTestSetup setup(rng);

        // Tangent residual (reverse uses rayDirSource = d' in source frame)
        Eigen::Matrix<double, 1, 6> J_tangent;
        double r_tangent = computeReverseJacobian(setup.srcPoint, setup.tgtPoint, setup.tgtNormal,
                                                  setup.R, setup.t, setup.rayDirSource, J_tangent,
                                                  noWeighting());

        // Ambient residual (reverse uses rayDirTarget = dT0 in target frame)
        ReverseRayCostSimplified cost(
            setup.srcPoint.cast<double>(),
            setup.tgtPoint.cast<double>(),
            setup.tgtNormal.cast<double>(),
            setup.rayDirTarget.cast<double>(),
            noWeighting());

        double const* params[1] = {setup.x};
        double r_ambient;
        cost.Evaluate(params, &r_ambient, nullptr);

        CHECK_THAT(r_tangent, WithinAbs(r_ambient, 1e-6));  // Relax for float precision
    }
}

/**
 * Verify weighted tangent Jacobians match weighted ambient Jacobians.
 *
 * Both Simplified Jacobians ignore the derivative of the denominator (and weight)
 * with respect to R, so the adjoint relationship should still hold even with
 * weighting enabled.
 */
TEST_CASE("Weighted tangent vs ambient: adjoint relationship", "[jacobians][tangent][weighted]")
{
    std::mt19937 rng(RandomSeeds::JACOBIAN_POLICY_TEST + 100);

    // Use default weighting (SqrtAbs mode)
    GeometryWeighting weighting;
    weighting.enable_weight = true;
    weighting.enable_gate = false;  // Disable gating to avoid zero weights
    weighting.tau = 0.0;

    constexpr int numTrials = 20;
    ValidationResult validation;

    for (int trial = 0; trial < numTrials; ++trial)
    {
        RandomTestSetup setup(rng);

        // Get weighted tangent Jacobian (left perturbation, space frame)
        auto J_tangent = computeTangentJacobian(setup, weighting);

        // Get weighted ambient Jacobian projected to 6D (right perturbation, body frame)
        auto J_ambient_6D = computeAmbientJacobian6D(setup, weighting);

        // Build inverse adjoint: Ad_T^{-1}
        auto AdInv = buildInverseAdjoint(setup.R, setup.t);

        // Predicted: J_tangent = J_ambient_6D * Ad_T^{-1}
        Eigen::Matrix<double, 1, 6> J_predicted = J_ambient_6D * AdInv;

        // Compare
        double err = (J_tangent - J_predicted).norm();
        double scale = std::max(J_tangent.norm(), 1.0);
        double rel_err = err / scale;

        std::ostringstream oss;
        oss << "trial=" << trial << " rel_err=" << rel_err;
        validation.check(rel_err, 1e-6, oss.str());

        if (trial == 0)
        {
            WARN("=== Weighted Jacobian comparison (trial 0) ===");
            WARN("J_tangent    = [" << J_tangent << "]");
            WARN("J_ambient_6D = [" << J_ambient_6D << "]");
            WARN("J_predicted  = [" << J_predicted << "]");
            WARN("rel_err = " << rel_err);
        }
    }

    INFO("Worst failure: " << validation.worst_desc << " (tol=" << validation.worst_tol << ")");
    CHECK(validation.passed);
}

/**
 * Verify weighted residuals match between tangent and ambient.
 */
TEST_CASE("Weighted tangent and ambient produce same residual", "[jacobians][tangent][weighted][residual]")
{
    std::mt19937 rng(98765);

    GeometryWeighting weighting;
    weighting.enable_weight = true;
    weighting.enable_gate = false;
    weighting.tau = 0.0;

    constexpr int numTrials = 20;

    for (int trial = 0; trial < numTrials; ++trial)
    {
        RandomTestSetup setup(rng);

        // Tangent weighted residual
        Eigen::Matrix<double, 1, 6> J_tangent;
        double r_tangent = computeForwardJacobian(setup.srcPoint, setup.tgtPoint, setup.tgtNormal,
                                                  setup.R, setup.t, setup.rayDirTarget, J_tangent, weighting);

        // Ambient weighted residual
        ForwardRayCostSimplified cost(
            setup.srcPoint.cast<double>(),
            setup.tgtPoint.cast<double>(),
            setup.tgtNormal.cast<double>(),
            setup.rayDirSource.cast<double>(),
            weighting);

        double const* params[1] = {setup.x};
        double r_ambient;
        cost.Evaluate(params, &r_ambient, nullptr);

        CHECK_THAT(r_tangent, WithinAbs(r_ambient, 1e-6));
    }
}
