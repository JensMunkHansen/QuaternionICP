// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once
/**
 * Left-perturbation (tangent/6D) Jacobians for ray-projection ICP.
 *
 * Left-perturbation Jacobians using se(3) tangent space parameterization.
 * Policy-based design via template specialization (matching JacobiansAmbient.h).
 *
 * Residual: r = a/b where
 *   a = n^T (x_transformed - q_surface)
 *   b = n^T d  (ray direction dot normal)
 *
 * These Jacobians use LEFT perturbation (space/fixed frame):
 *   T_new = exp(delta_xi_hat) * T
 *
 * Compare to JacobiansAmbient.h which uses RIGHT perturbation (body/moving frame):
 *   T_new = T * exp(delta_xi_hat)
 *
 * The relationship between them is: J_left = J_right * Ad_T^{-1}
 */

#include <Eigen/Core>
#include <iostream>

#include <ICP/ICPParams.h>

namespace ICP
{

// Debug flag - set to true to dump first few correspondences
inline bool g_debugLeftJacobians = false;
inline int g_debugLeftCount = 0;
inline int g_debugLeftMaxCount = 5;

// ============================================================================
// Forward ray-projection left cost - primary template (not defined)
// ============================================================================

template<typename JacobianPolicy>
class ForwardRayLeft;

// ============================================================================
// ForwardRayLeft<RayJacobianSimplified>
// ============================================================================

/**
 * @brief Forward left-perturbation Jacobian for ray-projected point-to-plane ICP.
 *
 * Given a correspondence from source grid to target grid:
 * - Source point p (in source-local coords)
 * - Target point q (in target-local coords)
 * - Target normal n (in target-local coords)
 * - Ray direction d (in target-local coords)
 * - Current transform T = (R, t) from source-local to target-local
 *
 * The transformed source point is: x = R*p + t
 *
 * Residual (ray-projected point-to-plane distance):
 *   r = n^T (x - q) / (n^T d)
 *
 * Jacobian w.r.t. se(3) tangent [rho; phi] (left perturbation):
 *   J = [+n^T | (x × n)^T] / (n^T d)
 *
 * Simplified: ignores db/dphi term (derivative of denominator w.r.t. rotation)
 */
template<>
class ForwardRayLeft<RayJacobianSimplified>
{
public:
    using policy_tag = RayJacobianSimplified;

    ForwardRayLeft(const Eigen::Vector3f& pS, const Eigen::Vector3f& qT,
                   const Eigen::Vector3f& nT, const Eigen::Vector3f& dT,
                   const GeometryWeighting& weighting = GeometryWeighting())
        : pS_(pS), qT_(qT), nT_(nT), dT_(dT), weighting_(weighting)
    {
    }

    /**
     * @brief Evaluate residual and Jacobian.
     *
     * @param R Current rotation estimate (3x3 matrix)
     * @param t Current translation estimate (3-vector)
     * @param[out] J 6D Jacobian row vector [d/dt | d/dphi]
     * @return Weighted residual value
     */
    double Evaluate(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                    Eigen::Matrix<double, 1, 6>& J) const
    {
        // Transform source point to target frame
        Eigen::Vector3d p = pS_.cast<double>();
        Eigen::Vector3d x = R * p + t;

        Eigen::Vector3d n = nT_.cast<double>();
        Eigen::Vector3d q = qT_.cast<double>();
        Eigen::Vector3d d = dT_.cast<double>();

        // Ray normalization factor (denominator)
        double b = n.dot(d);

        // Compute weight based on incidence angle
        double w = weighting_.weight(b);

        // Numerator and residual
        double a = n.dot(x - q);
        double r = a / b;

        // Jacobian: w * [+n^T | (x × n)^T] / b
        // Simplified: ignores db/dphi
        J.head<3>() = (w * n.transpose()) / b;
        J.tail<3>() = (w * x.cross(n).transpose()) / b;

        if (g_debugLeftJacobians && g_debugLeftCount < g_debugLeftMaxCount)
        {
            std::cout << "  [FWD " << g_debugLeftCount << "] p=" << p.transpose()
                      << " -> x=" << x.transpose() << "\n";
            std::cout << "           q=" << q.transpose() << ", n=" << n.transpose() << "\n";
            std::cout << "           d=" << d.transpose() << ", b=" << b << ", w=" << w << "\n";
            std::cout << "           r=" << w * r << ", J=[" << J << "]\n";
            g_debugLeftCount++;
        }

        return w * r;
    }

    /**
     * @brief Evaluate residual only (no Jacobian).
     */
    double Residual(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) const
    {
        Eigen::Vector3d p = pS_.cast<double>();
        Eigen::Vector3d x = R * p + t;
        Eigen::Vector3d n = nT_.cast<double>();
        Eigen::Vector3d q = qT_.cast<double>();
        Eigen::Vector3d d = dT_.cast<double>();

        double b = n.dot(d);
        double w = weighting_.weight(b);
        double a = n.dot(x - q);

        return w * a / b;
    }

private:
    Eigen::Vector3f pS_, qT_, nT_, dT_;
    GeometryWeighting weighting_;
};

// ============================================================================
// Reverse ray-projection left cost - primary template (not defined)
// ============================================================================

template<typename JacobianPolicy>
class ReverseRayLeft;

// ============================================================================
// ReverseRayLeft<RayJacobianSimplified>
// ============================================================================

/**
 * @brief Reverse left-perturbation Jacobian for ray-projected point-to-plane ICP.
 *
 * Given a correspondence from target grid back to source grid:
 * - pT q: ray origin in target-local coordinates
 * - qS p: hit point in source-local coordinates
 * - nS: normal at hit in source-local coordinates
 * - dS: ray direction in source-local coordinates
 * - Transform T = (R, t) from source-local to target-local
 *
 * The inverse-transformed point is: x' = R^T(q - t)
 *
 * Residual (ray-projected point-to-plane distance in source frame):
 *   r' = nS^T (x' - p) / (nS^T dS)
 *
 * Jacobian w.r.t. se(3) tangent [rho; phi] (left perturbation):
 *   J' = [-nbar^T | (nbar × q)^T] / (nS^T dS)
 *
 * where nbar = R * nS (source normal rotated to target frame)
 *
 * Note: The negative sign on translation comes from how T^-1 responds to
 * perturbation: moving T forward moves inverse-transformed points backward.
 *
 * Simplified: ignores db/dphi term
 */
template<>
class ReverseRayLeft<RayJacobianSimplified>
{
public:
    using policy_tag = RayJacobianSimplified;

    ReverseRayLeft(const Eigen::Vector3f& pT, const Eigen::Vector3f& qS,
                   const Eigen::Vector3f& nS, const Eigen::Vector3f& dS,
                   const GeometryWeighting& weighting = GeometryWeighting())
        : pT_(pT), qS_(qS), nS_(nS), dS_(dS), weighting_(weighting)
    {
    }

    /**
     * @brief Evaluate residual and Jacobian.
     *
     * @param R Current rotation estimate (3x3 matrix)
     * @param t Current translation estimate (3-vector)
     * @param[out] J 6D Jacobian row vector [d/dt | d/dphi]
     * @return Weighted residual value
     */
    double Evaluate(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                    Eigen::Matrix<double, 1, 6>& J) const
    {
        Eigen::Vector3d q = pT_.cast<double>();   // Ray origin in target-local
        Eigen::Vector3d p = qS_.cast<double>();   // Hit point in source-local
        Eigen::Vector3d ns = nS_.cast<double>();  // Normal in source-local
        Eigen::Vector3d d = dS_.cast<double>();

        // Inverse-transform q to source frame
        Eigen::Vector3d xprime = R.transpose() * (q - t);

        // Rotated normal (source normal in target frame)
        Eigen::Vector3d nbar = R * ns;

        // Ray normalization factor (in source frame)
        double b = ns.dot(d);

        // Compute weight based on incidence angle
        double w = weighting_.weight(b);

        // Residual in source frame
        double a = ns.dot(xprime - p);
        double r = a / b;

        // Jacobian: w * [-nbar^T | (nbar × q)^T] / b
        // Simplified: ignores db/dphi
        J.head<3>() = (-w * nbar.transpose()) / b;
        J.tail<3>() = (w * nbar.cross(q).transpose()) / b;

        if (g_debugLeftJacobians && g_debugLeftCount < g_debugLeftMaxCount)
        {
            std::cout << "  [REV " << g_debugLeftCount << "] q(tgt)=" << q.transpose()
                      << " -> x'=" << xprime.transpose() << "\n";
            std::cout << "           p(src)=" << p.transpose() << ", ns=" << ns.transpose() << "\n";
            std::cout << "           nbar=" << nbar.transpose() << ", b=" << b << ", w=" << w << "\n";
            std::cout << "           r=" << w * r << ", J=[" << J << "]\n";
            g_debugLeftCount++;
        }

        return w * r;
    }

    /**
     * @brief Evaluate residual only (no Jacobian).
     */
    double Residual(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) const
    {
        Eigen::Vector3d q = pT_.cast<double>();
        Eigen::Vector3d p = qS_.cast<double>();
        Eigen::Vector3d ns = nS_.cast<double>();
        Eigen::Vector3d d = dS_.cast<double>();

        Eigen::Vector3d xprime = R.transpose() * (q - t);
        double b = ns.dot(d);
        double w = weighting_.weight(b);
        double a = ns.dot(xprime - p);

        return w * a / b;
    }

private:
    Eigen::Vector3f pT_, qS_, nS_, dS_;
    GeometryWeighting weighting_;
};

// ============================================================================
// Type aliases
// ============================================================================

using ForwardRayLeftSimplified = ForwardRayLeft<RayJacobianSimplified>;
using ReverseRayLeftSimplified = ReverseRayLeft<RayJacobianSimplified>;

// ============================================================================
// Convenience free functions (backward compatibility)
// ============================================================================

/**
 * @brief Compute forward left-perturbation Jacobian (simplified).
 *
 * @param srcPoint Source point p in source-local coordinates
 * @param tgtPoint Target point q in target-local coordinates
 * @param tgtNormal Target normal n in target-local coordinates
 * @param R Current rotation estimate
 * @param t Current translation estimate
 * @param rayDir Ray direction d in target-local coordinates
 * @param[out] J 6D Jacobian row vector [d/dt | d/dphi]
 * @param weighting Geometry weighting parameters
 * @return Weighted residual value
 */
inline double computeForwardJacobian(const Eigen::Vector3f& srcPoint,
    const Eigen::Vector3f& tgtPoint, const Eigen::Vector3f& tgtNormal,
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
    const Eigen::Vector3f& rayDir, Eigen::Matrix<double, 1, 6>& J,
    const GeometryWeighting& weighting = GeometryWeighting())
{
    ForwardRayLeftSimplified cost(srcPoint, tgtPoint, tgtNormal, rayDir, weighting);
    return cost.Evaluate(R, t, J);
}

/**
 * @brief Compute reverse left-perturbation Jacobian (simplified).
 *
 * @param srcPoint Ray origin q in target-local coordinates
 * @param tgtPoint Hit point p in source-local coordinates
 * @param tgtNormal Normal n_s in source-local coordinates
 * @param R Current rotation estimate
 * @param t Current translation estimate
 * @param rayDir Ray direction d' in source-local coordinates
 * @param[out] J 6D Jacobian row vector [d/dt | d/dphi]
 * @param weighting Geometry weighting parameters
 * @return Weighted residual value
 */
inline double computeReverseJacobian(const Eigen::Vector3f& srcPoint,
    const Eigen::Vector3f& tgtPoint, const Eigen::Vector3f& tgtNormal,
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
    const Eigen::Vector3f& rayDir, Eigen::Matrix<double, 1, 6>& J,
    const GeometryWeighting& weighting = GeometryWeighting())
{
    ReverseRayLeftSimplified cost(srcPoint, tgtPoint, tgtNormal, rayDir, weighting);
    return cost.Evaluate(R, t, J);
}

} // namespace ICP
