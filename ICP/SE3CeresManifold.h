// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Jens Munk Hansen

#pragma once
/**
 * @file SE3CeresManifold.h
 * @brief Custom Ceres manifold for SE(3) with rotation scaling.
 *
 * This header provides a Ceres-compatible SE(3) manifold that supports
 * rotation scaling for improved optimization conditioning.
 *
 * @see @ref effective_conditioning for the mathematical background
 */

// Ceres headers
#include <ceres/manifold.h>

// Internal headers
#include <ICP/EigenTypes.h>
#include <ICP/SE3.h>

namespace ICP
{

/**
 * @brief Custom Ceres manifold for SE(3) with rotation scaling.
 *
 * This manifold uses `se3PlusScaled` and `plusJacobian7x6Scaled` to implement
 * the Plus operation with a configurable rotation scale parameter.
 *
 * The scaling applies a characteristic length L to the rotation component:
 * \f[
 *   \delta' = S \delta, \quad S = \begin{bmatrix} I_3 & 0 \\ 0 & L I_3 \end{bmatrix}
 * \f]
 *
 * This makes "1 unit of rotation (radian)" comparable to "L units of translation",
 * improving conditioning when translation and rotation have very different scales.
 *
 * The ambient space is 7D: [qx, qy, qz, qw, tx, ty, tz]
 * The tangent space is 6D: [v_x, v_y, v_z, w_x, w_y, w_z]
 *
 * @note The final result is invariant to scaling; only convergence speed is affected.
 *
 * @see @ref effective_conditioning for mathematical background
 */
class SE3ScaledManifold : public ceres::Manifold
{
public:
    /**
     * @brief Construct manifold with specified rotation scale.
     * @param rotationScale Characteristic length L (1.0 = no scaling)
     */
    explicit SE3ScaledManifold(double rotationScale = 1.0)
        : rotationScale_(rotationScale)
    {
    }

    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }

    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override
    {
        Pose7 pose;
        pose << x[0], x[1], x[2], x[3], x[4], x[5], x[6];

        Tangent6 d;
        d << delta[0], delta[1], delta[2], delta[3], delta[4], delta[5];

        Pose7 result = se3PlusScaled(pose, d, rotationScale_);

        for (int i = 0; i < 7; ++i)
        {
            x_plus_delta[i] = result[i];
        }
        return true;
    }

    bool PlusJacobian(const double* x, double* jacobian) const override
    {
        Pose7 pose;
        pose << x[0], x[1], x[2], x[3], x[4], x[5], x[6];

        Matrix7x6 J = plusJacobian7x6Scaled(pose, rotationScale_);

        // Ceres expects row-major storage for the Jacobian
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J_map(jacobian);
        J_map = J;

        return true;
    }

    bool Minus(const double* y, const double* x, double* y_minus_x) const override
    {
        // Compute the tangent vector that takes x to y
        // For SE(3): delta = Log(T_x^{-1} * T_y)

        Quaternion qx(x[3], x[0], x[1], x[2]);
        Quaternion qy(y[3], y[0], y[1], y[2]);
        Vector3 tx(x[4], x[5], x[6]);
        Vector3 ty(y[4], y[5], y[6]);

        // Relative rotation: dq = qx^{-1} * qy
        Quaternion dq = qx.conjugate() * qy;
        dq.normalize();

        // Convert to axis-angle
        Eigen::AngleAxisd aa(dq);
        Vector3 w = aa.angle() * aa.axis();

        // Relative translation in body frame: R_x^T * (ty - tx)
        Matrix3 Rx = qx.toRotationMatrix();
        Vector3 dt = Rx.transpose() * (ty - tx);

        // Approximate V^{-1} for small angles (V ≈ I)
        Vector3 v = dt;

        // Apply inverse scaling: the tangent delta has scaled rotation
        y_minus_x[0] = v[0];
        y_minus_x[1] = v[1];
        y_minus_x[2] = v[2];
        y_minus_x[3] = w[0] / rotationScale_;
        y_minus_x[4] = w[1] / rotationScale_;
        y_minus_x[5] = w[2] / rotationScale_;

        return true;
    }

    bool MinusJacobian(const double* x, double* jacobian) const override
    {
        // The Jacobian of Minus at x is the pseudo-inverse of PlusJacobian
        Pose7 pose;
        pose << x[0], x[1], x[2], x[3], x[4], x[5], x[6];

        Matrix7x6 P = plusJacobian7x6Scaled(pose, rotationScale_);

        // Compute pseudo-inverse: (P^T P)^{-1} P^T
        Matrix6 PtP = P.transpose() * P;
        Eigen::Matrix<double, 6, 7> Pinv = PtP.ldlt().solve(P.transpose());

        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_map(jacobian);
        J_map = Pinv;

        return true;
    }

private:
    double rotationScale_;
};

} // namespace ICP
