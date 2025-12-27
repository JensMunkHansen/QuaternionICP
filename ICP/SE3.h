#pragma once
/**
 * SE(3) and SO(3) utilities for quaternion-based pose representation.
 *
 * Quaternion convention: Eigen stores [x, y, z, w] internally (coeffs()).
 * SE(3) pose: 7D vector [qx, qy, qz, qw, tx, ty, tz]
 * Tangent space: 6D vector [v_x, v_y, v_z, w_x, w_y, w_z] (translation, rotation)
 *
 * Uses right-multiplication (body/moving frame) convention:
 *   T_new = T * Exp(delta^)
 */

// Standard C++ headers
#include <array>
#include <cmath>

// Internal headers
#include <ICP/EigenTypes.h>

namespace ICP
{

/// Small angle threshold for Taylor expansion in SO(3)/SE(3) operations
constexpr double kSmallAngleThreshold = 1e-12;

// -----------------------------
// SO(3) utilities
// -----------------------------

/**
 * Skew-symmetric matrix from 3-vector: [w]_x such that [w]_x * v = w x v
 */
inline Matrix3 skew(const Vector3& w)
{
    Matrix3 S;
    S << 0, -w.z(), w.y(),
         w.z(), 0, -w.x(),
        -w.y(), w.x(), 0;
    return S;
}

/**
 * SO(3) exponential map: axis-angle vector -> quaternion.
 * Uses first-order Taylor expansion for small angles.
 *
 * Note: Eigen stores quaternion coeffs as [x, y, z, w] internally.
 */
inline Quaternion quatExpSO3(const Vector3& w)
{
    double th = w.norm();
    if (th < kSmallAngleThreshold)
    {
        // First-order Taylor: q ≈ [w/2, 1], then normalize
        Vector3 v = 0.5 * w;
        return Quaternion(1.0, v.x(), v.y(), v.z()).normalized();
    }
    Vector3 axis = w / th;
    double half = 0.5 * th;
    double s = std::sin(half);
    Vector3 v = axis * s;
    return Quaternion(std::cos(half), v.x(), v.y(), v.z());
}

/**
 * Left Jacobian of SO(3).
 * V such that for the SE(3) Plus operation: t_new = t + R @ V @ v
 *
 * V = I + B * [w]_x + C * [w]_x^2
 * where B = (1 - cos(th)) / th^2, C = (1 - sin(th)/th) / th^2
 */
inline Matrix3 Vso3(const Vector3& w)
{
    Matrix3 W = skew(w);
    double th = w.norm();
    if (th < kSmallAngleThreshold)
    {
        return Matrix3::Identity() + 0.5 * W;
    }
    double A = std::sin(th) / th;
    double B = (1.0 - std::cos(th)) / (th * th);
    double C = (1.0 - A) / (th * th);
    return Matrix3::Identity() + B * W + C * (W * W);
}

// -----------------------------
// SE(3) manifold operations (right-multiplication)
// -----------------------------

/// SE(3) pose as 7D vector: [qx, qy, qz, qw, tx, ty, tz]
using Pose7 = Vector7;

/// SE(3) tangent vector: [v_x, v_y, v_z, w_x, w_y, w_z]
using Tangent6 = Vector6;

/**
 * SE(3) Plus operation (right-multiplication):
 *   T_new = T * Exp(delta^)
 *
 * @param x     Current pose [qx, qy, qz, qw, tx, ty, tz]
 * @param delta Tangent increment [v_x, v_y, v_z, w_x, w_y, w_z]
 * @return      Updated pose
 */
inline Pose7 se3Plus(const Pose7& x, const Tangent6& delta)
{
    // Extract quaternion and translation
    Quaternion q(x[3], x[0], x[1], x[2]);  // w, x, y, z
    q.normalize();
    Vector3 t = x.tail<3>();

    // Extract tangent components
    Vector3 v = delta.head<3>();  // translation part
    Vector3 w = delta.tail<3>();  // rotation part

    // Rotation update: q_new = q * exp(w)
    Quaternion dq = quatExpSO3(w);
    Quaternion q_new = (q * dq).normalized();

    // Translation update: t_new = t + R * V * v
    Matrix3 R = q.toRotationMatrix();
    Matrix3 V = Vso3(w);
    Vector3 t_new = t + R * (V * v);

    // Pack result
    Pose7 result;
    result << q_new.x(), q_new.y(), q_new.z(), q_new.w(), t_new;
    return result;
}

/**
 * PlusJacobian: d Plus(x, delta) / d delta |_{delta=0}
 *
 * Returns 7x6 matrix mapping tangent increments to ambient changes.
 * Used as: J_local = J_ambient @ PlusJacobian
 */
inline Matrix7x6 plusJacobian7x6(const Pose7& x)
{
    // Extract normalized quaternion
    Quaternion q(x[3], x[0], x[1], x[2]);
    q.normalize();

    Vector3 vq(q.x(), q.y(), q.z());
    double s = q.w();
    Matrix3 R = q.toRotationMatrix();

    Matrix7x6 J = Matrix7x6::Zero();

    // dq/dw (at delta=0): 4x3 block
    // dq = 0.5 * q * [0, w]  for small w
    // => d(vq)/dw = 0.5 * (s*I + [vq]_x)
    // => d(s)/dw  = -0.5 * vq
    Matrix4x3 dq_dw;
    dq_dw.topRows<3>() = 0.5 * (s * Matrix3::Identity() + skew(vq));
    dq_dw.row(3) = -0.5 * vq.transpose();

    // dt/dv (at delta=0, V=I): R
    J.block<4, 3>(0, 3) = dq_dw;   // quaternion rows, rotation columns
    J.block<3, 3>(4, 0) = R;       // translation rows, translation columns

    return J;
}

// -----------------------------
// Quaternion derivatives (for ambient Jacobians)
// -----------------------------

/**
 * Derivatives of rotation matrix R w.r.t. quaternion components.
 * Returns [dR/dx, dR/dy, dR/dz, dR/dw] as an array of four 3x3 matrices.
 *
 * @param q Normalized quaternion (will normalize internally)
 */
inline std::array<Matrix3, 4> dR_dq_mats(const Quaternion& q)
{
    Quaternion qn = q.normalized();
    double x = qn.x(), y = qn.y(), z = qn.z(), w = qn.w();

    Matrix3 dRdx, dRdy, dRdz, dRdw;

    dRdx << 0.0,   2*y,   2*z,
            2*y,  -4*x,  -2*w,
            2*z,   2*w,  -4*x;

    dRdy << -4*y,  2*x,   2*w,
             2*x,  0.0,   2*z,
            -2*w,  2*z,  -4*y;

    dRdz << -4*z, -2*w,  2*x,
             2*w, -4*z,  2*y,
             2*x,  2*y,  0.0;

    dRdw << 0.0, -2*z,  2*y,
            2*z,  0.0, -2*x,
           -2*y,  2*x,  0.0;

    return {dRdx, dRdy, dRdz, dRdw};
}

/**
 * Derivative of R(q) * v w.r.t. quaternion q.
 * Returns 3x4 matrix: [d(Rv)/dx, d(Rv)/dy, d(Rv)/dz, d(Rv)/dw]
 */
inline Matrix3x4 dRv_dq(const Quaternion& q, const Vector3& v)
{
    auto [dRdx, dRdy, dRdz, dRdw] = dR_dq_mats(q);
    Matrix3x4 J;
    J.col(0) = dRdx * v;
    J.col(1) = dRdy * v;
    J.col(2) = dRdz * v;
    J.col(3) = dRdw * v;
    return J;
}

/**
 * Derivative of R(q).T * v w.r.t. quaternion q.
 * Returns 3x4 matrix.
 */
inline Matrix3x4 dRTv_dq(const Quaternion& q, const Vector3& v)
{
    auto [dRdx, dRdy, dRdz, dRdw] = dR_dq_mats(q);
    Matrix3x4 J;
    J.col(0) = dRdx.transpose() * v;
    J.col(1) = dRdy.transpose() * v;
    J.col(2) = dRdz.transpose() * v;
    J.col(3) = dRdw.transpose() * v;
    return J;
}

} // namespace ICP
