/**
 * @file SE3.h
 * @brief SE(3) Lie group utilities for manifold Gauss-Newton optimization.
 *
 * This header provides the mathematical foundations for SE(3) (Special Euclidean group)
 * operations used in point cloud registration. The implementation follows the
 * **right-multiplicative** (body-frame) convention for tangent space updates.
 *
 * @section se3_conventions Conventions
 *
 * - Quaternion storage: Eigen internal order `[x, y, z, w]` via `coeffs()`
 * - SE(3) pose: 7D vector `[qx, qy, qz, qw, tx, ty, tz]`
 * - Tangent space: 6D vector `[v_x, v_y, v_z, w_x, w_y, w_z]` (translation, rotation)
 *
 * @section se3_groups Lie Groups
 *
 * @subsection se3_so3 SO(3): Special Orthogonal Group
 *
 * SO(3) is the group of 3D rotations:
 * \f[
 *   \mathrm{SO}(3) = \{ R \in \mathbb{R}^{3 \times 3} : R^\top R = I, \det(R) = 1 \}
 * \f]
 *
 * @subsection se3_se3 SE(3): Special Euclidean Group
 *
 * SE(3) is the group of rigid body transformations:
 * \f[
 *   \mathrm{SE}(3) = \left\{ T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}
 *   : R \in \mathrm{SO}(3), \mathbf{t} \in \mathbb{R}^3 \right\}
 * \f]
 *
 * @section se3_skew Skew-Symmetric Matrix (Hat Operator)
 *
 * The hat operator \f$[\cdot]_\times : \mathbb{R}^3 \to \mathfrak{so}(3)\f$ creates
 * a skew-symmetric matrix encoding the cross product:
 * \f[
 *   [\mathbf{w}]_\times = \begin{bmatrix}
 *     0 & -w_z & w_y \\
 *     w_z & 0 & -w_x \\
 *     -w_y & w_x & 0
 *   \end{bmatrix}
 * \f]
 *
 * @section se3_expso3 SO(3) Exponential Map (Quaternion Form)
 *
 * The unit quaternion corresponding to \f$\exp([\boldsymbol{\phi}]_\times)\f$ is:
 * \f[
 *   \mathbf{q} = \begin{bmatrix} \cos(\theta/2) \\ \sin(\theta/2) \hat{\mathbf{a}} \end{bmatrix}
 * \f]
 *
 * where \f$\theta = \|\boldsymbol{\phi}\|\f$ and \f$\hat{\mathbf{a}} = \boldsymbol{\phi}/\theta\f$.
 *
 * @section se3_leftjac Left Jacobian of SO(3)
 *
 * The left Jacobian \f$V\f$ couples rotation and translation in SE(3):
 * \f[
 *   V = I + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times
 *         + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\phi}]_\times^2
 * \f]
 *
 * @section se3_update Right-Multiplicative Update (Body Frame)
 *
 * Pose updates use **right multiplication**:
 * \f[
 *   T \leftarrow T \cdot \exp(\widehat{\delta\boldsymbol{\xi}})
 * \f]
 *
 * For a tangent vector \f$\boldsymbol{\xi} = (\mathbf{v}, \boldsymbol{\omega})\f$:
 * \f[
 *   R \leftarrow R \cdot \Delta R, \quad
 *   \mathbf{t} \leftarrow \mathbf{t} + R \cdot V \cdot \mathbf{v}
 * \f]
 *
 * This corresponds to perturbations in the **body/moving frame**, where increments
 * are expressed relative to the current pose rather than the world frame.
 */

#pragma once

// Standard C++ headers
#include <array>
#include <cmath>

// Internal headers
#include <ICP/EigenTypes.h>

namespace ICP
{

/**
 * @brief Small angle threshold for Taylor expansion in SO(3)/SE(3) operations.
 *
 * When \f$\|\boldsymbol{\omega}\| < \epsilon\f$, first-order Taylor expansions
 * are used to avoid numerical instability in trigonometric functions.
 */
constexpr double kSmallAngleThreshold = 1e-12;

// -----------------------------
// SO(3) utilities
// -----------------------------

/**
 * @brief Compute the skew-symmetric matrix from a 3D vector.
 *
 * Creates the matrix \f$[\mathbf{w}]_\times\f$ such that
 * \f$[\mathbf{w}]_\times \mathbf{v} = \mathbf{w} \times \mathbf{v}\f$:
 * \f[
 *   [\mathbf{w}]_\times = \begin{bmatrix}
 *     0 & -w_z & w_y \\
 *     w_z & 0 & -w_x \\
 *     -w_y & w_x & 0
 *   \end{bmatrix}
 * \f]
 *
 * @param w Input 3D vector
 * @return 3x3 skew-symmetric matrix
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
 * @brief SO(3) exponential map in quaternion form.
 *
 * Computes the unit quaternion corresponding to \f$\exp([\boldsymbol{\omega}]_\times)\f$:
 * \f[
 *   \mathbf{q} = \begin{bmatrix} w \\ \mathbf{u} \end{bmatrix}
 *             = \begin{bmatrix} \cos(\theta/2) \\ \sin(\theta/2) \hat{\mathbf{a}} \end{bmatrix}
 * \f]
 *
 * where \f$\theta = \|\boldsymbol{\omega}\|\f$ is the rotation angle and
 * \f$\hat{\mathbf{a}} = \boldsymbol{\omega}/\theta\f$ is the unit rotation axis.
 *
 * For small angles (\f$\theta < 10^{-12}\f$), uses first-order Taylor expansion.
 *
 * @note Eigen stores quaternion coeffs as [x, y, z, w] internally.
 *
 * @param w Rotation vector (axis-angle representation)
 * @return Unit quaternion (w, x, y, z)
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
 * @brief Left Jacobian of SO(3).
 *
 * Computes the matrix \f$V\f$ used in the SE(3) Plus operation:
 * \f$\mathbf{t}_{\text{new}} = \mathbf{t} + R \cdot V \cdot \mathbf{v}\f$
 *
 * \f[
 *   V = I + B \cdot [\boldsymbol{\omega}]_\times + C \cdot [\boldsymbol{\omega}]_\times^2
 * \f]
 *
 * where \f$\theta = \|\boldsymbol{\omega}\|\f$ and:
 * - \f$B = (1 - \cos\theta) / \theta^2\f$
 * - \f$C = (1 - \sin\theta / \theta) / \theta^2\f$
 *
 * For small angles (\f$\theta < 10^{-12}\f$), uses \f$V \approx I + \frac{1}{2}[\boldsymbol{\omega}]_\times\f$.
 *
 * @param w Rotation vector
 * @return 3x3 left Jacobian matrix
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

/// SE(3) tangent vector: [v_x, v_y, v_z, w_x, w_y, w_z]
using Tangent6 = Vector6;

/**
 * @brief SE(3) Plus operation using right-multiplication.
 *
 * Computes \f$T_{\text{new}} = T \cdot \exp(\widehat{\boldsymbol{\xi}})\f$ where
 * \f$\boldsymbol{\xi} = (\mathbf{v}, \boldsymbol{\omega}) \in \mathbb{R}^6\f$:
 * \f[
 *   R \leftarrow R \cdot \Delta R, \quad
 *   \mathbf{t} \leftarrow \mathbf{t} + R \cdot V \cdot \mathbf{v}
 * \f]
 *
 * @param x     Current pose [qx, qy, qz, qw, tx, ty, tz]
 * @param delta Tangent increment [v_x, v_y, v_z, w_x, w_y, w_z]
 * @return Updated pose
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
 * @brief Jacobian of the Plus operation w.r.t. the tangent increment.
 *
 * Computes \f$\frac{\partial \text{Plus}(x, \delta)}{\partial \delta}\big|_{\delta=0}\f$.
 *
 * Returns a 7x6 matrix mapping tangent increments to ambient (7D) changes.
 * Used for chain rule: \f$J_{\text{local}} = J_{\text{ambient}} \cdot P\f$
 *
 * @param x Current pose [qx, qy, qz, qw, tx, ty, tz]
 * @return 7x6 Jacobian matrix
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

/**
 * @brief Project 7D ambient Jacobian to 6D local (tangent space) Jacobian.
 *
 * Computes \f$J_{\text{local}} = J_{\text{ambient}} \cdot P(x)\f$ where
 * \f$P(x)\f$ is the Plus Jacobian at pose \f$x\f$.
 *
 * @param J7 7D ambient Jacobian (1x7 row vector)
 * @param x  7D pose [qx, qy, qz, qw, tx, ty, tz]
 * @return 6D local Jacobian (1x6 row vector)
 */
inline Eigen::Matrix<double, 1, 6> jacobian7Dto6D(const double* J7, const double* x)
{
    Pose7 pose;
    pose << x[0], x[1], x[2], x[3], x[4], x[5], x[6];
    auto P = plusJacobian7x6(pose);
    Eigen::Map<const Eigen::RowVectorXd> J7_map(J7, 7);
    return J7_map * P;
}

// -----------------------------
// Quaternion derivatives (for ambient Jacobians)
// -----------------------------

/**
 * @brief Derivatives of rotation matrix R w.r.t. quaternion components.
 *
 * Computes \f$\frac{\partial R}{\partial q_i}\f$ for each quaternion component.
 *
 * @param q Normalized quaternion (will normalize internally)
 * @return Array of four 3x3 matrices: [dR/dx, dR/dy, dR/dz, dR/dw]
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
 * @brief Derivative of R(q) * v w.r.t. quaternion q.
 *
 * Computes \f$\frac{\partial (R \mathbf{v})}{\partial \mathbf{q}}\f$.
 *
 * @param q Quaternion
 * @param v 3D vector
 * @return 3x4 matrix: [d(Rv)/dx, d(Rv)/dy, d(Rv)/dz, d(Rv)/dw]
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
 * @brief Derivative of R(q)^T * v w.r.t. quaternion q.
 *
 * Computes \f$\frac{\partial (R^\top \mathbf{v})}{\partial \mathbf{q}}\f$.
 *
 * @param q Quaternion
 * @param v 3D vector
 * @return 3x4 matrix: [d(R^T v)/dx, d(R^T v)/dy, d(R^T v)/dz, d(R^T v)/dw]
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
