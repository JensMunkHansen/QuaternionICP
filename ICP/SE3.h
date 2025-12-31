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
 * - Eigen::Quaterniond::coeffs() returns [x,y,z,w]
 * - We store poses as [qx, qy, qz, qw]
 * - We construct quaternions as Quaternion(w, x, y, z) (scalar-first)
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
 *   \mathrm{SE}(3) = \left\{ T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1
 *   \end{bmatrix} : R \in \mathrm{SO}(3), \mathbf{t} \in \mathbb{R}^3 \right\}
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
 * @section se3_manifold Ambient 7D Storage vs. 6D Manifold Update
 *
 * The pose is stored in a 7D **ambient** parameter vector:
 * \f[
 *   \mathbf{p} = [q_x, q_y, q_z, q_w, t_x, t_y, t_z] \in \mathbb{R}^7,
 * \f]
 * where \f$\mathbf{q}\f$ is a **unit** quaternion and \f$\mathbf{t}\f$ is translation.
 *
 * Optimization, however, is performed on the SE(3) manifold using a 6D tangent
 * increment (twist) in the Lie algebra \f$\mathfrak{se}(3)\f$:
 * \f[
 *   \delta\boldsymbol{\xi} = (\mathbf{v}, \boldsymbol{\omega})
 *   = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z] \in \mathbb{R}^6.
 * \f]
 *
 * The mapping from a tangent increment to an SE(3) perturbation uses the
 * exponential map:
 * \f[
 *   \Delta T = \exp(\widehat{\delta\boldsymbol{\xi}})
 *   = \begin{bmatrix} \Delta R & \Delta\mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}.
 * \f]
 *
 * With the **right-multiplicative (body-frame)** convention, the state update is:
 * \f[
 *   T \leftarrow T \cdot \Delta T.
 * \f]
 *
 * Writing \f$T = (R,\mathbf{t})\f$ and \f$\Delta T = (\Delta R, V\,\mathbf{v})\f$,
 * the update becomes:
 * \f[
 *   R \leftarrow R\,\Delta R, \qquad
 *   \mathbf{t} \leftarrow \mathbf{t} + R\,V\,\mathbf{v}.
 * \f]
 *
 * In quaternion form this is:
 * \f[
 *   \mathbf{q} \leftarrow \mathbf{q} \otimes \delta\mathbf{q}, \qquad
 *   \mathbf{t} \leftarrow \mathbf{t} + R(\mathbf{q})\,V\,\mathbf{v},
 * \f]
 * where \f$\delta\mathbf{q} = \exp(\boldsymbol{\omega})\f$ is the unit quaternion
 * corresponding to \f$\Delta R = \exp([\boldsymbol{\omega}]_\times)\f$.
 *
 * @subsection se3_manifold_notes Practical notes
 * - The quaternion must remain unit-length; implementations typically renormalize
 *   \f$\mathbf{q}\f$ after applying \f$\delta\mathbf{q}\f$.
 * - For small \f$\|\boldsymbol{\omega}\|\f$, use series expansions for
 *   \f$\sin(\theta)/\theta\f$, \f$(1-\cos\theta)/\theta^2\f$, etc., to avoid
 *   numerical issues, and to ensure \f$V \approx I\f$ as \f$\theta \to 0\f$.
 * - Because this is a **body-frame** increment, \f$\delta\boldsymbol{\xi}\f$
 *   is interpreted in the moving frame attached to the current pose.
 *
 * @section se3_chart_jac Chart Jacobian for Pulling Back Ambient Jacobians
 *
 * Many residual/Jacobian implementations compute derivatives w.r.t. the 7D ambient
 * parameters (q,t). Gauss-Newton, however, solves in the 6D tangent space.
 *
 * Define the Plus chart:  x_plus(delta) = Plus(x, delta).
 * The chart Jacobian at the expansion point is:
 *   P(x) = ∂ Plus(x,delta) / ∂ delta |_{delta=0}   (size 7×6).
 *
 * Given an ambient Jacobian J7 = ∂r/∂x (size 1×7), the corresponding local Jacobian is:
 *   J6 = ∂r/∂delta = J7 * P(x)    (size 1×6).
 *
 * Note: P(x) here is evaluated at delta=0, so terms like ∂t/∂w vanish at the expansion
 * point (they are proportional to v, and v=0 at delta=0).
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
    S << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
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
 * For small angles (\f$\theta < 10^{-12}\f$), uses \f$V \approx I +
 * \frac{1}{2}[\boldsymbol{\omega}]_\times\f$.
 *
 * @param w Rotation vector
 * @return 3x3 left Jacobian matrix
 */
//! [Vso3]
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
//! [Vso3]

// -----------------------------
// SE(3) manifold operations (right-multiplication)
// -----------------------------

/// SE(3) tangent vector: [v_x, v_y, v_z, w_x, w_y, w_z]
using Tangent6 = Vector6;

/**
 * @brief SE(3) Plus operation using right-multiplication (body-frame perturbation).
 *
 * Applies a tangent-space increment via the SE(3) exponential map:
 * \f[
 *   T_{\text{new}} = T \cdot \exp(\widehat{\delta\boldsymbol{\xi}}), \quad
 *   \delta\boldsymbol{\xi} = (\mathbf{v}, \boldsymbol{\omega}) \in \mathbb{R}^6
 * \f]
 *
 * With \f$\Delta R = \exp([\boldsymbol{\omega}]_\times)\f$ and the SE(3) coupling
 * matrix \f$V(\boldsymbol{\omega})\f$ (SO(3) left Jacobian), the component update is:
 * \f[
 *   R_{\text{new}} = R \cdot \Delta R, \qquad
 *   \mathbf{t}_{\text{new}} = \mathbf{t} + R \cdot V(\boldsymbol{\omega}) \cdot \mathbf{v}.
 * \f]
 *
 * Gotchas / conventions:
 * - **Right (body-frame) perturbation:** the quaternion update is
 *   \f$\mathbf{q}_{\text{new}} = \mathbf{q} \otimes \delta\mathbf{q}\f$
 *   (not \f$\delta\mathbf{q} \otimes \mathbf{q}\f$).
 * - **Translation uses the current rotation:** the translational increment is rotated by
 *   the *current* \f$R(\mathbf{q})\f$ (not \f$R(\mathbf{q}_{\text{new}})\f$).
 * - **Quaternion sign ambiguity:** \f$\mathbf{q}\f$ and \f$-\mathbf{q}\f$ represent the
 *   same rotation. The update may flip the quaternion sign without affecting
 *   \f$R(\mathbf{q})\f$ or any geometric quantity.
 *
 * @param x     Current pose [qx, qy, qz, qw, tx, ty, tz]
 * @param delta Tangent increment [v_x, v_y, v_z, w_x, w_y, w_z]
 * @return Updated pose
 */
inline Pose7 se3Plus(const Pose7& x, const Tangent6& delta)
{
    // Extract quaternion and translation
    Quaternion q(x[3], x[0], x[1], x[2]); // w, x, y, z
    q.normalize();
    Vector3 t = x.tail<3>();

    // Extract tangent components
    // w is axis-angle in radians; v is translation in the tangent space (same length units as t).
    Vector3 v = delta.head<3>(); // translation part
    Vector3 w = delta.tail<3>(); // rotation part

    // Right-multiplicative (body-frame) update:
    //   q_new = q ⊗ Exp(w)
    //   t_new = t + R(q) * V(w) * v
    // Note: R(q) is the rotation of the *current* pose (not q_new).
    // For small ||w||, V(w) ≈ I and the update reduces to t_new ≈ t + R(q)*v.

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
 * @brief SE(3) chart Jacobian P(x) for the right-multiplicative Plus operation.
 *
 * Let Plus(x, delta) apply a body-frame (right) perturbation:
 *   T_new = T(x) * Exp(delta^),  with delta = (v, w) ∈ R^6.
 *
 * This function returns the chart Jacobian evaluated at the linearization point:
 *   P(x) = ∂ Plus(x, delta) / ∂ delta |_{delta = 0}    (size 7×6),
 * mapping a small tangent increment δξ ∈ R^6 to a first-order change in the
 * 7D ambient parameterization (quaternion + translation).
 *
 * Block structure at delta = 0:
 * - Translation part:
 *     ∂t/∂v = R(q)        because V(0) = I  and  t_new = t + R * V(w) * v.
 *     ∂t/∂w = 0           at the expansion point, since this term is proportional
 *                         to v (and v = 0 at delta = 0 in the partial derivative).
 *
 * - Quaternion part (right perturbation):
 *     q_new = q ⊗ Exp(w).
 *   Using the small-angle approximation Exp(w) ≈ [1, 0.5 w] in quaternion form,
 *   we obtain (with q = [v_q, s]):
 *     ∂v_q/∂w = 0.5 ( s I + [v_q]_× ),
 *     ∂s  /∂w = -0.5 v_q^T.
 *
 * Sign convention note:
 * - The above expressions are for **right-multiplication** (q_new = q ⊗ δq).
 * - If using **left-multiplication** (q_new = δq ⊗ q), the cross-term changes sign
 *   (the skew term appears with the opposite sign).
 *
 * Quaternion renormalization is treated as a no-op at first order.
 *
 * @note
 * For the full 7x6 SE(3) chart Jacobian away from the linearization point,
 * see @ref se3_full_chart_jacobian.
 *
 * @param x Current pose [qx, qy, qz, qw, tx, ty, tz] (quaternion stored as x,y,z,w)
 * @return  7×6 chart Jacobian P(x) evaluated at delta = 0
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

    // Quaternion block: ∂q_new/∂w at delta=0 for right perturbation q_new = q ⊗ Exp(w).
    // Using Exp(w) ≈ [1, 0.5 w] (unit quaternion), we have:
    //   q ⊗ [1, 0.5 w]  =>  d(v_q)/dw = 0.5 * (s I + [v_q]_x),
    //                      d(s)/dw    = -0.5 * v_q^T,
    // where q = [v_q, s] (vector part v_q = [x,y,z], scalar part s = q_w).
    // Note: for left perturbation q_new = Exp(w) ⊗ q, the skew/cross term changes sign.
    Matrix4x3 dq_dw;
    dq_dw.topRows<3>() = 0.5 * (s * Matrix3::Identity() + skew(vq));
    dq_dw.row(3) = -0.5 * vq.transpose();

    // dt/dv (at delta=0, V=I): R
    J.block<4, 3>(0, 3) = dq_dw; // quaternion rows, rotation columns
    J.block<3, 3>(4, 0) = R;     // translation rows, translation columns

    return J;
}

/**
 * @brief Project 7D ambient Jacobian to 6D local (tangent space) Jacobian.
 *
 * Computes \f$J_{\text{local}} = J_{\text{ambient}} \cdot P(x)\f$ where
 * \f$P(x)\f$ is the Plus Jacobian at pose \f$x\f$.
 *
 * @note This is a pullback through the Plus chart, not a Euclidean projection
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

    dRdx << 0.0, 2 * y, 2 * z, 2 * y, -4 * x, -2 * w, 2 * z, 2 * w, -4 * x;

    dRdy << -4 * y, 2 * x, 2 * w, 2 * x, 0.0, 2 * z, -2 * w, 2 * z, -4 * y;

    dRdz << -4 * z, -2 * w, 2 * x, 2 * w, -4 * z, 2 * y, 2 * x, 2 * y, 0.0;

    dRdw << 0.0, -2 * z, 2 * y, 2 * z, 0.0, -2 * x, -2 * y, 2 * x, 0.0;

    return { dRdx, dRdy, dRdz, dRdw };
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
