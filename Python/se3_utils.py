"""
SE(3) and SO(3) utilities for quaternion-based pose representation.

Quaternion convention: [x, y, z, w] (scalar last) - matches Eigen coeffs() storage.
SE(3) pose: 7D vector [qx, qy, qz, qw, tx, ty, tz]
Tangent space: 6D vector [v_x, v_y, v_z, w_x, w_y, w_z] (translation, rotation)

Uses right-multiplication (body/moving frame) convention:
  T_new = T * Exp(delta^)
"""

import numpy as np


# -----------------------------
# SO(3) utilities
# -----------------------------

def skew(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector: [w]_x such that [w]_x v = w x v"""
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion [x,y,z,w]. Returns identity if near-zero."""
    n = np.linalg.norm(q)
    if n < 1e-18:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_mul(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q * p, both in [x,y,z,w] format."""
    qv, qs = q[:3], q[3]
    pv, ps = p[:3], p[3]
    v = qs * pv + ps * qv + np.cross(qv, pv)
    s = qs * ps - np.dot(qv, pv)
    return np.array([v[0], v[1], v[2], s], dtype=float)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to 3x3 rotation matrix."""
    q = quat_normalize(q)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=float)


def quat_exp_so3(w: np.ndarray) -> np.ndarray:
    """
    SO(3) exponential map: axis-angle w -> quaternion [x,y,z,w].

    For small angles, uses first-order Taylor expansion.
    """
    th = np.linalg.norm(w)
    if th < 1e-12:
        v = 0.5 * w
        return quat_normalize(np.array([v[0], v[1], v[2], 1.0], dtype=float))
    axis = w / th
    half = 0.5 * th
    s = np.sin(half)
    v = axis * s
    return np.array([v[0], v[1], v[2], np.cos(half)], dtype=float)


def V_matrix_so3(w: np.ndarray) -> np.ndarray:
    """
    Left Jacobian of SO(3): V such that Exp(w^) = I + V @ w^ for rotation.
    Used in SE(3) Plus: t_new = t + R @ V @ v
    """
    W = skew(w)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + 0.5 * W
    A = np.sin(th) / th
    B = (1 - np.cos(th)) / (th * th)
    C = (1 - A) / (th * th)
    return np.eye(3) + B * W + C * (W @ W)


# -----------------------------
# SE(3) manifold operations (right-multiplication)
# -----------------------------

def se3_plus(x: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    SE(3) Plus operation (right-multiplication):
      x_new = x * Exp(delta^)

    x: 7D pose [qx, qy, qz, qw, tx, ty, tz]
    delta: 6D tangent [v_x, v_y, v_z, w_x, w_y, w_z]
    """
    q = quat_normalize(x[:4])
    t = x[4:].copy()
    v = delta[:3]
    w = delta[3:]

    dq = quat_exp_so3(w)
    q_plus = quat_normalize(quat_mul(q, dq))

    R = quat_to_R(q)
    V = V_matrix_so3(w)
    t_plus = t + R @ (V @ v)

    return np.hstack([q_plus, t_plus])


def plus_jacobian_7x6(x: np.ndarray) -> np.ndarray:
    """
    PlusJacobian: d Plus(x, delta) / d delta |_{delta=0}

    Returns 7x6 matrix mapping tangent increments to ambient changes.
    Used as: J_local = J_ambient @ PlusJacobian
    """
    q = quat_normalize(x[:4])
    vq = q[:3]
    s = q[3]
    R = quat_to_R(q)

    J = np.zeros((7, 6), dtype=float)

    # dq/dw (at delta=0)
    dq_dw = np.zeros((4, 3), dtype=float)
    dq_dw[:3, :] = s * np.eye(3) + skew(vq)
    dq_dw[3, :] = -vq
    dq_dw *= 0.5

    J[:4, 3:6] = dq_dw   # quaternion depends on rotation part
    J[4:7, 0:3] = R      # translation depends on translation part (at delta=0, V=I)
    return J


# -----------------------------
# Quaternion derivatives (for ambient Jacobians)
# -----------------------------

def dR_dq_mats(q: np.ndarray):
    """
    Returns (dR/dx, dR/dy, dR/dz, dR/dw) - four 3x3 matrices.
    These are derivatives of the rotation matrix w.r.t. each quaternion component.
    """
    q = quat_normalize(q)
    x, y, z, w = q

    dRdx = np.array([[0.0,  2*y,  2*z],
                     [2*y, -4*x, -2*w],
                     [2*z,  2*w, -4*x]], dtype=float)

    dRdy = np.array([[-4*y, 2*x,  2*w],
                     [ 2*x, 0.0,  2*z],
                     [-2*w, 2*z, -4*y]], dtype=float)

    dRdz = np.array([[-4*z, -2*w, 2*x],
                     [ 2*w, -4*z, 2*y],
                     [ 2*x,  2*y, 0.0]], dtype=float)

    dRdw = np.array([[0.0, -2*z,  2*y],
                     [2*z,  0.0, -2*x],
                     [-2*y, 2*x,  0.0]], dtype=float)
    return dRdx, dRdy, dRdz, dRdw


def dR_times_v_dq(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Derivative of R(q) @ v w.r.t. quaternion q.
    Returns 3x4 matrix: [d(Rv)/dx, d(Rv)/dy, d(Rv)/dz, d(Rv)/dw]
    """
    dRdx, dRdy, dRdz, dRdw = dR_dq_mats(q)
    return np.column_stack([dRdx @ v, dRdy @ v, dRdz @ v, dRdw @ v])


def dRT_times_v_dq(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Derivative of R(q).T @ v w.r.t. quaternion q.
    Returns 3x4 matrix.
    """
    dRdx, dRdy, dRdz, dRdw = dR_dq_mats(q)
    return np.column_stack([dRdx.T @ v, dRdy.T @ v, dRdz.T @ v, dRdw.T @ v])
