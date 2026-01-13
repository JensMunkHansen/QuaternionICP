#!/usr/bin/env python3
"""
Ray-projection ICP with 6D tangent parameterization and left-multiplication.

Key differences from RayProjection7D.py:
  - State is (R, t) not a 7D vector
  - Jacobians are 1x6 directly in tangent space (no ambient + projection)
  - Update is left-multiplication: T_new = exp(delta^) * T

Tangent vector: delta = [rho; phi] where rho is translation, phi is rotation (axis-angle)
"""

import numpy as np

from test_grids import make_heightfield_mesh
from icp_utils import build_corr_forward, build_corr_reverse
from jacobians_tangent import TangentSimplified, IncidenceParams


# -----------------------------
# SO(3) / SE(3) for left-multiplication
# -----------------------------

def skew(w):
    """Skew-symmetric matrix [w]_x such that [w]_x v = w × v"""
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=float)


def so3_exp(phi):
    """Exponential map: axis-angle -> rotation matrix (Rodrigues)"""
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3) + skew(phi) + 0.5*skew(phi)@skew(phi)
        #return np.eye(3) + skew(phi)
    K = skew(phi / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def V_so3(phi):
    """Left Jacobian V(phi) used in SE(3) exponential."""
    theta = np.linalg.norm(phi)
    Phi = skew(phi)
    Phi2 = Phi @ Phi

    if theta < 1e-12:
        return np.eye(3) + 0.5 * Phi + (1.0/6.0) * Phi2

    theta2 = theta * theta
    A = (1 - np.cos(theta)) / theta2
    B = (theta - np.sin(theta)) / (theta2 * theta)
    return np.eye(3) + A * Phi + B * Phi2


def se3_plus_left(R, t, delta):
    """
    Left SE(3) update: T_new = exp(delta^) * T

    delta = [rho; phi] where rho is translation tangent, phi is rotation tangent
    """
    rho, phi = delta[:3], delta[3:]
    R_delta = so3_exp(phi)
    V = V_so3(phi)
    return R_delta @ R, V @ rho + R_delta @ t


# For initial pose setup (matching RayProjection7D.py)
def quat_from_axis_angle(w):
    """Axis-angle to quaternion [x,y,z,w]"""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.array([0, 0, 0, 1], dtype=float)
    half = 0.5 * theta
    v = (w / theta) * np.sin(half)
    return np.array([v[0], v[1], v[2], np.cos(half)])


def R_from_quat(q):
    """Quaternion [x,y,z,w] to rotation matrix"""
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def quat_from_R(R):
    """Rotation matrix to quaternion [x,y,z,w]"""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1)
        return np.array([(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        return np.array([0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s])
    elif R[1,1] > R[2,2]:
        s = 2 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s])
    else:
        s = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s])


# -----------------------------
# ICP solver
# -----------------------------

INCIDENCE_PARAMS = IncidenceParams(enable_weight=True, enable_gate=True, tau=0.2, mode="sqrtabs")


def solve_one_pose(meshS, meshT, P_S, P_T, R, t, dS0, dT0,
                   params=INCIDENCE_PARAMS, outer=6, inner=12):
    """
    Bidirectional ICP with 6D tangent Jacobians and left-multiplication update.
    """
    total_inner = 0
    rms = np.inf

    for out in range(outer):
        print(f"  outer {out}:")

        # Build correspondences at current pose
        corr_fwd = build_corr_forward(meshT, R, t, P_S, dS0)
        corr_rev = build_corr_reverse(meshS, R, t, P_T, dT0)
        qT, nT, ok_fwd = corr_fwd
        qS, nS, ok_rev = corr_rev
        print(f"    fwd_corrs={np.sum(ok_fwd)}, rev_corrs={np.sum(ok_rev)}")

        # Inner loop with fixed correspondences
        prev_rms = None
        for it in range(inner):
            # Build linear system directly in 6D tangent space
            rows, rhs, residuals = [], [], []

            # Forward: source points -> target surface
            for i in range(len(P_S)):
                if not ok_fwd[i]:
                    continue
                r, J = TangentSimplified.residual_and_jac_fwd(R, t, P_S[i], qT[i], nT[i], dS0, params)
                if np.any(J):
                    rows.append(J)
                    rhs.append(-r)
                    residuals.append(r)

            # Reverse: target points -> source surface
            for j in range(len(P_T)):
                if not ok_rev[j]:
                    continue
                r, J = TangentSimplified.residual_and_jac_rev(R, t, P_T[j], qS[j], nS[j], dT0, params)
                if np.any(J):
                    rows.append(J)
                    rhs.append(-r)
                    residuals.append(r)

            if len(rows) < 6:
                break

            # Solve normal equations: (A^T A) delta = A^T b
            A = np.asarray(rows)
            b = np.asarray(rhs)
            # delta = np.linalg.lstsq(A.T @ A, A.T @ b, rcond=None)[0]

            delta = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Left-multiplication update
            R, t = se3_plus_left(R, t, delta)

            rms = np.sqrt(np.mean(np.square(residuals)))
            print(f"    inner {it}: rms={rms:.6e}, valid={len(residuals)}")

            # Convergence checks
            step_norm = np.linalg.norm(delta)
            if step_norm < 1e-9:
                break
            if prev_rms is not None and abs(prev_rms - rms) / max(prev_rms, 1e-12) < 1e-9:
                break
            prev_rms = rms

        n_inner = it + 1
        total_inner += n_inner
        print(f"    outer {out}: rms={rms:.6e}, valid={len(residuals)}, inner_iters={n_inner}")

    print(f"Total inner iterations: {total_inner}")
    return R, t, total_inner, rms


def main():
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(0)

    # Same setup as RayProjection7D.py
    meshS, VS = make_heightfield_mesh(z0=0.00)
    meshT, VT = make_heightfield_mesh(z0=0.02)
    P_S = VS[rng.choice(len(VS), size=350, replace=False)].copy()
    P_T = VT[rng.choice(len(VT), size=350, replace=False)].copy()

    # Initial pose as (R, t)
    q0 = quat_from_axis_angle(np.array([0.02, 0.01, -0.03]))
    R0 = R_from_quat(q0)
    t0 = np.array([-0.02, 0.01, 0.05])

    # Ray directions in local frames
    dS0 = np.array([0.0, 0.0, -1.0])
    dT0 = np.array([0.0, 0.0, -1.0])

    print(f"Incidence settings: {INCIDENCE_PARAMS}")
    print("\n--- One-pose ICP ---")

    R_est, t_est, n_inner, rms = solve_one_pose(
        meshS, meshT, P_S, P_T, R0, t0, dS0, dT0, INCIDENCE_PARAMS)

    # Output in same format as RayProjection7D.py
    q_est = quat_from_R(R_est)
    print("Estimated x =", np.hstack([q_est, t_est]))

    # Assertions
    assert n_inner == 11, f"expected 11 inner iterations, got {n_inner}"
    assert rms < 1e-15, f"expected rms < 1e-15, got {rms}"
    print("\n*** All assertions passed ***")


if __name__ == "__main__":
    main()
