#!/usr/bin/env python3
"""
Test inner solver with one outer iteration (fixed correspondences).
This should produce the same residuals as the C++ implementation.
"""

import numpy as np

from test_grids import make_heightfield_mesh
from se3_utils import (
    quat_normalize as quat_normalize_xyzw,
    quat_to_R as quat_to_R_xyzw,
    se3_plus as se3_plus_7d,
    plus_jacobian_7x6,
)
from jacobians_ambient import Ambient, IncidenceParams


def raycast_one(mesh, origin, direction):
    d = direction / (np.linalg.norm(direction) + 1e-18)
    loc, idx_ray, idx_tri = mesh.ray.intersects_location(
        ray_origins=origin[None, :],
        ray_directions=d[None, :],
        multiple_hits=False
    )
    if len(loc) == 0:
        return None, None
    q = loc[0]
    n = mesh.face_normals[idx_tri[0]]
    n = n / (np.linalg.norm(n) + 1e-18)
    return q, n


def build_corr_forward(meshT, R, t, P_S, dS0, ray_offset=0.6):
    d = R @ dS0
    d = d / (np.linalg.norm(d) + 1e-18)
    qT = np.zeros((len(P_S), 3))
    nT = np.zeros((len(P_S), 3))
    ok = np.zeros((len(P_S),), dtype=bool)
    for i, pS in enumerate(P_S):
        xT = R @ pS + t
        o = xT - ray_offset * d
        q, n = raycast_one(meshT, o, d)
        if q is None:
            continue
        ok[i] = True
        qT[i] = q
        nT[i] = n
    return qT, nT, ok


def build_corr_reverse(meshS, R, t, P_T, dT0, ray_offset=0.6):
    d = R.T @ dT0
    d = d / (np.linalg.norm(d) + 1e-18)
    qS = np.zeros((len(P_T), 3))
    nS = np.zeros((len(P_T), 3))
    ok = np.zeros((len(P_T),), dtype=bool)
    for i, pT in enumerate(P_T):
        yS = R.T @ (pT - t)
        o = yS - ray_offset * d
        q, n = raycast_one(meshS, o, d)
        if q is None:
            continue
        ok[i] = True
        qS[i] = q
        nS[i] = n
    return qS, nS, ok


def solve_inner_one_pose(x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0,
                         params, max_iters=12, damping=0.0,
                         step_tol=1e-9):
    """Inner solver: iterate with fixed correspondences."""
    qT, nT, ok_fwd = corr_fwd
    qS, nS, ok_rev = corr_rev

    for it in range(max_iters):
        P = plus_jacobian_7x6(x)

        rows7 = []
        rhs = []
        rs = []

        # Forward correspondences
        for i in range(len(P_S)):
            if not ok_fwd[i]:
                continue
            _, r, J7 = Ambient.residual_and_jac_fwd(x, P_S[i], qT[i], nT[i], dS0, params)
            if not np.any(J7):
                continue
            rows7.append(J7)
            rhs.append(-r)
            rs.append(r)

        # Reverse correspondences
        for j in range(len(P_T)):
            if not ok_rev[j]:
                continue
            _, r, J7 = Ambient.residual_and_jac_rev(x, P_T[j], qS[j], nS[j], dT0, params)
            if not np.any(J7):
                continue
            rows7.append(J7)
            rhs.append(-r)
            rs.append(r)

        if len(rows7) < 6:
            break

        A7 = np.asarray(rows7)
        b = np.asarray(rhs)
        A = A7 @ P

        H = A.T @ A + damping * np.eye(6)
        g = A.T @ b
        step = np.linalg.lstsq(H, g, rcond=None)[0]

        x = se3_plus_7d(x, step)
        x[:4] = quat_normalize_xyzw(x[:4])

        rms = float(np.sqrt(np.mean(np.square(rs)))) if len(rs) else np.inf
        print(f"  inner {it}: rms={rms:.6e}, step_norm={np.linalg.norm(step):.6e}, valid={len(rs)}")

        if float(np.linalg.norm(step)) < step_tol:
            print(f"  converged at iteration {it}")
            break

    return x, it + 1, rms


def main():
    np.set_printoptions(precision=6, suppress=True)

    # No weighting for comparison with C++
    params = IncidenceParams(
        enable_weight=False,
        enable_gate=False,
        tau=0.2,
        mode="sqrtabs"
    )

    # Create grids - source and target are the same (like C++ test)
    meshS, VS = make_heightfield_mesh(z0=0.0)
    meshT, VT = make_heightfield_mesh(z0=0.0)  # Same as source

    # Use all vertices as sample points
    P_S = VS.copy()
    P_T = VT.copy()

    dS0 = np.array([0.0, 0.0, -1.0])
    dT0 = np.array([0.0, 0.0, -1.0])

    # Test 1: Identity pose - should have zero residual
    print("=== Test 1: Identity pose ===")
    x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    R = quat_to_R_xyzw(x[:4])
    t = x[4:]

    corr_fwd = build_corr_forward(meshT, R, t, P_S, dS0)
    corr_rev = build_corr_reverse(meshS, R, t, P_T, dT0)
    print(f"Forward correspondences: {np.sum(corr_fwd[2])}")
    print(f"Reverse correspondences: {np.sum(corr_rev[2])}")

    x_out, n_iters, rms = solve_inner_one_pose(
        x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0, params, max_iters=12)
    print(f"Result: iterations={n_iters}, rms={rms:.6e}")
    print()

    # Test 2: Small translation offset
    print("=== Test 2: Translation offset tx=0.05 ===")
    x = np.array([0.0, 0.0, 0.0, 1.0, 0.05, 0.0, 0.0])
    R = quat_to_R_xyzw(x[:4])
    t = x[4:]

    corr_fwd = build_corr_forward(meshT, R, t, P_S, dS0)
    corr_rev = build_corr_reverse(meshS, R, t, P_T, dT0)
    print(f"Forward correspondences: {np.sum(corr_fwd[2])}")
    print(f"Reverse correspondences: {np.sum(corr_rev[2])}")

    x_out, n_iters, rms = solve_inner_one_pose(
        x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0, params, max_iters=12)
    print(f"Result: iterations={n_iters}, rms={rms:.6e}")
    print(f"Final pose: {x_out}")
    print()

    # Test 3: Rotation offset
    print("=== Test 3: Rotation offset (0.05, 0.02, -0.03) ===")
    from se3_utils import quat_exp_so3 as quat_exp_so3_xyzw
    axis_angle = np.array([0.05, 0.02, -0.03])
    q = quat_exp_so3_xyzw(axis_angle)
    x = np.hstack([q, [0.0, 0.0, 0.0]])
    R = quat_to_R_xyzw(x[:4])
    t = x[4:]

    corr_fwd = build_corr_forward(meshT, R, t, P_S, dS0)
    corr_rev = build_corr_reverse(meshS, R, t, P_T, dT0)
    print(f"Forward correspondences: {np.sum(corr_fwd[2])}")
    print(f"Reverse correspondences: {np.sum(corr_rev[2])}")

    x_out, n_iters, rms = solve_inner_one_pose(
        x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0, params, max_iters=12)
    print(f"Result: iterations={n_iters}, rms={rms:.6e}")
    print(f"Final pose: {x_out}")
    print()


if __name__ == "__main__":
    main()
