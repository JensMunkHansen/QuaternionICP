#!/usr/bin/env python3
"""
Shared ICP utility functions for ray-projection ICP.

Functions:
- raycast_one: Single ray intersection with mesh
- build_corr_forward: Build forward correspondences (source->target)
- build_corr_reverse: Build reverse correspondences (target->source)
- solve_inner_one_pose: One-pose inner loop solver with fixed correspondences
- solve_inner_two_pose: Two-pose inner loop solver with fixed correspondences
- icp_one_pose: Full one-pose ICP with outer loop (correspondence updates)
- icp_two_pose: Full two-pose ICP with outer loop (correspondence updates)
"""

import numpy as np
from se3_utils import (
    quat_normalize as quat_normalize_xyzw,
    quat_to_R as quat_to_R_xyzw,
    se3_plus as se3_plus_7d,
    plus_jacobian_7x6,
)
from jacobians_ambient import Ambient


def raycast_one(mesh, origin, direction):
    """
    Cast a single ray and find the first intersection.

    Args:
        mesh: Trimesh mesh object
        origin: Ray origin (3,)
        direction: Ray direction (3,)

    Returns:
        (q, n): Intersection point and normal, or (None, None) if no hit
    """
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
    """
    Build forward correspondences: source points -> target mesh.

    Args:
        meshT: Target mesh
        R: Rotation matrix (3x3)
        t: Translation vector (3,)
        P_S: Source points (N, 3)
        dS0: Ray direction in source frame (3,)
        ray_offset: Ray starting offset distance

    Returns:
        (qT, nT, ok): Target points (N,3), normals (N,3), validity mask (N,)
    """
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
    """
    Build reverse correspondences: target points -> source mesh.

    Args:
        meshS: Source mesh
        R: Rotation matrix (3x3)
        t: Translation vector (3,)
        P_T: Target points (N, 3)
        dT0: Ray direction in target frame (3,)
        ray_offset: Ray starting offset distance

    Returns:
        (qS, nS, ok): Source points (N,3), normals (N,3), validity mask (N,)
    """
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
                         step_tol=1e-9, rel_rms_tol=1e-9, verbose=False):
    """
    Inner solver: iterate with fixed correspondences.

    Args:
        x: Initial pose (7,) [qx, qy, qz, qw, tx, ty, tz]
        P_S: Source points (N, 3)
        P_T: Target points (M, 3)
        corr_fwd: Forward correspondences (qT, nT, ok_fwd)
        corr_rev: Reverse correspondences (qS, nS, ok_rev)
        dS0: Ray direction in source frame (3,)
        dT0: Ray direction in target frame (3,)
        params: IncidenceParams for geometry weighting
        max_iters: Maximum inner iterations
        damping: LM damping factor
        step_tol: Convergence tolerance on step norm
        rel_rms_tol: Relative RMS change convergence tolerance
        verbose: Print per-iteration info

    Returns:
        (x, iterations, rms, valid_count)
    """
    qT, nT, ok_fwd = corr_fwd
    qS, nS, ok_rev = corr_rev

    prev_rms = None

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

        if verbose:
            print(f"    inner {it}: rms={rms:.6e}, valid={len(rs)}")

        # Check relative RMS convergence
        if prev_rms is not None:
            rel = abs(prev_rms - rms) / max(1e-12, prev_rms)
            if rel < rel_rms_tol:
                break
        prev_rms = rms

        # Check step size convergence
        if float(np.linalg.norm(step)) < step_tol:
            break

    return x, it + 1, rms, len(rs)


def solve_inner_two_pose(xA, xB, PA, PB, corr_fwd, corr_rev, dA0, dB0,
                         params, max_iters=12, damping=0.0,
                         step_tol=1e-9, rel_rms_tol=1e-9, verbose=False):
    """
    Two-pose inner solver: iterate with fixed correspondences.

    Args:
        xA: Initial pose A (7,) [qx, qy, qz, qw, tx, ty, tz]
        xB: Initial pose B (7,)
        PA: Points in frame A (N, 3)
        PB: Points in frame B (M, 3)
        corr_fwd: Forward correspondences (qB, nB, ok_fwd) - A points hitting B surface
        corr_rev: Reverse correspondences (qA, nA, ok_rev) - B points hitting A surface
        dA0: Ray direction in frame A (3,)
        dB0: Ray direction in frame B (3,)
        params: IncidenceParams for geometry weighting
        max_iters: Maximum inner iterations
        damping: LM damping factor
        step_tol: Convergence tolerance on step norm
        rel_rms_tol: Relative RMS change convergence tolerance
        verbose: Print per-iteration info

    Returns:
        (xA, xB, iterations, rms, valid_count)
    """
    qB, nB, ok_fwd = corr_fwd
    qA, nA, ok_rev = corr_rev

    prev_rms = None
    rms = np.inf
    valid_count = 0

    for it in range(max_iters):
        PAj = plus_jacobian_7x6(xA)
        PBj = plus_jacobian_7x6(xB)
        Pblk = np.block([
            [PAj, np.zeros((7, 6))],
            [np.zeros((7, 6)), PBj]
        ])  # 14x12

        rows14 = []
        rhs = []
        rs = []

        # Forward correspondences (A -> B)
        for i in range(len(PA)):
            if not ok_fwd[i]:
                continue
            _, r, J7A, J7B = Ambient.residual_and_jac_fwd_two_pose(xA, xB, PA[i], qB[i], nB[i], dA0, params)
            if not np.any(J7A):
                continue
            rows14.append(np.hstack([J7A, J7B]))
            rhs.append(-r)
            rs.append(r)

        # Reverse correspondences (B -> A)
        for j in range(len(PB)):
            if not ok_rev[j]:
                continue
            _, r, J7A, J7B = Ambient.residual_and_jac_rev_two_pose(xA, xB, PB[j], qA[j], nA[j], dB0, params)
            if not np.any(J7A):
                continue
            rows14.append(np.hstack([J7A, J7B]))
            rhs.append(-r)
            rs.append(r)

        if len(rows14) < 12:
            break

        A14 = np.asarray(rows14)  # m x 14
        b = np.asarray(rhs)

        A = A14 @ Pblk  # m x 12 (local)

        H = A.T @ A + damping * np.eye(12)
        g = A.T @ b
        step = np.linalg.lstsq(H, g, rcond=None)[0]

        xA = se3_plus_7d(xA, step[:6])
        xA[:4] = quat_normalize_xyzw(xA[:4])
        xB = se3_plus_7d(xB, step[6:])
        xB[:4] = quat_normalize_xyzw(xB[:4])

        rms = float(np.sqrt(np.mean(np.square(rs)))) if len(rs) else np.inf
        valid_count = len(rs)

        if verbose:
            print(f"    inner {it}: rms={rms:.6e}, valid={valid_count}")

        # Check relative RMS convergence
        if prev_rms is not None:
            rel = abs(prev_rms - rms) / max(1e-12, prev_rms)
            if rel < rel_rms_tol:
                break
        prev_rms = rms

        # Check step size convergence
        if float(np.linalg.norm(step)) < step_tol:
            break

    return xA, xB, it + 1, rms, valid_count


def icp_one_pose(meshS, meshT, P_S, P_T, x0,
                 params, max_outer_iterations=6, inner=12,
                 damping=0.0, step_tol=1e-9, verbose=False):
    """
    Full ICP solver with outer loop (correspondence updates) and inner loop.

    Args:
        meshS: Source mesh
        meshT: Target mesh
        P_S: Source sample points (N, 3)
        P_T: Target sample points (M, 3)
        x0: Initial pose (7,) [qx, qy, qz, qw, tx, ty, tz]
        params: IncidenceParams for geometry weighting
        max_outer_iterations: Maximum outer iterations
        inner: Maximum inner iterations per outer iteration
        damping: LM damping factor
        step_tol: Inner loop convergence tolerance
        verbose: Print per-iteration info

    Returns:
        (x_final, total_inner_iterations, final_rms)
    """
    dS0 = np.array([0.0, 0.0, -1.0])
    dT0 = np.array([0.0, 0.0, -1.0])

    x = x0.copy()
    x[:4] = quat_normalize_xyzw(x[:4])

    total_inner = 0
    final_rms = np.nan

    for out in range(max_outer_iterations):
        if verbose:
            print(f"  outer {out}:")

        R = quat_to_R_xyzw(x[:4])
        t = x[4:]

        # Recompute correspondences at current pose
        corr_fwd = build_corr_forward(meshT, R, t, P_S, dS0, ray_offset=0.6)
        corr_rev = build_corr_reverse(meshS, R, t, P_T, dT0, ray_offset=0.6)

        qT, nT, ok_fwd = corr_fwd
        qS, nS, ok_rev = corr_rev

        if verbose:
            print(f"    fwd_corrs={np.sum(ok_fwd)}, rev_corrs={np.sum(ok_rev)}")

        # Run inner loop with these correspondences
        x, n_inner, rms, valid = solve_inner_one_pose(
            x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0,
            params, max_iters=inner, damping=damping, step_tol=step_tol, verbose=verbose)
        total_inner += n_inner
        final_rms = rms

        if verbose:
            print(f"    outer {out}: rms={rms:.6e}, valid={valid}, inner_iters={n_inner}")

    if verbose:
        print(f"Total inner iterations: {total_inner}")

    return x, total_inner, final_rms


def icp_two_pose(meshA, meshB, PA, PB, xA0, xB0,
                 params, outer=6, inner=12,
                 damping=0.0, step_tol=1e-9, verbose=True):
    """
    Full two-pose ICP solver with outer loop (correspondence updates) and inner loop.

    Args:
        meshA: Mesh A
        meshB: Mesh B
        PA: Sample points in frame A (N, 3)
        PB: Sample points in frame B (M, 3)
        xA0: Initial pose A (7,) [qx, qy, qz, qw, tx, ty, tz]
        xB0: Initial pose B (7,)
        params: IncidenceParams for geometry weighting
        outer: Maximum outer iterations
        inner: Maximum inner iterations per outer iteration
        damping: LM damping factor
        step_tol: Inner loop convergence tolerance
        verbose: Print per-iteration info

    Returns:
        (xA_final, xB_final, total_inner_iterations, final_rms)
    """
    dA0 = np.array([0.0, 0.0, -1.0])
    dB0 = np.array([0.0, 0.0, -1.0])

    xA = xA0.copy()
    xA[:4] = quat_normalize_xyzw(xA[:4])
    xB = xB0.copy()
    xB[:4] = quat_normalize_xyzw(xB[:4])

    total_inner = 0
    rms = np.nan

    for out in range(outer):
        if verbose:
            print(f"  outer {out}:")

        # Compute relative transform A->B for correspondences
        RA = quat_to_R_xyzw(xA[:4])
        tA = xA[4:]
        RB = quat_to_R_xyzw(xB[:4])
        tB = xB[4:]
        RBA = RB.T @ RA
        tBA = RB.T @ (tA - tB)

        # Build correspondences
        corr_fwd = build_corr_forward(meshB, RBA, tBA, PA, dA0, ray_offset=0.6)
        corr_rev = build_corr_reverse(meshA, RBA, tBA, PB, dB0, ray_offset=0.6)

        if verbose:
            _, _, ok_fwd = corr_fwd
            _, _, ok_rev = corr_rev
            print(f"    fwd_corrs={np.sum(ok_fwd)}, rev_corrs={np.sum(ok_rev)}")

        # Run inner loop with these correspondences
        xA, xB, n_inner, rms, valid = solve_inner_two_pose(
            xA, xB, PA, PB, corr_fwd, corr_rev, dA0, dB0,
            params, max_iters=inner, damping=damping, step_tol=step_tol, verbose=verbose)
        total_inner += n_inner

        if verbose:
            print(f"    outer {out}: rms={rms:.6e}, valid={valid}, inner_iters={n_inner}")

    if verbose:
        print(f"Total inner iterations: {total_inner}")

    return xA, xB, total_inner, rms
