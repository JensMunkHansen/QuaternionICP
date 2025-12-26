#!/usr/bin/env python3
"""
Ray-projection ICP demo with:
  - 7D ambient pose blocks: [qx,qy,qz,qw, tx,ty,tz]  (xyzw, scalar last)
  - Sophus/Ceres-style Manifold: Plus(T,delta) = T * Exp(delta^)  (right-multiplication)
  - DIRECT analytic AMBIENT Jacobians (1x7) for Ceres Evaluate()
  - Bidirectional ICP:
      * one-pose (single SE(3))
      * two-pose (estimate xA and xB) using only direct 7D Jacobians
  - Incidence weighting/gating based on c = n^T d (updated each inner iter, NOT differentiated)
  - Outer loop recomputes correspondences, inner loop keeps them fixed.

No local (1x6) residual Jacobians are used anywhere.
We still solve in 6D locally exactly like Ceres does:
  A_local = A_ambient @ PlusJacobian(x)
"""

import numpy as np
import trimesh

from se3_utils import (
    skew,
    quat_normalize as quat_normalize_xyzw,
    quat_mul as quat_mul_xyzw,
    quat_to_R as quat_to_R_xyzw,
    quat_exp_so3 as quat_exp_so3_xyzw,
    V_matrix_so3,
    se3_plus as se3_plus_7d,
    plus_jacobian_7x6,
    dR_dq_mats as dR_dq_mats_xyzw,
    dR_times_v_dq as dR_times_v_dq_xyzw,
    dRT_times_v_dq as dRT_times_v_dq_xyzw,
)

damping = 0#1e-6
# -----------------------------
# Incidence weighting / grazing-angle handling
# -----------------------------
ENABLE_INCIDENCE_WEIGHT = True
ENABLE_GRAZING_GATE = True
INCIDENCE_TAU = 0.2          # 0.1-0.4 typical
INCIDENCE_MODE = "sqrtabs"   # "abs" or "sqrtabs"

def incidence_weight(c: float) -> float:
    ac = abs(float(c))
    if ENABLE_GRAZING_GATE and ac < INCIDENCE_TAU:
        return 0.0
    if not ENABLE_INCIDENCE_WEIGHT:
        return 1.0
    ac = max(INCIDENCE_TAU, min(1.0, ac))
    if INCIDENCE_MODE == "abs":
        return ac
    if INCIDENCE_MODE == "sqrtabs":
        return ac ** 0.5
    raise ValueError(INCIDENCE_MODE)


# -----------------------------
# Heightfield mesh generation (LOCAL frames)
# -----------------------------

def make_heightfield_mesh(nx=45, ny=45, size=1.0, amp=0.10, freq=3.0, z0=0.0):
    xs = np.linspace(-size, size, nx)
    ys = np.linspace(-size, size, ny)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Z = z0 + amp * np.sin(freq * X) * np.sin(freq * Y)
    V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    def vid(i, j): return j * nx + i
    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = vid(i, j); v10 = vid(i + 1, j); v01 = vid(i, j + 1); v11 = vid(i + 1, j + 1)
            F.append([v00, v10, v11])
            F.append([v00, v11, v01])
    mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(F, dtype=np.int64), process=True)
    return mesh, V


# -----------------------------
# Ray casting (LOCAL intersections)
# -----------------------------

def raycast_one(mesh: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray):
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


# -----------------------------
# DIRECT ambient residual Jacobians (ONE pose)
# Return: r, J7 (1x7), c=b=n^T d
# -----------------------------

def residual_and_jac_fwd_ambient_direct(x, pS, qT, nT, dS0):
    q = quat_normalize_xyzw(x[:4])
    t = x[4:]
    R = quat_to_R_xyzw(q)

    xT = R @ pS + t
    d  = R @ dS0

    a = float(nT @ (xT - qT))
    b = float(nT @ d)

    dr_dt = nT / b

    da_dq = nT @ dR_times_v_dq_xyzw(q, pS)
    db_dq = nT @ dR_times_v_dq_xyzw(q, dS0)
    dr_dq = (da_dq * b - a * db_dq) / (b*b)

    J7 = np.hstack([dr_dq, dr_dt])
    return a / b, J7, b


def residual_and_jac_rev_ambient_direct(x, pT, qS, nS, dT0):
    q = quat_normalize_xyzw(x[:4])
    t = x[4:]
    R = quat_to_R_xyzw(q)

    u  = (pT - t)
    yS = R.T @ u
    d  = R.T @ dT0

    a = float(nS @ (yS - qS))
    b = float(nS @ d)

    dr_dt = (nS @ (-R.T)) / b

    da_dq = nS @ dRT_times_v_dq_xyzw(q, u)
    db_dq = nS @ dRT_times_v_dq_xyzw(q, dT0)
    dr_dq = (da_dq * b - a * db_dq) / (b*b)

    J7 = np.hstack([dr_dq, dr_dt])
    return a / b, J7, b


# -----------------------------
# DIRECT ambient residual Jacobians (TWO poses)
# Return: r, J7A, J7B, c=b=n^T d
# -----------------------------

def residual_and_jac_fwd_two_pose_ambient_direct(xA, xB, pA, qB_pt, nB, dA0):
    qA = quat_normalize_xyzw(xA[:4]); tA = xA[4:]; RA = quat_to_R_xyzw(qA)
    qB = quat_normalize_xyzw(xB[:4]); tB = xB[4:]; RB = quat_to_R_xyzw(qB)

    U = RA @ pA + (tA - tB)     # in world
    xBv = RB.T @ U              # in B
    u_d = RA @ dA0              # in world
    d   = RB.T @ u_d            # in B

    a = float(nB @ (xBv - qB_pt))
    b = float(nB @ d)

    # A block
    da_dtA = nB @ (RB.T)
    db_dtA = np.zeros(3)

    dx_dqA = RB.T @ dR_times_v_dq_xyzw(qA, pA)
    da_dqA = nB @ dx_dqA

    dd_dqA = RB.T @ dR_times_v_dq_xyzw(qA, dA0)
    db_dqA = nB @ dd_dqA

    dr_dqA = (da_dqA * b - a * db_dqA) / (b*b)
    dr_dtA = (da_dtA * b - a * db_dtA) / (b*b)
    J7A = np.hstack([dr_dqA, dr_dtA])

    # B block
    da_dtB = nB @ (-RB.T)
    db_dtB = np.zeros(3)

    dx_dqB = dRT_times_v_dq_xyzw(qB, U)
    da_dqB = nB @ dx_dqB

    dd_dqB = dRT_times_v_dq_xyzw(qB, u_d)
    db_dqB = nB @ dd_dqB

    dr_dqB = (da_dqB * b - a * db_dqB) / (b*b)
    dr_dtB = (da_dtB * b - a * db_dtB) / (b*b)
    J7B = np.hstack([dr_dqB, dr_dtB])

    return a / b, J7A, J7B, b


def residual_and_jac_rev_two_pose_ambient_direct(xA, xB, pB, qA_pt, nA, dB0):
    qA = quat_normalize_xyzw(xA[:4]); tA = xA[4:]; RA = quat_to_R_xyzw(qA)
    qB = quat_normalize_xyzw(xB[:4]); tB = xB[4:]; RB = quat_to_R_xyzw(qB)

    V = RB @ pB + (tB - tA)     # in world
    xAv = RA.T @ V              # in A
    v_d = RB @ dB0              # in world
    d   = RA.T @ v_d            # in A

    a = float(nA @ (xAv - qA_pt))
    b = float(nA @ d)

    # B block
    da_dtB = nA @ (RA.T)
    db_dtB = np.zeros(3)

    dx_dqB = RA.T @ dR_times_v_dq_xyzw(qB, pB)
    da_dqB = nA @ dx_dqB

    dd_dqB = RA.T @ dR_times_v_dq_xyzw(qB, dB0)
    db_dqB = nA @ dd_dqB

    dr_dqB = (da_dqB * b - a * db_dqB) / (b*b)
    dr_dtB = (da_dtB * b - a * db_dtB) / (b*b)
    J7B = np.hstack([dr_dqB, dr_dtB])

    # A block
    da_dtA = nA @ (-RA.T)
    db_dtA = np.zeros(3)

    dx_dqA = dRT_times_v_dq_xyzw(qA, V)
    da_dqA = nA @ dx_dqA

    dd_dqA = dRT_times_v_dq_xyzw(qA, v_d)
    db_dqA = nA @ dd_dqA

    dr_dqA = (da_dqA * b - a * db_dqA) / (b*b)
    dr_dtA = (da_dtA * b - a * db_dtA) / (b*b)
    J7A = np.hstack([dr_dqA, dr_dtA])

    return a / b, J7A, J7B, b


# -----------------------------
# ICP solvers (Ceres-style: assemble ambient, project with PlusJacobian, solve local)
# -----------------------------

def solve_inner_one_pose(x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0,
                         max_iters=12, damping=damping,
                         step_tol=1e-9, rel_rms_tol=1e-9):
    qT, nT, ok_fwd = corr_fwd
    qS, nS, ok_rev = corr_rev

    prev_rms = None

    for it in range(max_iters):
        P = plus_jacobian_7x6(x)  # 7x6

        rows7 = []
        rhs = []
        rs = []

        for i in range(len(P_S)):
            if not ok_fwd[i]:
                continue
            r, J7, c = residual_and_jac_fwd_ambient_direct(x, P_S[i], qT[i], nT[i], dS0)
            w = incidence_weight(c)
            if w == 0.0:
                continue
            r *= w
            rows7.append(w * J7)
            rhs.append(-r)
            rs.append(r)

        for j in range(len(P_T)):
            if not ok_rev[j]:
                continue
            r, J7, c = residual_and_jac_rev_ambient_direct(x, P_T[j], qS[j], nS[j], dT0)
            w = incidence_weight(c)
            if w == 0.0:
                continue
            r *= w
            rows7.append(w * J7)
            rhs.append(-r)
            rs.append(r)

        if len(rows7) < 6:
            break

        A7 = np.asarray(rows7)       # m x 7
        b = np.asarray(rhs)

        A = A7 @ P                   # m x 6  (local), like Ceres

        H = A.T @ A + damping * np.eye(6)
        g = A.T @ b
        step = np.linalg.lstsq(H, g, rcond=None)[0]

        x = se3_plus_7d(x, step)
        x[:4] = quat_normalize_xyzw(x[:4])

        rms = float(np.sqrt(np.mean(np.square(rs)))) if len(rs) else np.inf
        print(f"[one-pose] inner {it} rms: {rms}")
        if prev_rms is not None:
            rel = abs(prev_rms - rms) / max(1e-12, prev_rms)
            if rel < rel_rms_tol:
                break
        prev_rms = rms
        if float(np.linalg.norm(step)) < step_tol:
            break

    return x, it + 1


def solve_inner_two_pose(xA, xB, PA, PB, corr_fwd, corr_rev,
                         max_iters=12, damping=damping,
                         step_tol=1e-9, rel_rms_tol=1e-9):
    dA0 = np.array([0.0, 0.0, -1.0])
    dB0 = np.array([0.0, 0.0, -1.0])

    qB, nB, ok_fwd = corr_fwd
    qA, nA, ok_rev = corr_rev

    prev_rms = None

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

        for i in range(len(PA)):
            if not ok_fwd[i]:
                continue
            r, J7A, J7B, c = residual_and_jac_fwd_two_pose_ambient_direct(xA, xB, PA[i], qB[i], nB[i], dA0)
            w = incidence_weight(c)
            if w == 0.0:
                continue
            r *= w
            rows14.append(np.hstack([w * J7A, w * J7B]))
            rhs.append(-r)
            rs.append(r)

        for j in range(len(PB)):
            if not ok_rev[j]:
                continue
            r, J7A, J7B, c = residual_and_jac_rev_two_pose_ambient_direct(xA, xB, PB[j], qA[j], nA[j], dB0)
            w = incidence_weight(c)
            if w == 0.0:
                continue
            r *= w
            rows14.append(np.hstack([w * J7A, w * J7B]))
            rhs.append(-r)
            rs.append(r)

        if len(rows14) < 12:
            break

        A14 = np.asarray(rows14)  # m x 14
        b = np.asarray(rhs)

        A = A14 @ Pblk            # m x 12 (local), like Ceres

        H = A.T @ A + damping * np.eye(12)
        g = A.T @ b
        step = np.linalg.lstsq(H, g, rcond=None)[0]

        xA = se3_plus_7d(xA, step[:6]); xA[:4] = quat_normalize_xyzw(xA[:4])
        xB = se3_plus_7d(xB, step[6:]); xB[:4] = quat_normalize_xyzw(xB[:4])

        rms = float(np.sqrt(np.mean(np.square(rs)))) if len(rs) else np.inf
        print(f"[two-pose] inner {it} rms: {rms}")
        if prev_rms is not None:
            rel = abs(prev_rms - rms) / max(1e-12, prev_rms)
            if rel < rel_rms_tol:
                break
        prev_rms = rms
        if float(np.linalg.norm(step)) < step_tol:
            break

    return xA, xB, it + 1


def icp_one_pose(meshS, meshT, P_S, P_T, x0, outer=6, inner=12):
    dS0 = np.array([0.0, 0.0, -1.0])
    dT0 = np.array([0.0, 0.0, -1.0])

    x = x0.copy()
    x[:4] = quat_normalize_xyzw(x[:4])

    total_inner = 0
    for out in range(outer):
        R = quat_to_R_xyzw(x[:4]); t = x[4:]

        qT, nT, ok_fwd = build_corr_forward(meshT, R, t, P_S, dS0, ray_offset=0.6)
        qS, nS, ok_rev = build_corr_reverse(meshS, R, t, P_T, dT0, ray_offset=0.6)

        x, n_inner = solve_inner_one_pose(x, P_S, P_T, (qT, nT, ok_fwd), (qS, nS, ok_rev), dS0, dT0, max_iters=inner)
        total_inner += n_inner

        # report weighted rms
        rs = []
        for i in range(len(P_S)):
            if ok_fwd[i]:
                r, _, c = residual_and_jac_fwd_ambient_direct(x, P_S[i], qT[i], nT[i], dS0)
                w = incidence_weight(c)
                if w != 0.0:
                    rs.append(w * r)
        for j in range(len(P_T)):
            if ok_rev[j]:
                r, _, c = residual_and_jac_rev_ambient_direct(x, P_T[j], qS[j], nS[j], dT0)
                w = incidence_weight(c)
                if w != 0.0:
                    rs.append(w * r)
        rms = np.sqrt(np.mean(np.square(rs))) if len(rs) else np.nan
        print(f"[one-pose] outer {out:02d}  rms={rms:.6e}  #res={len(rs)}")

    print(f"[one-pose]: nOuter: {outer}, nInner: {total_inner}")
    return x


def icp_two_pose(meshA, meshB, PA, PB, xA0, xB0, outer=6, inner=12):
    dA0 = np.array([0.0, 0.0, -1.0])
    dB0 = np.array([0.0, 0.0, -1.0])

    xA = xA0.copy(); xA[:4] = quat_normalize_xyzw(xA[:4])
    xB = xB0.copy(); xB[:4] = quat_normalize_xyzw(xB[:4])

    total_inner = 0
    for out in range(outer):
        # relative A->B for correspondences: R_BA, t_BA
        RA = quat_to_R_xyzw(xA[:4]); tA = xA[4:]
        RB = quat_to_R_xyzw(xB[:4]); tB = xB[4:]
        RBA = RB.T @ RA
        tBA = RB.T @ (tA - tB)

        qBcorr, nBcorr, ok_fwd = build_corr_forward(meshB, RBA, tBA, PA, dA0, ray_offset=0.6)
        qAcorr, nAcorr, ok_rev = build_corr_reverse(meshA, RBA, tBA, PB, dB0, ray_offset=0.6)

        xA, xB, n_inner = solve_inner_two_pose(xA, xB, PA, PB, (qBcorr, nBcorr, ok_fwd), (qAcorr, nAcorr, ok_rev), max_iters=inner)
        total_inner += n_inner

        rs = []
        for i in range(len(PA)):
            if ok_fwd[i]:
                r, _, _, c = residual_and_jac_fwd_two_pose_ambient_direct(xA, xB, PA[i], qBcorr[i], nBcorr[i], dA0)
                w = incidence_weight(c)
                if w != 0.0:
                    rs.append(w * r)
        for j in range(len(PB)):
            if ok_rev[j]:
                r, _, _, c = residual_and_jac_rev_two_pose_ambient_direct(xA, xB, PB[j], qAcorr[j], nAcorr[j], dB0)
                w = incidence_weight(c)
                if w != 0.0:
                    rs.append(w * r)
        rms = np.sqrt(np.mean(np.square(rs))) if len(rs) else np.nan
        print(f"[two-pose] outer {out:02d}  rms={rms:.6e}  #res={len(rs)}")

    print(f"[two-poses]: nOuter: {outer}, nInner: {total_inner}")
    return xA, xB


def main():
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(0)

    meshS, VS = make_heightfield_mesh(z0=0.00)
    meshT, VT = make_heightfield_mesh(z0=0.02)

    P_S = VS[rng.choice(len(VS), size=350, replace=False)].copy()
    P_T = VT[rng.choice(len(VT), size=350, replace=False)].copy()

    # initial guess
    q0 = quat_exp_so3_xyzw(np.array([0.02, 0.01, -0.03]))
    t0 = np.array([-0.02, 0.01, 0.05])
    x0 = np.hstack([q0, t0])

    print("Incidence settings:",
          "weight" if ENABLE_INCIDENCE_WEIGHT else "no-weight",
          "gate" if ENABLE_GRAZING_GATE else "no-gate",
          f"tau={INCIDENCE_TAU}",
          f"mode={INCIDENCE_MODE}")

    print("\n--- One-pose ICP ---")
    x_est = icp_one_pose(meshS, meshT, P_S, P_T, x0, outer=6, inner=12)
    print("Estimated x =", x_est)

    print("\n--- Two-pose ICP ---")
    qA0 = quat_exp_so3_xyzw(np.array([0.03, -0.01, 0.02])); tA0 = np.array([0.02, 0.00, -0.03])
    qB0 = quat_exp_so3_xyzw(np.array([0.10, -0.07, 0.05])); tB0 = np.array([0.03, -0.02, 0.12])
    xA0 = np.hstack([qA0, tA0])
    xB0 = np.hstack([qB0, tB0])
    xA, xB = icp_two_pose(meshS, meshT, P_S, P_T, xA0, xB0, outer=6, inner=12)

    # print relative estimate (B<-A)
    RA = quat_to_R_xyzw(xA[:4]); tA = xA[4:]
    RB = quat_to_R_xyzw(xB[:4]); tB = xB[4:]
    tBA = RB.T @ (tA - tB)
    print("Estimated relative t_BA =", tBA)


if __name__ == "__main__":
    main()
