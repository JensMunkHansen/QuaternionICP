#!/usr/bin/env python3
"""
Ray-projection ICP demo with:
  - 7D ambient pose blocks: [qx,qy,qz,qw, tx,ty,tz]  (xyzw, scalar last)
  - Sophus/Ceres-style Manifold: Plus(T,delta) = T * Exp(delta^)  (right-multiplication)
  - Simplified AMBIENT Jacobians (1x7) - treats denominator as constant for dq
  - Bidirectional ICP: one-pose and two-pose
  - Incidence weighting/gating based on c = n^T d

We solve in 6D locally exactly like Ceres does:
  A_local = A_ambient @ PlusJacobian(x)
"""

import numpy as np

from se3_utils import (
    quat_normalize as quat_normalize_xyzw,
    quat_to_R as quat_to_R_xyzw,
    quat_exp_so3 as quat_exp_so3_xyzw,
    se3_plus as se3_plus_7d,
    plus_jacobian_7x6,
)
from jacobians_ambient import Ambient, IncidenceParams
from test_grids import make_heightfield_mesh
from icp_utils import build_corr_forward, build_corr_reverse, icp_one_pose

damping = 0#1e-6

# Incidence weighting parameters
INCIDENCE_PARAMS = IncidenceParams(
    enable_weight=True,
    enable_gate=True,
    tau=0.2,
    mode="sqrtabs"
)


# -----------------------------
# ICP solvers (Ceres-style: assemble ambient, project with PlusJacobian, solve local)
# -----------------------------

def solve_inner_one_pose(x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0,
                         params=INCIDENCE_PARAMS,
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
            _, r, J7 = Ambient.residual_and_jac_fwd(x, P_S[i], qT[i], nT[i], dS0, params)
            if not np.any(J7):
                continue
            rows7.append(J7)
            rhs.append(-r)
            rs.append(r)

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
                         params=INCIDENCE_PARAMS,
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
            _, r, J7A, J7B = Ambient.residual_and_jac_fwd_two_pose(xA, xB, PA[i], qB[i], nB[i], dA0, params)
            if not np.any(J7A):
                continue
            rows14.append(np.hstack([J7A, J7B]))
            rhs.append(-r)
            rs.append(r)

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


def icp_two_pose(meshA, meshB, PA, PB, xA0, xB0, params=INCIDENCE_PARAMS, outer=6, inner=12):
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

        xA, xB, n_inner = solve_inner_two_pose(xA, xB, PA, PB, (qBcorr, nBcorr, ok_fwd), (qAcorr, nAcorr, ok_rev), params=params, max_iters=inner)
        total_inner += n_inner

        rs = []
        for i in range(len(PA)):
            if ok_fwd[i]:
                _, r, J7A, _ = Ambient.residual_and_jac_fwd_two_pose(xA, xB, PA[i], qBcorr[i], nBcorr[i], dA0, params)
                if np.any(J7A):
                    rs.append(r)
        for j in range(len(PB)):
            if ok_rev[j]:
                _, r, J7A, _ = Ambient.residual_and_jac_rev_two_pose(xA, xB, PB[j], qAcorr[j], nAcorr[j], dB0, params)
                if np.any(J7A):
                    rs.append(r)
        rms = np.sqrt(np.mean(np.square(rs))) if len(rs) else np.nan
        print(f"[two-pose] outer {out:02d}  rms={rms:.6e}  #res={len(rs)}")

    print(f"[two-poses]: nOuter: {outer}, nInner: {total_inner}")
    return xA, xB, total_inner, rms


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

    print(f"Incidence settings: {INCIDENCE_PARAMS}")

    print("\n--- One-pose ICP ---")
    x_est, n_inner_1, rms_1 = icp_one_pose(meshS, meshT, P_S, P_T, x0, INCIDENCE_PARAMS, max_outer_iterations=6, inner=12, verbose=True)
    print("Estimated x =", x_est)

    print("\n--- Two-pose ICP ---")
    qA0 = quat_exp_so3_xyzw(np.array([0.03, -0.01, 0.02])); tA0 = np.array([0.02, 0.00, -0.03])
    qB0 = quat_exp_so3_xyzw(np.array([0.10, -0.07, 0.05])); tB0 = np.array([0.03, -0.02, 0.12])
    xA0 = np.hstack([qA0, tA0])
    xB0 = np.hstack([qB0, tB0])
    xA, xB, n_inner_2, rms_2 = icp_two_pose(meshS, meshT, P_S, P_T, xA0, xB0, outer=6, inner=12)

    # print relative estimate (B<-A)
    RA = quat_to_R_xyzw(xA[:4]); tA = xA[4:]
    RB = quat_to_R_xyzw(xB[:4]); tB = xB[4:]
    tBA = RB.T @ (tA - tB)
    print("Estimated relative t_BA =", tBA)

    # Regression assertions
    assert n_inner_1 == 11, f"one-pose inner iterations: expected 11, got {n_inner_1}"
    assert n_inner_2 == 12, f"two-pose inner iterations: expected 12, got {n_inner_2}"
    assert abs(rms_1 - 7.113145e-17) < 1e-15, f"one-pose final rms: expected ~7.11e-17, got {rms_1}"
    assert abs(rms_2 - 7.407067e-17) < 1e-15, f"two-pose final rms: expected ~7.41e-17, got {rms_2}"
    print("\n*** All assertions passed ***")


if __name__ == "__main__":
    main()
