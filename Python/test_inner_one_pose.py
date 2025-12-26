#!/usr/bin/env python3
"""
Test inner solver with one outer iteration (fixed correspondences).
This should produce the same residuals as the C++ implementation.
"""

import numpy as np

from test_grids import make_heightfield_mesh
from se3_utils import quat_to_R as quat_to_R_xyzw
from jacobians_ambient import IncidenceParams
from icp_utils import build_corr_forward, build_corr_reverse, solve_inner_one_pose


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

    x_out, n_iters, rms, valid = solve_inner_one_pose(
        x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0, params, max_iters=12, verbose=True)
    print(f"Result: iterations={n_iters}, rms={rms:.6e}, valid={valid}")
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

    x_out, n_iters, rms, valid = solve_inner_one_pose(
        x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0, params, max_iters=12, verbose=True)
    print(f"Result: iterations={n_iters}, rms={rms:.6e}, valid={valid}")
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

    x_out, n_iters, rms, valid = solve_inner_one_pose(
        x, P_S, P_T, corr_fwd, corr_rev, dS0, dT0, params, max_iters=12, verbose=True)
    print(f"Result: iterations={n_iters}, rms={rms:.6e}, valid={valid}")
    print(f"Final pose: {x_out}")
    print()


if __name__ == "__main__":
    main()
