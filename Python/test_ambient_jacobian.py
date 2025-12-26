#!/usr/bin/env python3
"""
Test for simplified ambient jacobian - prints residual and jacobian for comparison with C++.
Uses the Ambient class (simplified jacobians: dr/dq = da_dq / b, ignoring db_dq term).
"""

import numpy as np
from jacobians_ambient import IncidenceParams, Ambient
from se3_utils import quat_exp_so3 as quat_exp_so3_xyzw


def main():
    np.set_printoptions(precision=15, suppress=False)

    # Test inputs
    pS = np.array([0.1, 0.2, 0.3])       # source point
    qT = np.array([0.12, 0.18, 0.35])    # target surface point
    nT = np.array([0.0, 0.0, 1.0])       # target normal (pointing up)
    dS0 = np.array([0.0, 0.0, -1.0])     # ray direction (pointing down)

    # Pose: small rotation + translation
    q = quat_exp_so3_xyzw(np.array([0.01, 0.02, 0.03]))
    t = np.array([0.01, -0.01, 0.02])
    x = np.hstack([q, t])

    # Default incidence params
    params = IncidenceParams()

    print("=== Test Inputs ===")
    print(f"pS = {pS}")
    print(f"qT = {qT}")
    print(f"nT = {nT}")
    print(f"dS0 = {dS0}")
    print(f"x (pose) = {x}")
    print(f"  q = {x[:4]}")
    print(f"  t = {x[4:]}")
    print(f"params = {params}")

    # Compute simplified jacobian using Ambient class
    ok, r, J7 = Ambient.residual_and_jac_fwd(x, pS, qT, nT, dS0, params)

    print("\n=== Outputs (SIMPLIFIED jacobian via Ambient class) ===")
    print(f"ok = {ok}")
    print(f"residual = {r:.15e}")
    print(f"J7 = {J7}")
    print(f"  J7[0:4] (dq) = {J7[:4]}")
    print(f"  J7[4:7] (dt) = {J7[4:]}")


if __name__ == "__main__":
    main()
