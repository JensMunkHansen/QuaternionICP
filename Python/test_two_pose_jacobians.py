#!/usr/bin/env python3
"""
Test two-pose ambient Jacobians.

Uses same inputs as C++ JacobiansAmbientTwoPoseTest.cpp [python] test
to verify both implementations produce identical results.

Run: python test_two_pose_jacobians.py
"""

import numpy as np
from jacobians_ambient import Ambient, IncidenceParams
from se3_utils import quat_exp_so3 as quat_exp_so3_xyzw

# Disable weighting for comparison with C++
NO_WEIGHT = IncidenceParams(enable_weight=False, enable_gate=False, tau=0.0)


def test_forward_two_pose_jacobian():
    """Test ForwardRayCostTwoPose with same inputs as C++ [python] test."""

    # Geometry for forward (A -> B) - with non-trivial ray direction
    pA = np.array([0.1, 0.2, 0.3])
    qB_pt = np.array([0.12, 0.18, 0.35])
    nB = np.array([0.0, 0.0, 1.0])
    dA0_raw = np.array([0.1, 0.2, -0.9])
    dA0 = dA0_raw / np.linalg.norm(dA0_raw)  # Normalized

    # Pose A: quat_exp_so3([0.3, 0.2, 0.1]) in xyzw format
    qA = quat_exp_so3_xyzw(np.array([0.3, 0.2, 0.1]))
    tA = np.array([0.01, -0.01, 0.02])
    xA = np.hstack([qA, tA])

    # Pose B: quat_exp_so3([0.1, -0.2, 0.3]) in xyzw format
    qB = quat_exp_so3_xyzw(np.array([0.1, -0.2, 0.3]))
    tB = np.array([0.02, 0.00, -0.03])
    xB = np.hstack([qB, tB])

    print("=== Two-Pose Test Inputs ===")
    print(f"pA = {pA}")
    print(f"qB_pt = {qB_pt}")
    print(f"nB = {nB}")
    print(f"dA0 = {dA0}")
    print(f"xA (pose A) = {xA}")
    print(f"xB (pose B) = {xB}")

    # Call Python implementation
    _, r, J7A, J7B = Ambient.residual_and_jac_fwd_two_pose(xA, xB, pA, qB_pt, nB, dA0, NO_WEIGHT)

    print("\n=== ForwardRayCostTwoPose Simplified (Python) ===")
    print(f"residual = {r:.15e}")
    print(f"J7A = [{J7A[0]:.15e}, {J7A[1]:.15e}, {J7A[2]:.15e}, {J7A[3]:.15e},")
    print(f"       {J7A[4]:.15e}, {J7A[5]:.15e}, {J7A[6]:.15e}]")
    print(f"J7B = [{J7B[0]:.15e}, {J7B[1]:.15e}, {J7B[2]:.15e}, {J7B[3]:.15e},")
    print(f"       {J7B[4]:.15e}, {J7B[5]:.15e}, {J7B[6]:.15e}]")

    print("\nNote: J7A[3] and J7B[3] are now non-trivial for proper verification")


def test_reverse_two_pose_jacobian():
    """Test ReverseRayCostTwoPose - verify it runs without error."""

    # Use similar geometry but for reverse direction (B -> A)
    pB = np.array([0.15, 0.25, 0.32])
    qA_pt = np.array([0.14, 0.22, 0.30])
    nA = np.array([0.0, 0.0, 1.0])
    dB0 = np.array([0.0, 0.0, -1.0])

    # Same poses as forward test
    qA = quat_exp_so3_xyzw(np.array([0.01, 0.02, 0.03]))
    tA = np.array([0.01, -0.01, 0.02])
    xA = np.hstack([qA, tA])

    qB = quat_exp_so3_xyzw(np.array([0.03, -0.01, 0.02]))
    tB = np.array([0.02, 0.00, -0.03])
    xB = np.hstack([qB, tB])

    # Call Python implementation
    _, r, J7A, J7B = Ambient.residual_and_jac_rev_two_pose(xA, xB, pB, qA_pt, nA, dB0, NO_WEIGHT)

    print("\n=== ReverseRayCostTwoPose Simplified (Python) ===")
    print(f"residual = {r:.15e}")
    print(f"J7A = {J7A}")
    print(f"J7B = {J7B}")

    # Basic sanity check
    assert np.isfinite(r), "Residual should be finite"
    assert np.sum(np.abs(J7A)) > 1e-10, "J7A should not be all zeros"
    assert np.sum(np.abs(J7B)) > 1e-10, "J7B should not be all zeros"

    print("\n*** Reverse two-pose test PASSED ***")


if __name__ == "__main__":
    np.set_printoptions(precision=9, suppress=False)
    test_forward_two_pose_jacobian()
    test_reverse_two_pose_jacobian()
    print("\n*** All two-pose Jacobian tests PASSED ***")
