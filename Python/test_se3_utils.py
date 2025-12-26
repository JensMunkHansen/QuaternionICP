#!/usr/bin/env python3
"""
Test script for se3_utils - outputs values for C++ comparison.
Run this to generate reference values, then compare with C++ test output.

Usage:
  python test_se3_utils.py          # Run all tests with full output
  python test_se3_utils.py --brief  # Run brief output matching C++ WARN format
"""

import sys
import numpy as np
from se3_utils import (
    skew, quat_normalize, quat_mul, quat_to_R, quat_exp_so3,
    V_matrix_so3, se3_plus, plus_jacobian_7x6,
    dR_dq_mats, dR_times_v_dq, dRT_times_v_dq
)

np.set_printoptions(precision=6, suppress=True, linewidth=200)


def test_skew():
    print("=== skew ===")
    w = np.array([1.0, 2.0, 3.0])
    S = skew(w)
    print(f"w = {w}")
    print(f"skew(w) =\n{S}")
    print()


def test_quat_normalize():
    print("=== quat_normalize ===")
    q = np.array([1.0, 2.0, 3.0, 4.0])
    qn = quat_normalize(q)
    print(f"q = {q}")
    print(f"quat_normalize(q) = {qn}")
    print(f"norm = {np.linalg.norm(qn)}")
    print()


def test_quat_mul():
    print("=== quat_mul ===")
    q1 = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
    q2 = quat_normalize(np.array([0.5, -1.0, 0.3, 2.0]))
    qp = quat_mul(q1, q2)
    print(f"q1 = {q1}")
    print(f"q2 = {q2}")
    print(f"q1 * q2 = {qp}")
    print()


def test_quat_to_R():
    print("=== quat_to_R ===")
    q = quat_normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    R = quat_to_R(q)
    print(f"q = {q}")
    print(f"R =\n{R}")
    print(f"det(R) = {np.linalg.det(R)}")
    print()


def test_quat_exp_so3():
    print("=== quat_exp_so3 ===")
    # Test small angle
    w_small = np.array([0.001, 0.002, -0.001])
    q_small = quat_exp_so3(w_small)
    print(f"w_small = {w_small}")
    print(f"quat_exp_so3(w_small) = {q_small}")

    # Test larger angle
    w = np.array([0.1, 0.2, -0.15])
    q = quat_exp_so3(w)
    print(f"w = {w}")
    print(f"quat_exp_so3(w) = {q}")
    print()


def test_V_matrix_so3():
    print("=== V_matrix_so3 ===")
    # Test small angle
    w_small = np.array([0.001, 0.002, -0.001])
    V_small = V_matrix_so3(w_small)
    print(f"w_small = {w_small}")
    print(f"V_matrix_so3(w_small) =\n{V_small}")

    # Test larger angle
    w = np.array([0.1, 0.2, -0.15])
    V = V_matrix_so3(w)
    print(f"w = {w}")
    print(f"V_matrix_so3(w) =\n{V}")
    print()


def test_se3_plus():
    print("=== se3_plus ===")
    q0 = quat_exp_so3(np.array([0.02, 0.01, -0.03]))
    t0 = np.array([-0.02, 0.01, 0.05])
    x = np.hstack([q0, t0])
    print(f"x = {x}")

    delta = np.array([0.01, -0.02, 0.005, 0.03, -0.01, 0.02])
    x_plus = se3_plus(x, delta)
    print(f"delta = {delta}")
    print(f"se3_plus(x, delta) = {x_plus}")
    print()


def test_plus_jacobian_7x6():
    print("=== plus_jacobian_7x6 ===")
    q0 = quat_exp_so3(np.array([0.02, 0.01, -0.03]))
    t0 = np.array([-0.02, 0.01, 0.05])
    x = np.hstack([q0, t0])
    print(f"x = {x}")

    P = plus_jacobian_7x6(x)
    print(f"plus_jacobian_7x6(x) =\n{P}")
    print()


def test_dR_dq():
    print("=== dR_dq_mats ===")
    q = quat_normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    dRdx, dRdy, dRdz, dRdw = dR_dq_mats(q)
    print(f"q = {q}")
    print(f"dR/dx =\n{dRdx}")
    print(f"dR/dy =\n{dRdy}")
    print(f"dR/dz =\n{dRdz}")
    print(f"dR/dw =\n{dRdw}")
    print()


def test_dR_times_v_dq():
    print("=== dR_times_v_dq ===")
    q = quat_normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    v = np.array([1.0, 2.0, 3.0])
    J = dR_times_v_dq(q, v)
    print(f"q = {q}")
    print(f"v = {v}")
    print(f"dR_times_v_dq(q, v) =\n{J}")
    print()


def test_dRT_times_v_dq():
    print("=== dRT_times_v_dq ===")
    q = quat_normalize(np.array([0.1, 0.2, 0.3, 0.9]))
    v = np.array([1.0, 2.0, 3.0])
    J = dRT_times_v_dq(q, v)
    print(f"q = {q}")
    print(f"v = {v}")
    print(f"dRT_times_v_dq(q, v) =\n{J}")
    print()


def brief_output():
    """Output matching C++ WARN format for easy comparison."""
    print("=== Python SE3 test output (for C++ comparison) ===\n")

    # skew
    w_skew = np.array([1.0, 2.0, 3.0])
    S = skew(w_skew)
    print("Python skew([1, 2, 3]):")
    print(f"  [{S[0,0]}, {S[0,1]}, {S[0,2]}]")
    print(f"  [{S[1,0]}, {S[1,1]}, {S[1,2]}]")
    print(f"  [{S[2,0]}, {S[2,1]}, {S[2,2]}]")

    # quatExpSO3 small angle
    w_small = np.array([0.001, 0.002, -0.001])
    q_small = quat_exp_so3(w_small)
    print("Python quatExpSO3([0.001, 0.002, -0.001]):")
    print(f"  [{q_small[0]}, {q_small[1]}, {q_small[2]}, {q_small[3]}]")

    # quatExpSO3 larger angle
    w = np.array([0.1, 0.2, -0.15])
    q = quat_exp_so3(w)
    print("Python quatExpSO3([0.1, 0.2, -0.15]):")
    print(f"  [{q[0]}, {q[1]}, {q[2]}, {q[3]}]")

    # quat_to_R
    q_unnorm = np.array([0.1, 0.2, 0.3, 0.9])
    q_norm = quat_normalize(q_unnorm)
    R = quat_to_R(q_norm)
    print("Python quat_to_R([0.1, 0.2, 0.3, 0.9] normalized):")
    print(f"  [{R[0,0]}, {R[0,1]}, {R[0,2]}]")
    print(f"  [{R[1,0]}, {R[1,1]}, {R[1,2]}]")
    print(f"  [{R[2,0]}, {R[2,1]}, {R[2,2]}]")

    # Vso3
    V = V_matrix_so3(w)
    print("Python Vso3([0.1, 0.2, -0.15]):")
    print(f"  [{V[0,0]}, {V[0,1]}, {V[0,2]}]")
    print(f"  [{V[1,0]}, {V[1,1]}, {V[1,2]}]")
    print(f"  [{V[2,0]}, {V[2,1]}, {V[2,2]}]")

    # se3_plus
    w0 = np.array([0.02, 0.01, -0.03])
    q0 = quat_exp_so3(w0)
    t0 = np.array([-0.02, 0.01, 0.05])
    x = np.hstack([q0, t0])
    print("Python initial pose x:")
    print(f"  [{x[0]}, {x[1]}, {x[2]}, {x[3]}, {x[4]}, {x[5]}, {x[6]}]")

    delta = np.array([0.01, -0.02, 0.005, 0.03, -0.01, 0.02])
    x_plus = se3_plus(x, delta)
    print("Python se3_plus(x, delta):")
    print(f"  [{x_plus[0]}, {x_plus[1]}, {x_plus[2]}, {x_plus[3]}, {x_plus[4]}, {x_plus[5]}, {x_plus[6]}]")

    # plusJacobian7x6
    P = plus_jacobian_7x6(x)
    print("Python plusJacobian7x6(x):")
    for i in range(7):
        print(f"  [{P[i,0]}, {P[i,1]}, {P[i,2]}, {P[i,3]}, {P[i,4]}, {P[i,5]}]")

    # dRv_dq
    v = np.array([1.0, 2.0, 3.0])
    J_Rv = dR_times_v_dq(q_norm, v)
    print("Python dRv_dq(q, v) where q=normalized([0.1,0.2,0.3,0.9]), v=[1,2,3]:")
    print(f"  [{J_Rv[0,0]}, {J_Rv[0,1]}, {J_Rv[0,2]}, {J_Rv[0,3]}]")
    print(f"  [{J_Rv[1,0]}, {J_Rv[1,1]}, {J_Rv[1,2]}, {J_Rv[1,3]}]")
    print(f"  [{J_Rv[2,0]}, {J_Rv[2,1]}, {J_Rv[2,2]}, {J_Rv[2,3]}]")

    # dRTv_dq
    J_RTv = dRT_times_v_dq(q_norm, v)
    print("Python dRTv_dq(q, v):")
    print(f"  [{J_RTv[0,0]}, {J_RTv[0,1]}, {J_RTv[0,2]}, {J_RTv[0,3]}]")
    print(f"  [{J_RTv[1,0]}, {J_RTv[1,1]}, {J_RTv[1,2]}, {J_RTv[1,3]}]")
    print(f"  [{J_RTv[2,0]}, {J_RTv[2,1]}, {J_RTv[2,2]}, {J_RTv[2,3]}]")


if __name__ == "__main__":
    if "--brief" in sys.argv:
        brief_output()
    else:
        test_skew()
        test_quat_normalize()
        test_quat_mul()
        test_quat_to_R()
        test_quat_exp_so3()
        test_V_matrix_so3()
        test_se3_plus()
        test_plus_jacobian_7x6()
        test_dR_dq()
        test_dR_times_v_dq()
        test_dRT_times_v_dq()
