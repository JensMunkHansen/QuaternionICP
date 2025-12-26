#!/usr/bin/env python3
"""
Test script for se3_utils - outputs values for C++ comparison.
Run this to generate reference values, then compare with C++ test output.
"""

import numpy as np
from se3_utils import (
    skew, quat_normalize, quat_mul, quat_to_R, quat_exp_so3,
    V_matrix_so3, se3_plus, plus_jacobian_7x6,
    dR_dq_mats, dR_times_v_dq, dRT_times_v_dq
)

np.set_printoptions(precision=12, suppress=True, linewidth=200)


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


if __name__ == "__main__":
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
