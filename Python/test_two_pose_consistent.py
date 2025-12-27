#!/usr/bin/env python3
"""
Compare Python AmbientConsistent two-pose jacobians with C++ output.
"""

import numpy as np
from jacobians_ambient import AmbientConsistent, IncidenceParams

# Test inputs from C++ PythonTwoPoseInputs
pA = np.array([0.1, 0.2, 0.3])
qB = np.array([0.12, 0.18, 0.35])
nB = np.array([0.0, 0.0, 1.0])
dA0 = np.array([0.107832773203438, 0.215665546406877, -0.970494958830946])

# Pose A: quat_exp_so3([0.3, 0.2, 0.1]) in xyzw format
xA = np.array([0.149126529974578, 0.099417686649719, 0.049708843324859,
               0.982550982155259, 0.01, -0.01, 0.02])

# Pose B: quat_exp_so3([0.1, -0.2, 0.3]) in xyzw format
xB = np.array([0.049708843324859, -0.099417686649719, 0.149126529974578,
               0.982550982155259, 0.02, 0.00, -0.03])

# No weighting for comparison
params = IncidenceParams(enable_weight=False, enable_gate=False, tau=0.2, mode="sqrtabs")

print("=== Two-Pose Test Inputs ===")
print(f"pA = {pA}")
print(f"qB = {qB}")
print(f"nB = {nB}")
print(f"dA0 = {dA0}")
print(f"xA = {xA}")
print(f"xB = {xB}")

print("\n=== ForwardRayCostTwoPose Consistent (Python) ===")
ok, r, J7A, J7B = AmbientConsistent.residual_and_jac_fwd_two_pose(xA, xB, pA, qB, nB, dA0, params)
print(f"residual = {r:.15e}")
print(f"J7A = [{J7A[0]:.9f}, {J7A[1]:.9f}, {J7A[2]:.9f}, {J7A[3]:.9f}, {J7A[4]:.9f}, {J7A[5]:.9f}, {J7A[6]:.9f}]")
print(f"  J7A[0:4] (dqA) = [{J7A[0]:.9f}, {J7A[1]:.9f}, {J7A[2]:.9f}, {J7A[3]:.9f}]")
print(f"  J7A[4:7] (dtA) = [{J7A[4]:.9f}, {J7A[5]:.9f}, {J7A[6]:.9f}]")
print(f"J7B = [{J7B[0]:.9f}, {J7B[1]:.9f}, {J7B[2]:.9f}, {J7B[3]:.9f}, {J7B[4]:.9f}, {J7B[5]:.9f}, {J7B[6]:.9f}]")
print(f"  J7B[0:4] (dqB) = [{J7B[0]:.9f}, {J7B[1]:.9f}, {J7B[2]:.9f}, {J7B[3]:.9f}]")
print(f"  J7B[4:7] (dtB) = [{J7B[4]:.9f}, {J7B[5]:.9f}, {J7B[6]:.9f}]")

print("\n=== C++ ForwardRayCostTwoPose Consistent ===")
cpp_r = 2.766120755890689e-02
cpp_J7A = np.array([-0.308344248, 0.476777435, -0.105206617, -0.0459194179, 0.204385713, 0.144152857, -1.10410613])
cpp_J7B = np.array([0.251746584, -0.522317078, 0.00544468368, 0.0421480613, -0.204385713, -0.144152857, 1.10410613])
print(f"residual = {cpp_r:.15e}")
print(f"J7A = [{cpp_J7A[0]:.9f}, {cpp_J7A[1]:.9f}, {cpp_J7A[2]:.9f}, {cpp_J7A[3]:.9f}, {cpp_J7A[4]:.9f}, {cpp_J7A[5]:.9f}, {cpp_J7A[6]:.9f}]")
print(f"J7B = [{cpp_J7B[0]:.9f}, {cpp_J7B[1]:.9f}, {cpp_J7B[2]:.9f}, {cpp_J7B[3]:.9f}, {cpp_J7B[4]:.9f}, {cpp_J7B[5]:.9f}, {cpp_J7B[6]:.9f}]")

print("\n=== Comparison ===")
print(f"residual diff: {abs(r - cpp_r):.2e}")
print(f"J7A max diff: {np.max(np.abs(J7A - cpp_J7A)):.2e}")
print(f"J7B max diff: {np.max(np.abs(J7B - cpp_J7B)):.2e}")

if abs(r - cpp_r) < 1e-12 and np.max(np.abs(J7A - cpp_J7A)) < 1e-8 and np.max(np.abs(J7B - cpp_J7B)) < 1e-8:
    print("\n*** Forward Consistent: Python and C++ MATCH ***")
else:
    print("\n*** Forward Consistent: MISMATCH ***")

# Test reverse direction - use same geometry but swapped roles
# For reverse: ray from B hits surface in A
pB = pA  # Point in frame B
qA_pt = qB  # Surface point in A
nA = nB  # Surface normal in A
dB0 = dA0  # Ray direction in frame B

print("\n" + "="*60)
print("\n=== ReverseRayCostTwoPose Consistent (Python) ===")
ok, r_rev, J7A_rev, J7B_rev = AmbientConsistent.residual_and_jac_rev_two_pose(xA, xB, pB, qA_pt, nA, dB0, params)
print(f"residual = {r_rev:.15e}")
print(f"J7A = [{J7A_rev[0]:.9f}, {J7A_rev[1]:.9f}, {J7A_rev[2]:.9f}, {J7A_rev[3]:.9f}, {J7A_rev[4]:.9f}, {J7A_rev[5]:.9f}, {J7A_rev[6]:.9f}]")
print(f"  J7A[0:4] (dqA) = [{J7A_rev[0]:.9f}, {J7A_rev[1]:.9f}, {J7A_rev[2]:.9f}, {J7A_rev[3]:.9f}]")
print(f"  J7A[4:7] (dtA) = [{J7A_rev[4]:.9f}, {J7A_rev[5]:.9f}, {J7A_rev[6]:.9f}]")
print(f"J7B = [{J7B_rev[0]:.9f}, {J7B_rev[1]:.9f}, {J7B_rev[2]:.9f}, {J7B_rev[3]:.9f}, {J7B_rev[4]:.9f}, {J7B_rev[5]:.9f}, {J7B_rev[6]:.9f}]")
print(f"  J7B[0:4] (dqB) = [{J7B_rev[0]:.9f}, {J7B_rev[1]:.9f}, {J7B_rev[2]:.9f}, {J7B_rev[3]:.9f}]")
print(f"  J7B[4:7] (dtB) = [{J7B_rev[4]:.9f}, {J7B_rev[5]:.9f}, {J7B_rev[6]:.9f}]")
