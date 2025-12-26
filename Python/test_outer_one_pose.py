#!/usr/bin/env python3
"""
Test outer loop solver (ICP with correspondence updates).
This should produce the same residuals as the C++ implementation.

Run all tests: pytest test_outer_one_pose.py -v -s
Run single test: pytest test_outer_one_pose.py::test_small_perturbation -v -s
"""

import numpy as np
import pytest

from test_grids import make_heightfield_mesh
from se3_utils import quat_exp_so3 as quat_exp_so3_xyzw
from jacobians_ambient import IncidenceParams
from icp_utils import icp_one_pose


@pytest.fixture
def test_grids():
    """Common test inputs: meshes and sample points."""
    # Create grids - source and target are the same (like C++ test)
    meshS, VS = make_heightfield_mesh(z0=0.0)
    meshT, VT = make_heightfield_mesh(z0=0.0)  # Same as source

    # Use all vertices as sample points
    P_S = VS.copy()
    P_T = VT.copy()

    # No weighting for comparison with C++
    params = IncidenceParams(
        enable_weight=False,
        enable_gate=False,
        tau=0.2,
        mode="sqrtabs"
    )

    return meshS, meshT, P_S, P_T, params


def test_small_perturbation(test_grids):
    """Test 1: Small perturbation - should converge to identity."""
    meshS, meshT, P_S, P_T, params = test_grids
    np.set_printoptions(precision=6, suppress=True)

    print("\n=== Test 1: Small perturbation ===")
    axis_angle = np.array([0.01, 0.005, -0.008])
    q0 = quat_exp_so3_xyzw(axis_angle)
    t0 = np.array([0.01, -0.005, 0.02])
    x0 = np.hstack([q0, t0])

    print(f"Initial pose: q={q0}, t={t0}")
    x_est, total, rms = icp_one_pose(meshS, meshT, P_S, P_T, x0, params, max_outer_iterations=6, inner=12, verbose=True)
    print(f"Final pose: {x_est}")

    # Should converge close to identity
    assert np.linalg.norm(x_est[:3]) < 1e-6, "Rotation should be near zero"
    assert np.abs(x_est[3] - 1.0) < 1e-6, "Quaternion w should be near 1"
    assert np.linalg.norm(x_est[4:]) < 1e-6, "Translation should be near zero"


def test_larger_perturbation(test_grids):
    """Test 2: Larger perturbation - should still converge to identity."""
    meshS, meshT, P_S, P_T, params = test_grids
    np.set_printoptions(precision=6, suppress=True)

    print("\n=== Test 2: Larger perturbation ===")
    axis_angle = np.array([0.05, 0.02, -0.03])
    q0 = quat_exp_so3_xyzw(axis_angle)
    t0 = np.array([0.05, 0.02, 0.08])
    x0 = np.hstack([q0, t0])

    print(f"Initial pose: q={q0}, t={t0}")
    x_est, total, rms = icp_one_pose(meshS, meshT, P_S, P_T, x0, params, max_outer_iterations=6, inner=12, verbose=True)
    print(f"Final pose: {x_est}")

    # Should converge close to identity
    assert np.linalg.norm(x_est[:3]) < 1e-6, "Rotation should be near zero"
    assert np.abs(x_est[3] - 1.0) < 1e-6, "Quaternion w should be near 1"
    assert np.linalg.norm(x_est[4:]) < 1e-6, "Translation should be near zero"


def test_translation_only(test_grids):
    """Test 3: Translation only - should converge to identity."""
    meshS, meshT, P_S, P_T, params = test_grids
    np.set_printoptions(precision=6, suppress=True)

    print("\n=== Test 3: Translation only ===")
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    t0 = np.array([0.03, -0.02, 0.05])
    x0 = np.hstack([q0, t0])

    print(f"Initial pose: q={q0}, t={t0}")
    x_est, total, rms = icp_one_pose(meshS, meshT, P_S, P_T, x0, params, max_outer_iterations=6, inner=12, verbose=True)
    print(f"Final pose: {x_est}")

    # Should converge close to identity
    assert np.linalg.norm(x_est[:3]) < 1e-6, "Rotation should be near zero"
    assert np.abs(x_est[3] - 1.0) < 1e-6, "Quaternion w should be near 1"
    assert np.linalg.norm(x_est[4:]) < 1e-6, "Translation should be near zero"
