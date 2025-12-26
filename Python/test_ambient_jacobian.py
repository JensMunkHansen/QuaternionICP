#!/usr/bin/env python3
"""
Test for ambient jacobians - prints residual and jacobian for comparison with C++.
Tests both simplified (Ambient) and consistent (AmbientConsistent) jacobians.

Run all tests: pytest test_ambient_jacobian.py -v -s
Run single test: pytest test_ambient_jacobian.py::test_simplified -v -s
"""

import numpy as np
import pytest
from jacobians_ambient import IncidenceParams, Ambient, AmbientConsistent, incidence_weight
from se3_utils import quat_exp_so3 as quat_exp_so3_xyzw


@pytest.fixture
def test_inputs():
    """Common test inputs for all jacobian tests."""
    pS = np.array([0.1, 0.2, 0.3])
    qT = np.array([0.12, 0.18, 0.35])
    nT = np.array([0.0, 0.0, 1.0])
    dS0 = np.array([0.0, 0.0, -1.0])

    q = quat_exp_so3_xyzw(np.array([0.01, 0.02, 0.03]))
    t = np.array([0.01, -0.01, 0.02])
    x = np.hstack([q, t])

    params = IncidenceParams()

    return pS, qT, nT, dS0, x, params


def test_simplified(test_inputs):
    """Test simplified jacobian (Ambient class)."""
    pS, qT, nT, dS0, x, params = test_inputs

    print("\n=== Test Inputs ===")
    print(f"pS = {pS}")
    print(f"qT = {qT}")
    print(f"nT = {nT}")
    print(f"dS0 = {dS0}")
    print(f"x (pose) = {x}")
    print(f"params = {params}")

    ok, r, J7 = Ambient.residual_and_jac_fwd(x, pS, qT, nT, dS0, params)

    print("\n=== SIMPLIFIED jacobian (Ambient class) ===")
    print(f"residual = {r:.15e}")
    print(f"J7 = {J7}")
    print(f"  J7[0:4] (dq) = {J7[:4]}")
    print(f"  J7[4:7] (dt) = {J7[4:]}")

    assert ok


def test_consistent(test_inputs):
    """Test consistent jacobian (AmbientConsistent class)."""
    pS, qT, nT, dS0, x, params = test_inputs

    print("\n=== Test Inputs ===")
    print(f"pS = {pS}")
    print(f"qT = {qT}")
    print(f"nT = {nT}")
    print(f"dS0 = {dS0}")
    print(f"x (pose) = {x}")
    print(f"params = {params}")

    r_cons, J7_cons, b = AmbientConsistent.residual_and_jac_fwd(x, pS, qT, nT, dS0)
    w = incidence_weight(b, params)

    print("\n=== CONSISTENT jacobian (AmbientConsistent class) ===")
    print(f"residual (weighted) = {w * r_cons:.15e}")
    print(f"J7 (weighted) = {w * J7_cons}")
    print(f"  J7[0:4] (dq) = {w * J7_cons[:4]}")
    print(f"  J7[4:7] (dt) = {w * J7_cons[4:]}")

    assert w > 0
