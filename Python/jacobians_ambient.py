"""
Ambient (7D) Jacobians and residuals for ray-projection ICP.

Two static classes:
  - AmbientConsistent: fully consistent quotient-rule Jacobians
  - Ambient: simplified Jacobians (treating denominator as constant for dq)

Residual: r = a/b where
  - a = n^T (x_transformed - q_surface)  (signed distance along normal)
  - b = n^T d                            (ray denominator)

SE(3) pose: 7D vector [qx, qy, qz, qw, tx, ty, tz]
"""

from dataclasses import dataclass
import numpy as np
from se3_utils import (
    quat_normalize as quat_normalize_xyzw,
    quat_to_R as quat_to_R_xyzw,
    dR_times_v_dq as dR_times_v_dq_xyzw,
    dRT_times_v_dq as dRT_times_v_dq_xyzw,
)


@dataclass
class IncidenceParams:
    """Parameters for incidence weighting / grazing-angle handling."""
    enable_weight: bool = True
    enable_gate: bool = True
    tau: float = 0.2        # 0.1-0.4 typical
    mode: str = "sqrtabs"   # "abs" or "sqrtabs"


def incidence_weight(c: float, params: IncidenceParams) -> float:
    """Compute incidence weight from denominator c = n^T d."""
    ac = abs(float(c))
    if params.enable_gate and ac < params.tau:
        return 0.0
    if not params.enable_weight:
        return 1.0
    ac = max(params.tau, min(1.0, ac))
    if params.mode == "abs":
        return ac
    if params.mode == "sqrtabs":
        return ac ** 0.5
    raise ValueError(params.mode)


class AmbientConsistent:
    """
    Fully consistent quotient-rule Jacobians.
    dr = (b*da - a*db) / b^2
    """

    @staticmethod
    def residual_and_jac_fwd(x, pS, qT, nT, dS0):
        """
        One-pose forward residual and Jacobian.

        Args:
            x: 7D pose [qx,qy,qz,qw, tx,ty,tz]
            pS: source point (3,)
            qT: target surface point (3,)
            nT: target surface normal (3,)
            dS0: ray direction in source frame (3,)

        Returns:
            r: scalar residual
            J7: 1x7 ambient Jacobian
            b: denominator (n^T d), for incidence weighting
        """
        q = quat_normalize_xyzw(x[:4])
        t = x[4:]
        R = quat_to_R_xyzw(q)

        xT = R @ pS + t
        d = R @ dS0

        a = float(nT @ (xT - qT))
        b = float(nT @ d)

        # dr/dt: db/dt = 0 since d = R @ dS0 doesn't depend on t
        dr_dt = nT / b

        # dr/dq: full quotient rule
        da_dq = nT @ dR_times_v_dq_xyzw(q, pS)
        db_dq = nT @ dR_times_v_dq_xyzw(q, dS0)
        dr_dq = (da_dq * b - a * db_dq) / (b * b)

        J7 = np.hstack([dr_dq, dr_dt])
        return a / b, J7, b

    @staticmethod
    def residual_and_jac_rev(x, pT, qS, nS, dT0):
        """
        One-pose reverse residual and Jacobian.

        Args:
            x: 7D pose [qx,qy,qz,qw, tx,ty,tz]
            pT: target point (3,)
            qS: source surface point (3,)
            nS: source surface normal (3,)
            dT0: ray direction in target frame (3,)

        Returns:
            r: scalar residual
            J7: 1x7 ambient Jacobian
            b: denominator (n^T d), for incidence weighting
        """
        q = quat_normalize_xyzw(x[:4])
        t = x[4:]
        R = quat_to_R_xyzw(q)

        u = pT - t
        yS = R.T @ u
        d = R.T @ dT0

        a = float(nS @ (yS - qS))
        b = float(nS @ d)

        # dr/dt
        dr_dt = (nS @ (-R.T)) / b

        # dr/dq: full quotient rule
        da_dq = nS @ dRT_times_v_dq_xyzw(q, u)
        db_dq = nS @ dRT_times_v_dq_xyzw(q, dT0)
        dr_dq = (da_dq * b - a * db_dq) / (b * b)

        J7 = np.hstack([dr_dq, dr_dt])
        return a / b, J7, b

    @staticmethod
    def residual_and_jac_fwd_two_pose(xA, xB, pA, qB_pt, nB, dA0):
        """
        Two-pose forward residual and Jacobians (A -> B).

        Args:
            xA: 7D pose A
            xB: 7D pose B
            pA: point in frame A (3,)
            qB_pt: target surface point in B (3,)
            nB: target surface normal in B (3,)
            dA0: ray direction in frame A (3,)

        Returns:
            r: scalar residual
            J7A: 1x7 ambient Jacobian w.r.t. xA
            J7B: 1x7 ambient Jacobian w.r.t. xB
            b: denominator (n^T d), for incidence weighting
        """
        qA = quat_normalize_xyzw(xA[:4]); tA = xA[4:]; RA = quat_to_R_xyzw(qA)
        qB = quat_normalize_xyzw(xB[:4]); tB = xB[4:]; RB = quat_to_R_xyzw(qB)

        U = RA @ pA + (tA - tB)     # in world
        xBv = RB.T @ U              # in B
        u_d = RA @ dA0              # in world
        d = RB.T @ u_d              # in B

        a = float(nB @ (xBv - qB_pt))
        b = float(nB @ d)

        # A block
        da_dtA = nB @ RB.T
        db_dtA = np.zeros(3)

        dx_dqA = RB.T @ dR_times_v_dq_xyzw(qA, pA)
        da_dqA = nB @ dx_dqA

        dd_dqA = RB.T @ dR_times_v_dq_xyzw(qA, dA0)
        db_dqA = nB @ dd_dqA

        dr_dqA = (da_dqA * b - a * db_dqA) / (b * b)
        dr_dtA = (da_dtA * b - a * db_dtA) / (b * b)
        J7A = np.hstack([dr_dqA, dr_dtA])

        # B block
        da_dtB = nB @ (-RB.T)
        db_dtB = np.zeros(3)

        dx_dqB = dRT_times_v_dq_xyzw(qB, U)
        da_dqB = nB @ dx_dqB

        dd_dqB = dRT_times_v_dq_xyzw(qB, u_d)
        db_dqB = nB @ dd_dqB

        dr_dqB = (da_dqB * b - a * db_dqB) / (b * b)
        dr_dtB = (da_dtB * b - a * db_dtB) / (b * b)
        J7B = np.hstack([dr_dqB, dr_dtB])

        return a / b, J7A, J7B, b

    @staticmethod
    def residual_and_jac_rev_two_pose(xA, xB, pB, qA_pt, nA, dB0):
        """
        Two-pose reverse residual and Jacobians (B -> A).

        Args:
            xA: 7D pose A
            xB: 7D pose B
            pB: point in frame B (3,)
            qA_pt: target surface point in A (3,)
            nA: target surface normal in A (3,)
            dB0: ray direction in frame B (3,)

        Returns:
            r: scalar residual
            J7A: 1x7 ambient Jacobian w.r.t. xA
            J7B: 1x7 ambient Jacobian w.r.t. xB
            b: denominator (n^T d), for incidence weighting
        """
        qA = quat_normalize_xyzw(xA[:4]); tA = xA[4:]; RA = quat_to_R_xyzw(qA)
        qB = quat_normalize_xyzw(xB[:4]); tB = xB[4:]; RB = quat_to_R_xyzw(qB)

        V = RB @ pB + (tB - tA)     # in world
        xAv = RA.T @ V              # in A
        v_d = RB @ dB0              # in world
        d = RA.T @ v_d              # in A

        a = float(nA @ (xAv - qA_pt))
        b = float(nA @ d)

        # B block
        da_dtB = nA @ RA.T
        db_dtB = np.zeros(3)

        dx_dqB = RA.T @ dR_times_v_dq_xyzw(qB, pB)
        da_dqB = nA @ dx_dqB

        dd_dqB = RA.T @ dR_times_v_dq_xyzw(qB, dB0)
        db_dqB = nA @ dd_dqB

        dr_dqB = (da_dqB * b - a * db_dqB) / (b * b)
        dr_dtB = (da_dtB * b - a * db_dtB) / (b * b)
        J7B = np.hstack([dr_dqB, dr_dtB])

        # A block
        da_dtA = nA @ (-RA.T)
        db_dtA = np.zeros(3)

        dx_dqA = dRT_times_v_dq_xyzw(qA, V)
        da_dqA = nA @ dx_dqA

        dd_dqA = dRT_times_v_dq_xyzw(qA, v_d)
        db_dqA = nA @ dd_dqA

        dr_dqA = (da_dqA * b - a * db_dqA) / (b * b)
        dr_dtA = (da_dtA * b - a * db_dtA) / (b * b)
        J7A = np.hstack([dr_dqA, dr_dtA])

        return a / b, J7A, J7B, b


class Ambient:
    """
    Simplified ambient Jacobians (treating denominator as constant for dq).
    dr/dq ≈ da_dq / b  (ignores db_dq term)

    Methods return (True, r, w*J7) where w is the incidence weight.
    If w=0 (grazing angle), J7 will be zeros.
    """

    @staticmethod
    def residual_and_jac_fwd(x, pS, qT, nT, dS0, params: IncidenceParams):
        """
        One-pose forward residual and Jacobian (simplified).

        Args:
            x: 7D pose [qx,qy,qz,qw, tx,ty,tz]
            pS: source point (3,)
            qT: target surface point (3,)
            nT: target surface normal (3,)
            dS0: ray direction in source frame (3,)
            params: incidence weighting parameters

        Returns:
            (True, r, w*J7) where w is incidence weight
        """
        q = quat_normalize_xyzw(x[:4])
        t = x[4:]
        R = quat_to_R_xyzw(q)

        xT = R @ pS + t
        d = R @ dS0

        a = float(nT @ (xT - qT))
        b = float(nT @ d)
        r = a / b

        w = incidence_weight(b, params)
        if w == 0.0:
            return True, r, np.zeros(7)

        # dr/dt
        dr_dt = nT / b

        # dr/dq: simplified (ignore db_dq term)
        da_dq = nT @ dR_times_v_dq_xyzw(q, pS)
        dr_dq = da_dq / b

        J7 = np.hstack([dr_dq, dr_dt])
        return True, w * r, w * J7

    @staticmethod
    def residual_and_jac_rev(x, pT, qS, nS, dT0, params: IncidenceParams):
        """
        One-pose reverse residual and Jacobian (simplified).

        Args:
            x: 7D pose [qx,qy,qz,qw, tx,ty,tz]
            pT: target point (3,)
            qS: source surface point (3,)
            nS: source surface normal (3,)
            dT0: ray direction in target frame (3,)
            params: incidence weighting parameters

        Returns:
            (True, r, w*J7) where w is incidence weight
        """
        q = quat_normalize_xyzw(x[:4])
        t = x[4:]
        R = quat_to_R_xyzw(q)

        u = pT - t
        yS = R.T @ u
        d = R.T @ dT0

        a = float(nS @ (yS - qS))
        b = float(nS @ d)
        r = a / b

        w = incidence_weight(b, params)
        if w == 0.0:
            return True, r, np.zeros(7)

        # dr/dt
        dr_dt = (nS @ (-R.T)) / b

        # dr/dq: simplified (ignore db_dq term)
        da_dq = nS @ dRT_times_v_dq_xyzw(q, u)
        dr_dq = da_dq / b

        J7 = np.hstack([dr_dq, dr_dt])
        return True, w * r, w * J7

    @staticmethod
    def residual_and_jac_fwd_two_pose(xA, xB, pA, qB_pt, nB, dA0, params: IncidenceParams):
        """
        Two-pose forward residual and Jacobians (A -> B).

        Args:
            xA: 7D pose A
            xB: 7D pose B
            pA: point in frame A (3,)
            qB_pt: target surface point in B (3,)
            nB: target surface normal in B (3,)
            dA0: ray direction in frame A (3,)
            params: incidence weighting parameters

        Returns:
            (True, r, w*J7A, w*J7B) where w is incidence weight
        """
        qA = quat_normalize_xyzw(xA[:4]); tA = xA[4:]; RA = quat_to_R_xyzw(qA)
        qB = quat_normalize_xyzw(xB[:4]); tB = xB[4:]; RB = quat_to_R_xyzw(qB)

        U = RA @ pA + (tA - tB)     # in world
        xBv = RB.T @ U              # in B
        u_d = RA @ dA0              # in world
        d = RB.T @ u_d              # in B

        a = float(nB @ (xBv - qB_pt))
        b = float(nB @ d)
        r = a / b

        w = incidence_weight(b, params)
        if w == 0.0:
            return True, r, np.zeros(7), np.zeros(7)

        # A block (simplified)
        da_dtA = nB @ RB.T
        dr_dtA = da_dtA / b

        dx_dqA = RB.T @ dR_times_v_dq_xyzw(qA, pA)
        da_dqA = nB @ dx_dqA
        dr_dqA = da_dqA / b

        J7A = np.hstack([dr_dqA, dr_dtA])

        # B block (simplified)
        da_dtB = nB @ (-RB.T)
        dr_dtB = da_dtB / b

        dx_dqB = dRT_times_v_dq_xyzw(qB, U)
        da_dqB = nB @ dx_dqB
        dr_dqB = da_dqB / b

        J7B = np.hstack([dr_dqB, dr_dtB])

        return True, w * r, w * J7A, w * J7B

    @staticmethod
    def residual_and_jac_rev_two_pose(xA, xB, pB, qA_pt, nA, dB0, params: IncidenceParams):
        """
        Two-pose reverse residual and Jacobians (B -> A), simplified.

        Args:
            xA: 7D pose A
            xB: 7D pose B
            pB: point in frame B (3,)
            qA_pt: target surface point in A (3,)
            nA: target surface normal in A (3,)
            dB0: ray direction in frame B (3,)
            params: incidence weighting parameters

        Returns:
            (True, r, w*J7A, w*J7B) where w is incidence weight
        """
        qA = quat_normalize_xyzw(xA[:4]); tA = xA[4:]; RA = quat_to_R_xyzw(qA)
        qB = quat_normalize_xyzw(xB[:4]); tB = xB[4:]; RB = quat_to_R_xyzw(qB)

        V = RB @ pB + (tB - tA)     # in world
        xAv = RA.T @ V              # in A
        v_d = RB @ dB0              # in world
        d = RA.T @ v_d              # in A

        a = float(nA @ (xAv - qA_pt))
        b = float(nA @ d)
        r = a / b

        w = incidence_weight(b, params)
        if w == 0.0:
            return True, r, np.zeros(7), np.zeros(7)

        # B block (simplified)
        da_dtB = nA @ RA.T
        dr_dtB = da_dtB / b

        dx_dqB = RA.T @ dR_times_v_dq_xyzw(qB, pB)
        da_dqB = nA @ dx_dqB
        dr_dqB = da_dqB / b

        J7B = np.hstack([dr_dqB, dr_dtB])

        # A block (simplified)
        da_dtA = nA @ (-RA.T)
        dr_dtA = da_dtA / b

        dx_dqA = dRT_times_v_dq_xyzw(qA, V)
        da_dqA = nA @ dx_dqA
        dr_dqA = da_dqA / b

        J7A = np.hstack([dr_dqA, dr_dtA])

        return True, w * r, w * J7A, w * J7B
