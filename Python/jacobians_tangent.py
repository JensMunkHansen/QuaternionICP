"""
Tangent (6D) Jacobians and residuals for ray-projection ICP with left-perturbation.

Matching JacobiansTangent.h in C++ - uses left-perturbation se(3) parameterization.

Left perturbation: T_new = exp(delta^) * T  (space/fixed frame)

Residual: r = a/b where
  - a = n^T (x_transformed - q_surface)  (signed distance along normal)
  - b = n^T d                            (ray denominator)

SE(3) pose: (R, t) - rotation matrix and translation vector
Tangent space: 6D vector [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z] (translation, rotation)
"""

from dataclasses import dataclass
import numpy as np


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


class TangentSimplified:
    """
    Simplified left-perturbation Jacobians (treating denominator as constant).
    dr/dphi ≈ da_dphi / b  (ignores db_dphi term)

    Methods return (r, w*J6) where w is the incidence weight.
    If w=0 (grazing angle), J6 will be zeros.
    """

    @staticmethod
    def residual_and_jac_fwd(R, t, pS, qT, nT, dS0, params: IncidenceParams):
        """
        One-pose forward residual and Jacobian (simplified, left perturbation).

        Given a correspondence from source grid to target grid:
        - Source point pS (in source-local coords)
        - Target point qT (in target-local coords)
        - Target normal nT (in target-local coords)
        - Ray direction dS0 (in source-local coords)
        - Current transform T = (R, t) from source-local to target-local

        The transformed source point is: x = R*pS + t

        Residual (ray-projected point-to-plane distance):
          r = n^T (x - q) / (n^T d)

        Jacobian w.r.t. se(3) tangent [rho; phi] (left perturbation):
          J = [+n^T | (x × n)^T] / (n^T d)

        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3,)
            pS: Source point (3,)
            qT: Target surface point (3,)
            nT: Target surface normal (3,)
            dS0: Ray direction in source frame (3,)
            params: Incidence weighting parameters

        Returns:
            (w*r, w*J6) where w is incidence weight
        """
        x = R @ pS + t
        d = R @ dS0

        n = nT
        q = qT

        a = float(n @ (x - q))
        b = float(n @ d)
        r = a / b

        w = incidence_weight(b, params)
        if w == 0.0:
            return r, np.zeros(6)

        # Jacobian: w * [+n^T | (x × n)^T] / b
        J6 = np.zeros(6)
        J6[:3] = (w * n) / b           # dr/d(rho)
        J6[3:] = (w * np.cross(x, n)) / b   # dr/d(phi)

        return w * r, J6

    @staticmethod
    def residual_and_jac_rev(R, t, pT, qS, nS, dT0, params: IncidenceParams):
        """
        One-pose reverse residual and Jacobian (simplified, left perturbation).

        Given a correspondence from target grid back to source grid:
        - pT: ray origin in target-local coordinates
        - qS: hit point in source-local coordinates
        - nS: normal at hit in source-local coordinates
        - dT0: ray direction in target-local coordinates
        - Transform T = (R, t) from source-local to target-local

        The inverse-transformed point is: x' = R^T(pT - t)

        Residual (ray-projected point-to-plane distance in source frame):
          r = nS^T (x' - qS) / (nS^T d')

        Jacobian w.r.t. se(3) tangent [rho; phi] (left perturbation):
          J = [-nbar^T | (nbar × pT)^T] / (nS^T d')

        where nbar = R * nS (source normal rotated to target frame)

        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3,)
            pT: Target point / ray origin in target frame (3,)
            qS: Source surface hit point (3,)
            nS: Source surface normal (3,)
            dT0: Ray direction in target frame (3,)
            params: Incidence weighting parameters

        Returns:
            (w*r, w*J6) where w is incidence weight
        """
        q = pT  # Ray origin in target-local
        p = qS  # Hit point in source-local
        ns = nS  # Normal in source-local

        # Inverse-transform q to source frame
        xprime = R.T @ (q - t)

        # Ray direction in source frame
        dprime = R.T @ dT0

        # Rotated normal (source normal in target frame)
        nbar = R @ ns

        # Ray normalization factor (in source frame)
        b = float(ns @ dprime)

        # Residual in source frame
        a = float(ns @ (xprime - p))
        r = a / b

        w = incidence_weight(b, params)
        if w == 0.0:
            return r, np.zeros(6)

        # Simplified Jacobian: w * [-nbar^T | (nbar × q)^T] / b
        J6 = np.zeros(6)
        J6[:3] = (-w * nbar) / b           # dr/d(rho)
        J6[3:] = (w * np.cross(nbar, q)) / b   # dr/d(phi)

        return w * r, J6


class TangentConsistent:
    """
    Fully consistent quotient-rule Jacobians (left perturbation).
    dr = (b*da - a*db) / b^2

    Methods return (r, w*J6) where w is the incidence weight.
    If w=0 (grazing angle), J6 will be zeros.

    The consistent version includes the db/dphi term from the
    denominator b = n^T d where d = R @ d0 (forward) or d = R^T @ d0 (reverse).
    """

    @staticmethod
    def residual_and_jac_fwd(R, t, pS, qT, nT, dS0, params: IncidenceParams):
        """
        One-pose forward residual and Jacobian (consistent, left perturbation).

        Full quotient rule: dr/dphi = (b * da/dphi - a * db/dphi) / b^2

        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3,)
            pS: Source point (3,)
            qT: Target surface point (3,)
            nT: Target surface normal (3,)
            dS0: Ray direction in source frame (3,)
            params: Incidence weighting parameters

        Returns:
            (w*r, w*J6) where w is incidence weight
        """
        x = R @ pS + t
        d = R @ dS0

        n = nT
        q = qT

        a = float(n @ (x - q))
        b = float(n @ d)
        r = a / b

        w = incidence_weight(b, params)
        if w == 0.0:
            return r, np.zeros(6)

        # da/drho = n, da/dphi = (x × n)
        # db/drho = 0, db/dphi = (d × n)
        da_drho = n
        da_dphi = np.cross(x, n)
        db_dphi = np.cross(d, n)

        # Full quotient rule: dr = (b*da - a*db) / b^2
        J6 = np.zeros(6)
        J6[:3] = (w * b * da_drho) / (b * b)  # db/drho = 0
        J6[3:] = (w * (b * da_dphi - a * db_dphi)) / (b * b)

        return w * r, J6

    @staticmethod
    def residual_and_jac_rev(R, t, pT, qS, nS, dT0, params: IncidenceParams):
        """
        One-pose reverse residual and Jacobian (consistent, left perturbation).

        Full quotient rule: dr/dphi = (b * da/dphi - a * db/dphi) / b^2

        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3,)
            pT: Target point / ray origin in target frame (3,)
            qS: Source surface hit point (3,)
            nS: Source surface normal (3,)
            dT0: Ray direction in target frame (3,)
            params: Incidence weighting parameters

        Returns:
            (w*r, w*J6) where w is incidence weight
        """
        q = pT  # Ray origin in target-local
        p = qS  # Hit point in source-local
        ns = nS  # Normal in source-local
        d0 = dT0  # Ray direction in target-local

        # Inverse-transform q to source frame
        xprime = R.T @ (q - t)

        # Ray direction in source frame
        dprime = R.T @ d0

        # Rotated normal (source normal in target frame)
        nbar = R @ ns

        # Denominators
        b = float(ns @ dprime)

        # Residual in source frame
        a = float(ns @ (xprime - p))
        r = a / b

        w = incidence_weight(b, params)
        if w == 0.0:
            return r, np.zeros(6)

        # da/drho = -nbar, da/dphi = (nbar × q)
        # db/drho = 0, db/dphi = (nbar × d0)
        da_drho = -nbar
        da_dphi = np.cross(nbar, q)
        db_dphi = np.cross(nbar, d0)

        # Full quotient rule: dr = (b*da - a*db) / b^2
        J6 = np.zeros(6)
        J6[:3] = (w * b * da_drho) / (b * b)  # db/drho = 0
        J6[3:] = (w * (b * da_dphi - a * db_dphi)) / (b * b)

        return w * r, J6


# Type alias for convenience - default to simplified
Tangent = TangentSimplified
