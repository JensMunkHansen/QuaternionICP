
# Finite-Difference Jacobian Validation on SE(3)
## Epsilon sweep and the plateau test (right-multiplication)

Goal
----
Validate an analytic Jacobian by comparing it to finite differences (FD).
Do NOT trust a single epsilon. Sweep epsilon and look for a stable region
(the "plateau") where the FD estimate is insensitive to epsilon.

This note is ASCII-only and focuses on right-multiplication:
  T_plus = T * Exp(delta)

-------------------------------------------------------------------------------
1) Central finite difference
-------------------------------------------------------------------------------

For a scalar residual r(x), the central difference approximation is:

  dr/dx ~= ( r(x + eps) - r(x - eps) ) / (2 * eps)

For a vector parameter, apply the same formula per component i:

  J_FD[i](eps) ~= ( r(Plus(x, +eps * e_i)) - r(Plus(x, -eps * e_i)) ) / (2 * eps)

Important:
- "Plus" must match the perturbation convention used by your analytic Jacobian.
- Here we use SE(3) right-multiplication (see Section 4).

-------------------------------------------------------------------------------
2) Why epsilon must be swept: two competing errors
-------------------------------------------------------------------------------

The FD estimate has two dominant error sources.

(1) Truncation error (eps too large)
- r(x +/- eps) includes higher-order Taylor terms.
- Central difference truncation error is typically O(eps^2).
- Result: biased derivative (nonlinear effects contaminate the slope).

(2) Floating-point / cancellation error (eps too small)
- r(x + eps) and r(x - eps) become numerically indistinguishable.
- Subtraction loses significant digits; rounding dominates.
- Error behaves roughly like O(machine_epsilon / eps).
- Result: noisy, unstable derivative estimates.

Because one error grows with eps and the other grows as eps shrinks, there is
an intermediate range where the estimate is stable.

-------------------------------------------------------------------------------
3) The plateau picture (what you want to see)
-------------------------------------------------------------------------------

If you sweep eps from large to small, a typical error-vs-eps curve looks like:

  error
    |
    |\
    | \
    |  \        truncation-dominated (eps too large)
    |   \
    |    --------------------  plateau (stable FD)
    |                     \
    |                      \  roundoff-dominated (eps too small)
    +-------------------------------------------------> log10(eps)

Interpretation:
- On the left: eps too large -> FD is biased (nonlinear terms).
- On the right: eps too small -> FD is noisy (cancellation/roundoff).
- In the middle: plateau -> FD is trustworthy.

Validation rule:
- A correct analytic Jacobian should match the FD values ON the plateau.

-------------------------------------------------------------------------------
4) SE(3) right-multiplication: correct "Plus" for FD checks
-------------------------------------------------------------------------------

Pose T = (R, t) in SE(3), acting on points as:
  x = R * p + t

Right-multiplication update:
  T_plus = T * Exp(delta)
where delta is 6D:
  delta = [ v ; w ]
  v = translation perturbation (local/body frame)
  w = rotation perturbation (local/body frame)

For small delta, the update has the form:
  R_plus = R * dR
  t_plus = t + R * dt

Key consequence for FD:
- Translation perturbations live in the LOCAL (body) frame.
- Therefore you cannot FD-check translation by doing t_plus = t + eps * e_i
  (that would correspond to a different parameterization / convention).

-------------------------------------------------------------------------------
5) FD perturbations for right-multiplication (translation and rotation)
-------------------------------------------------------------------------------

Let r(T) be any scalar residual computed from the current pose.

A) Translation columns (v_x, v_y, v_z)
--------------------------------------

To FD-check translation component i in {0,1,2}:

  R_plus  = R
  R_minus = R

  t_plus  = t + R * ( +eps * e_i )
  t_minus = t + R * ( -eps * e_i )

  J_FD[i](eps) ~= ( r(R_plus, t_plus) - r(R_minus, t_minus) ) / (2 * eps)

Notes:
- e_i is a unit basis vector in R^3.
- This matches right-multiplication because the translation increment is rotated by R.

B) Rotation columns (w_x, w_y, w_z)
-----------------------------------

To FD-check rotation component i in {0,1,2}:

  R_plus  = R * ExpSO3( +eps * e_i )
  R_minus = R * ExpSO3( -eps * e_i )

  t_plus  = t
  t_minus = t

  J_FD[3+i](eps) ~= ( r(R_plus, t_plus) - r(R_minus, t_minus) ) / (2 * eps)

Where ExpSO3(theta * e_i) is the SO(3) exponential map for axis e_i.

-------------------------------------------------------------------------------
6) How to sweep epsilon (practical recipe)
-------------------------------------------------------------------------------

Use a LOG sweep over multiple decades. For double precision, a typical range is:

  eps in [1e-2, 1e-9]

Example sweep values (log-spaced):
  1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 3e-9, 1e-9

Procedure:
1) Compute analytic Jacobian J_ana at the current pose T.
2) For each eps in the sweep:
   a) For each dimension i (0..5), compute J_FD[i](eps) using the correct SE(3) Plus above.
   b) Compute an error metric per i, e.g.:
        abs_err[i](eps) = abs( J_FD[i](eps) - J_ana[i] )
      or a relative error:
        rel_err[i](eps) = abs(J_FD[i](eps) - J_ana[i]) / max(1, abs(J_ana[i]))
3) Look for an interval of eps where:
   - J_FD[i](eps) is stable (changes little as eps changes), AND
   - J_FD[i](eps) matches J_ana[i].

This interval is the plateau.

-------------------------------------------------------------------------------
7) Diagnosing outcomes
-------------------------------------------------------------------------------

Case A: No plateau (FD values never stabilize)
- Likely wrong perturbation convention (left vs right).
- Common SE(3) pitfall: translation perturbed as t + eps*e_i instead of t + R*(eps*e_i).
- Also possible: residual has discontinuities (clamps, branching, hit/miss changes).

Case B: Plateau exists but analytic does not match
- Analytic Jacobian likely missing terms or has a sign/frame error.
- For ray-projection residuals, a common missing term is the derivative of the denominator.

Case C: Plateau exists and analytic matches
- Jacobian is validated for that residual and that implementation.

-------------------------------------------------------------------------------
8) One-line takeaway
-------------------------------------------------------------------------------

Never trust a single epsilon; trust the plateau obtained by sweeping epsilon,
using the SAME SE(3) right-multiplication "Plus" rule as your analytic Jacobian.
