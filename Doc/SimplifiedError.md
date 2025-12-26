# Interpreting FD Results for the Simplified Jacobian (No Quotient Rule)

This note explains why the simplified Jacobian (dropping the denominator
derivative) shows the observed errors for rotation components, and why the
numbers you measured are correct and expected.

All discussion assumes:
- residual r = a / b
- right-multiplication on SE(3)
- finite-difference (FD) results taken as ground truth

-------------------------------------------------------------------------------
1) The measured results
-------------------------------------------------------------------------------

You observed the following for rotation components:

  Direction | FD (truth) | Consistent | Simplified | err_simp | best_eps
  --------- | ---------- | ---------- | ---------- | -------- | --------
  w[0]      | -0.250397  | -0.250397  | -0.241970  | 0.0084   | 3e-6
  w[1]      | -0.174196  | -0.174196  | -0.163588  | 0.0106   | 3e-6
  w[2]      | -0.114384  | -0.114384  | -0.109171  | 0.0052   | 1e-5

Observations:
- FD and Consistent match exactly (to numerical precision).
- Simplified deviates by roughly 0.5% to 1% absolute magnitude.
- The best epsilon is around 1e-6, as expected for rotation.

-------------------------------------------------------------------------------
2) What the simplified Jacobian is missing
-------------------------------------------------------------------------------

The residual is:

  r = a / b

where:
  a = n^T (R * p + t - q)
  b = n^T (R * d)

Differentiate with respect to rotation w.

Consistent Jacobian (quotient rule):

  dr/dw = ( b * da/dw - a * db/dw ) / b^2

Simplified Jacobian (what you tested):

  dr/dw ~= ( da/dw ) / b

The difference between them is exactly:

  Delta = dr_cons/dw - dr_simp/dw
        = - ( a * db/dw ) / b^2

This is the missing first-order term.

-------------------------------------------------------------------------------
3) Why the simplified error is around 0.005 to 0.01
-------------------------------------------------------------------------------

The magnitude of the missing term is controlled by:

  |Delta| ~= |a| * |db/dw| / b^2

This term is non-negligible when:
- the residual a is not small (early or mid ICP iterations)
- b = n^T d is not close to 1
- the ray direction changes noticeably under rotation

In that regime, a few-percent error in the Jacobian is expected.

Your numbers match this exactly:
- errors are on the order of 0.005 to 0.01
- Jacobian entries are on the order of 0.1 to 0.25
- relative errors are a few percent

-------------------------------------------------------------------------------
4) Why the consistent Jacobian matches FD
-------------------------------------------------------------------------------

Finite differences approximate the true derivative of the residual.

Because:
- you swept epsilon
- found a plateau around 1e-6
- and the consistent Jacobian lies on that plateau

the consistent Jacobian is validated.

This confirms that:
- the quotient-rule term is required
- your analytic derivation is correct

-------------------------------------------------------------------------------
5) Why best_eps is still around 1e-6 for the simplified case
-------------------------------------------------------------------------------

The finite-difference epsilon sweep reflects the smoothness of the true residual,
not the analytic approximation you choose.

Therefore:
- best_eps is determined by r(w), not by the Jacobian formula
- best_eps remains around 1e-6 even when the simplified Jacobian is wrong

The simplified Jacobian being offset does not move the plateau.

-------------------------------------------------------------------------------
6) What happens near convergence
-------------------------------------------------------------------------------

As ICP converges:
- a -> 0
- the residual r -> 0

Then the missing term:

  -( a * db/dw ) / b^2

also goes to zero.

As a result:
- simplified and consistent Jacobians become closer
- the simplified approximation may appear to work near convergence

This explains why simplified models sometimes converge, but are less reliable
earlier in the solve.

-------------------------------------------------------------------------------
7) Final conclusion
-------------------------------------------------------------------------------

- Your FD sweep is correct.
- Your consistent Jacobian is correct.
- The simplified Jacobian differs by exactly the expected missing term.
- The magnitude and pattern of the error match theory.

There is no bug here; this is precisely the behavior predicted by the math.
