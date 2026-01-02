@page simplified_jacobian_analysis Interpreting FD Results for the Simplified Jacobian

This document explains why the simplified Jacobian (dropping the denominator
derivative) shows the observed errors for rotation components, and why the
measured results are correct and expected.

Assumptions:
- Residual \f$r = a/b\f$
- Right-multiplication on SE(3)
- Finite-difference (FD) results taken as ground truth

@section sja_results Measured Results

Observed FD sweep results for rotation components:

| Direction | FD (truth) | Consistent | Simplified | Error | Best \f$\varepsilon\f$ |
|-----------|------------|------------|------------|-------|------------------------|
| \f$\omega_0\f$ | \f$-0.2504\f$ | \f$-0.2504\f$ | \f$-0.2420\f$ | 0.0084 | \f$3 \times 10^{-6}\f$ |
| \f$\omega_1\f$ | \f$-0.1742\f$ | \f$-0.1742\f$ | \f$-0.1636\f$ | 0.0106 | \f$3 \times 10^{-6}\f$ |
| \f$\omega_2\f$ | \f$-0.1144\f$ | \f$-0.1144\f$ | \f$-0.1092\f$ | 0.0052 | \f$1 \times 10^{-5}\f$ |

Observations:
- FD and consistent match exactly (to numerical precision)
- Simplified deviates by roughly 0.5% to 1% absolute magnitude
- Best epsilon is around \f$10^{-6}\f$, as expected for rotation

@section sja_missing_term What the Simplified Jacobian is Missing

The residual is:
\f[
  r = \frac{a}{b}
\f]

where:
\f[
  a = \mathbf{n}^\top (R\mathbf{p} + \mathbf{t} - \mathbf{q}), \qquad
  b = \mathbf{n}^\top (R\mathbf{d})
\f]

**Consistent Jacobian** (full quotient rule):
\f[
  \frac{\partial r}{\partial \boldsymbol{\omega}}
  = \frac{b \dfrac{\partial a}{\partial \boldsymbol{\omega}}
        - a \dfrac{\partial b}{\partial \boldsymbol{\omega}}}{b^2}
\f]

**Simplified Jacobian** (ignores denominator derivative):
\f[
  \left.\frac{\partial r}{\partial \boldsymbol{\omega}}\right|_{\text{simp}}
  = \frac{1}{b}\frac{\partial a}{\partial \boldsymbol{\omega}}
\f]

The difference is exactly the missing first-order term:
\f[
  \Delta = \frac{\partial r}{\partial \boldsymbol{\omega}}\bigg|_{\text{cons}}
         - \frac{\partial r}{\partial \boldsymbol{\omega}}\bigg|_{\text{simp}}
         = -\frac{a}{b^2}\frac{\partial b}{\partial \boldsymbol{\omega}}
\f]

@section sja_error_magnitude Why the Simplified Error is Around 0.005 to 0.01

The magnitude of the missing term:
\f[
  |\Delta| \approx \frac{|a| \cdot |\partial b/\partial\boldsymbol{\omega}|}{b^2}
\f]

This term is non-negligible when:
- The residual \f$a\f$ is not small (early or mid ICP iterations)
- \f$b = \mathbf{n}^\top\mathbf{d}\f$ is not close to 1
- The ray direction changes noticeably under rotation

The measured numbers match this exactly:
- Errors on the order of 0.005 to 0.01
- Jacobian entries on the order of 0.1 to 0.25
- Relative errors of a few percent

@section sja_consistent_validated Why the Consistent Jacobian Matches FD

Finite differences approximate the true derivative of the residual.

Because:
1. Epsilon was swept over multiple decades
2. A plateau was found around \f$10^{-6}\f$
3. The consistent Jacobian lies on that plateau

The consistent Jacobian is validated. This confirms:
- The quotient-rule term is required
- The analytic derivation is correct

@section sja_best_eps Why Optimal Epsilon Remains 1e-6 for Simplified

The FD epsilon sweep reflects the smoothness of the **true residual**,
not the analytic approximation chosen.

Therefore:
- Best \f$\varepsilon\f$ is determined by \f$r(\boldsymbol{\omega})\f$, not by the Jacobian formula
- Best \f$\varepsilon\f$ remains around \f$10^{-6}\f$ even when the simplified Jacobian is wrong

The simplified Jacobian being offset does not move the plateau — it just
doesn't lie on it.

@section sja_convergence What Happens Near Convergence

As ICP converges:
\f[
  a \to 0, \qquad r \to 0
\f]

Then the missing term:
\f[
  -\frac{a}{b^2}\frac{\partial b}{\partial \boldsymbol{\omega}} \to 0
\f]

As a result:
- Simplified and consistent Jacobians become closer
- The simplified approximation may appear to work near convergence

This explains why simplified models sometimes converge, but are less reliable
earlier in the solve when \f$|a|\f$ is large.

@section sja_conclusion Conclusion

- The FD sweep is correct
- The consistent Jacobian is correct
- The simplified Jacobian differs by exactly the expected missing term
- The magnitude and pattern of error match theory

There is no bug — this is precisely the behavior predicted by the math.
