@page jacobian_validation Validating Left vs Right Jacobians on SO(3)

This document describes how to numerically validate Jacobians on SO(3) when using
left (space / fixed frame) and right (body / moving frame) perturbations.

@section jv_problem_setup Problem Setup

Let \f$R\f$ be a rotation matrix in \f$\mathrm{SO}(3)\f$.
Let \f$f(R)\f$ be a vector-valued function.

Two common perturbation models:

**Left perturbation (space frame):**
\f[
  R(\varepsilon) = \exp(\varepsilon \, \widehat{\boldsymbol{\delta}}) \, R
\f]

**Right perturbation (body frame):**
\f[
  R(\varepsilon) = R \, \exp(\varepsilon \, \widehat{\boldsymbol{\delta}})
\f]

The Jacobians depend on which perturbation model is used.

@section jv_fd_validation Finite-Difference Validation (Primary Test)

For a small \f$\varepsilon\f$ and direction \f$\boldsymbol{\delta} \in \mathbb{R}^3\f$,
define the central difference:
\f[
  \frac{\mathrm{d}f}{\mathrm{d}\varepsilon}\bigg|_{\text{num}}
  = \frac{f(R_+) - f(R_-)}{2\varepsilon}
\f]

Use central differences for accuracy.

@subsection jv_left_test Left Jacobian Test

Perturbations:
\f[
  R_+ = \exp(\varepsilon \, \widehat{\boldsymbol{\delta}}) \, R, \qquad
  R_- = \exp(-\varepsilon \, \widehat{\boldsymbol{\delta}}) \, R
\f]

Analytical prediction:
\f[
  \frac{\mathrm{d}f}{\mathrm{d}\varepsilon}\bigg|_{\text{ana}}
  = J_{\text{left}}(R) \, \boldsymbol{\delta}
\f]

Validation criterion:
\f[
  \frac{\left\| \frac{\mathrm{d}f}{\mathrm{d}\varepsilon}\big|_{\text{num}}
        - \frac{\mathrm{d}f}{\mathrm{d}\varepsilon}\big|_{\text{ana}} \right\|}
       {\max\left(1, \left\| \frac{\mathrm{d}f}{\mathrm{d}\varepsilon}\big|_{\text{num}} \right\|\right)}
  \ll 1
\f]

@subsection jv_right_test Right Jacobian Test

Perturbations:
\f[
  R_+ = R \, \exp(\varepsilon \, \widehat{\boldsymbol{\delta}}), \qquad
  R_- = R \, \exp(-\varepsilon \, \widehat{\boldsymbol{\delta}})
\f]

Analytical prediction:
\f[
  \frac{\mathrm{d}f}{\mathrm{d}\varepsilon}\bigg|_{\text{ana}}
  = J_{\text{right}}(R) \, \boldsymbol{\delta}
\f]

@subsection jv_practical_notes Practical Notes

- Use \f$\varepsilon \sim 10^{-6}\f$ to \f$10^{-8}\f$ (double precision)
- Sweep \f$\varepsilon\f$ over a decade and look for a plateau
- Test random \f$R\f$ and random \f$\boldsymbol{\delta}\f$ (unit and scaled)
- Compare directional derivatives, not full matrices

This catches sign errors, frame mixups, missing transposes, etc.

@section jv_cross_validation Cross-Validation Between Left and Right Jacobians

Left and right perturbations represent the same physical motion if their
tangent vectors satisfy:
\f[
  \boldsymbol{\delta}_{\text{space}} = R \, \boldsymbol{\delta}_{\text{body}}
\f]

This implies the Jacobians must satisfy:
\f[
  J_{\text{right}}(R) = J_{\text{left}}(R) \, R
\f]

Equivalently:
\f[
  J_{\text{left}}(R) = J_{\text{right}}(R) \, R^\top
\f]

Numeric check:
\f[
  \| J_{\text{right}} - J_{\text{left}} \, R \| \approx 0
\f]

If this fails but finite differences pass, your tangent convention differs
(e.g., reversed delta mapping, quaternion local parameterization, etc.).
An adjoint relationship still exists for the correct convention.

@section jv_sanity_test Simple Sanity Test Function

Use a function with a known derivative:
\f[
  f(R) = R \, \mathbf{p}
\f]
where \f$\mathbf{p}\f$ is a fixed vector in \f$\mathbb{R}^3\f$.

**Left perturbation result:**
\f[
  \mathrm{d}f = \boldsymbol{\delta}_{\text{space}} \times (R \, \mathbf{p})
\f]

**Right perturbation result:**
\f[
  \mathrm{d}f = R \, (\boldsymbol{\delta}_{\text{body}} \times \mathbf{p})
\f]

This test is extremely useful for debugging sign and frame errors.

@section jv_checklist Validation Checklist

1. Finite-difference test \f$J_{\text{left}}\f$
2. Finite-difference test \f$J_{\text{right}}\f$
3. Verify \f$J_{\text{right}} = J_{\text{left}} \, R\f$
4. Sanity check with \f$f(R) = R \, \mathbf{p}\f$

If all four pass, the Jacobians are correct.
