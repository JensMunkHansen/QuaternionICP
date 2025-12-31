@page epsilon_sweep_guide Finite-Difference Jacobian Validation on SE(3)

This document explains how to validate analytic Jacobians using finite differences
on SE(3) with right-multiplication perturbations. The key technique is
**epsilon sweeping**: never trust a single \f$\varepsilon\f$; trust the plateau.

@section esg_overview Overview

Finite-difference (FD) validation compares an analytic Jacobian against numerical
derivatives. For SE(3), this requires:

1. Using the correct perturbation convention (right-multiplication)
2. Sweeping \f$\varepsilon\f$ to find a stable plateau
3. Understanding why translation and rotation behave differently

@section esg_competing_errors Two Competing Errors

The central difference approximation:
\f[
  \frac{\partial r}{\partial x}
  \approx
  \frac{r(x + \varepsilon) - r(x - \varepsilon)}{2\varepsilon}
\f]

is affected by two competing error sources.

@subsection esg_truncation Truncation Error (large \f$\varepsilon\f$)

For large \f$\varepsilon\f$, higher-order Taylor terms dominate:
\f[
  r(x + \varepsilon) = r(x) + \varepsilon r'(x) + \frac{\varepsilon^2}{2} r''(x) + O(\varepsilon^3)
\f]

Central differences cancel the first-order error, leaving:
\f[
  \text{Truncation error} = O(\varepsilon^2)
\f]

@subsection esg_roundoff Roundoff Error (small \f$\varepsilon\f$)

For very small \f$\varepsilon\f$:
- \f$r(x + \varepsilon)\f$ and \f$r(x - \varepsilon)\f$ become numerically indistinguishable
- Subtraction causes catastrophic cancellation
- Floating-point rounding dominates

This error scales as:
\f[
  \text{Roundoff error} \sim O\!\left(\frac{\epsilon_{\text{machine}}}{\varepsilon}\right)
\f]

@section esg_plateau The Plateau Phenomenon

Because truncation error decreases with \f$\varepsilon\f$ while roundoff error
increases, plotting FD error versus \f$\varepsilon\f$ yields a characteristic shape:

```
  error
    |
    |\
    | \
    |  \        truncation-dominated
    |   \
    |    -------------------  <- plateau (sweet spot)
    |                     \
    |                      \  roundoff-dominated
    +-----------------------------> log10(eps)
```

The **plateau** is the stable region where both errors are small. A correct
analytic Jacobian must match the FD values on this plateau.

@subsection esg_plateau_interpretation Interpreting Results

**Case A: No plateau exists**
- Wrong perturbation convention (left vs right)
- Incorrect translation update formula
- Residual has discontinuities or branching

**Case B: Plateau exists but analytic Jacobian does not match**
- Missing Jacobian terms (e.g., denominator derivative)
- Sign errors or wrong frame for cross products
- Transpose errors

**Case C: Plateau exists and analytic Jacobian matches**
- Jacobian is validated

@section esg_se3_convention SE(3) Right-Multiplication Convention

A pose \f$T = (R, \mathbf{t}) \in \mathrm{SE}(3)\f$ acts on a point \f$\mathbf{p}\f$ as:
\f[
  \mathbf{x} = R\,\mathbf{p} + \mathbf{t}
\f]

Right-multiplication perturbation:
\f[
  T(\boldsymbol{\delta}) = T \exp(\widehat{\boldsymbol{\delta}}), \qquad
  \boldsymbol{\delta} = \begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^6
\f]

The SE(3) exponential produces:
\f[
  \exp(\widehat{\boldsymbol{\delta}}) =
  \begin{bmatrix}
    \Delta R & \Delta\mathbf{t} \\
    0 & 1
  \end{bmatrix}
\f]

Under right-multiplication, the pose updates as:
\f[
  R' = R \, \Delta R, \qquad
  \mathbf{t}' = \mathbf{t} + R \, \Delta\mathbf{t}
\f]

@section esg_critical_pitfall The Critical Translation Pitfall

This is the most common source of FD validation failures on SE(3).

For a pure translation perturbation (\f$\boldsymbol{\omega} = 0\f$):
\f[
  \Delta R = I, \qquad \Delta\mathbf{t} = \mathbf{v}
\f]

Therefore:
\f[
  \mathbf{t}' = \mathbf{t} + R\,\mathbf{v}
\f]

**Incorrect (world-frame translation):**
\f[
  \mathbf{t}' = \mathbf{t} + \varepsilon \mathbf{e}_k \quad \text{(WRONG)}
\f]

**Correct (right-multiplication translation):**
\f[
  \mathbf{t}' = \mathbf{t} + R(\varepsilon \mathbf{e}_k) \quad \text{(CORRECT)}
\f]

The translation increment lives in the **local/body frame** and must be
rotated by \f$R\f$ to obtain the world-frame change.

@section esg_fd_formulas Finite-Difference Formulas

Let \f$r(T)\f$ be a scalar residual.

@subsection esg_fd_translation Translation Components \f$v_x, v_y, v_z\f$

For \f$k \in \{0, 1, 2\}\f$:
\f[
  R_\pm = R, \qquad
  \mathbf{t}_\pm = \mathbf{t} + R(\pm\varepsilon \mathbf{e}_k)
\f]
\f[
  J_{\text{FD}}[k] \approx
  \frac{r(R_+, \mathbf{t}_+) - r(R_-, \mathbf{t}_-)}{2\varepsilon}
\f]

@subsection esg_fd_rotation Rotation Components \f$\omega_x, \omega_y, \omega_z\f$

For \f$k \in \{0, 1, 2\}\f$:
\f[
  R_\pm = R \exp((\pm\varepsilon \mathbf{e}_k)^\wedge), \qquad
  \mathbf{t}_\pm = \mathbf{t}
\f]
\f[
  J_{\text{FD}}[3+k] \approx
  \frac{r(R_+, \mathbf{t}_+) - r(R_-, \mathbf{t}_-)}{2\varepsilon}
\f]

@section esg_linear_vs_nonlinear Why Translation and Rotation Behave Differently

For the ray-projection residual \f$r = a/b\f$ with:
\f[
  a = \mathbf{n}^\top (R\mathbf{p} + \mathbf{t} - \mathbf{q}), \qquad
  b = \mathbf{n}^\top (R\mathbf{d})
\f]

@subsection esg_translation_linear Translation: Linear Dependence

The numerator \f$a\f$ is **affine linear** in \f$\mathbf{t}\f$, and the
denominator \f$b\f$ does not depend on \f$\mathbf{t}\f$.

Therefore:
\f[
  \frac{\partial r}{\partial \mathbf{t}} = \frac{\mathbf{n}^\top}{b} \quad \text{(constant)}
\f]

**Consequence:** Finite differences are exact for linear functions. There is
no truncation error, so:
- Large \f$\varepsilon\f$ values work fine
- Relative error reaches machine precision (\f$\sim 10^{-15}\f$)
- No visible plateau (flat line across all \f$\varepsilon\f$)

@subsection esg_rotation_nonlinear Rotation: Nonlinear Dependence

Both \f$a\f$ and \f$b\f$ depend on \f$R\f$:
\f[
  \frac{\partial r}{\partial \boldsymbol{\omega}}
  = \frac{1}{b}\frac{\partial a}{\partial \boldsymbol{\omega}}
  - \frac{a}{b^2}\frac{\partial b}{\partial \boldsymbol{\omega}}
\f]

The quotient rule is essential. Omitting \f$\partial b/\partial\boldsymbol{\omega}\f$
gives the "simplified" Jacobian, which can have \f$\sim 10^{-2}\f$ error.

**Consequence:**
- Truncation error is present
- A genuine plateau appears at \f$\varepsilon \sim 10^{-6}\f$ to \f$10^{-8}\f$
- Consistent Jacobian matches FD at \f$\sim 10^{-12}\f$

@section esg_practical_recipe Practical Epsilon Sweep Recipe

@subsection esg_sweep_range Choose a Logarithmic Sweep

For double precision, sweep:
\f[
  \varepsilon \in [10^{-2}, 10^{-9}]
\f]

Example values:
```
1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 3e-9, 1e-9
```

@subsection esg_sweep_procedure Procedure

1. Compute the analytic Jacobian \f$J_{\text{ana}}\f$ at the current pose
2. For each \f$\varepsilon\f$ in the sweep:
   - For each dimension \f$i \in \{0, \ldots, 5\}\f$:
     - Apply the correct SE(3) perturbation (see @ref esg_fd_formulas)
     - Compute \f$J_{\text{FD}}[i](\varepsilon)\f$
   - Compute error metrics:
     \f[
       \text{Absolute: } |J_{\text{FD}}[i] - J_{\text{ana}}[i]|
     \f]
     \f[
       \text{Relative: } \frac{|J_{\text{FD}}[i] - J_{\text{ana}}[i]|}{\max(1, |J_{\text{ana}}[i]|)}
     \f]
3. Plot or log error versus \f$\varepsilon\f$
4. Identify the plateau region

@subsection esg_typical_ranges Typical Plateau Ranges

| Component   | Plateau Range | Notes |
|-------------|---------------|-------|
| Translation | Any reasonable \f$\varepsilon\f$ | Linear dependence |
| Rotation    | \f$10^{-6}\f$ to \f$10^{-8}\f$ | Nonlinear, sweep required |

@section esg_quotient_rule Validating Simplified vs Consistent Jacobians

The difference between consistent and simplified rotation Jacobians is:
\f[
  J_{\text{cons}}(\omega_k) - J_{\text{simp}}(\omega_k)
  = -\frac{a}{b^2} \frac{\partial b}{\partial \omega_k}
\f]

This term becomes large when:
- \f$|a|\f$ is not small (far from convergence)
- \f$|b| = |\mathbf{n}^\top \mathbf{d}|\f$ is small (grazing angle)
- \f$|\partial b/\partial\omega_k|\f$ is significant

For the forward case (\f$\mathbf{d} = R\mathbf{d}_0\f$):
\f[
  \frac{\partial b}{\partial \omega_k} = -\mathbf{n}^\top R(\mathbf{d}_0 \times \mathbf{e}_k)
\f]

For the backward case (\f$\mathbf{d} = R^\top\mathbf{d}_0\f$):
\f[
  \frac{\partial b}{\partial \omega_k} = -\mathbf{n}^\top (\mathbf{d} \times \mathbf{e}_k)
\f]

@section esg_checklist Validation Checklist

1. Use correct SE(3) right-multiplication perturbations
2. Sweep \f$\varepsilon\f$ logarithmically from \f$10^{-2}\f$ to \f$10^{-9}\f$
3. Identify the plateau region
4. Verify analytic Jacobian matches FD on the plateau
5. Expect translation to be flat (linear) and rotation to show a plateau (nonlinear)
6. If using simplified Jacobians, verify the missing term analytically

@section esg_takeaway Key Takeaways

- **Never trust a single \f$\varepsilon\f$; trust the plateau**
- **For right-multiplication, translation FD must use \f$\mathbf{t} + R(\varepsilon\mathbf{e}_k)\f$**
- **Translation is linear (no plateau); rotation is nonlinear (plateau at \f$10^{-6}\f$ to \f$10^{-8}\f$)**
- **The quotient rule is essential for rotation Jacobians of ratio residuals**
