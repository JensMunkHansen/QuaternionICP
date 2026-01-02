@page ray_projection_jacobian Ray-Projection Residual Jacobians on SE(3)

This document derives the Jacobians of the ray-projection residual used in
point-to-plane ICP, explaining why the **consistent** Jacobian (full quotient rule)
is necessary for rotation while the **simplified** Jacobian suffices for translation.

@section rpj_residual Residual Definition

The ray-projection residual measures signed distance along a ray direction:
\f[
  r = \frac{a}{b}
\f]

where:
\f[
  a = \mathbf{n}^\top (\mathbf{x} - \mathbf{q}), \qquad
  b = \mathbf{n}^\top \mathbf{d}
\f]

- \f$\mathbf{x}\f$: transformed source point
- \f$\mathbf{q}\f$: target hit point
- \f$\mathbf{n}\f$: target surface normal
- \f$\mathbf{d}\f$: ray direction (transformed)

@subsection rpj_forward Forward Case (Source to Target)

\f[
  \mathbf{x} = R\mathbf{p} + \mathbf{t}, \qquad
  \mathbf{d} = R\mathbf{d}_0
\f]

where \f$\mathbf{p}\f$ is the source point and \f$\mathbf{d}_0\f$ is the local ray direction.

@subsection rpj_backward Backward Case (Target to Source)

\f[
  \mathbf{y} = R^\top(\mathbf{p} - \mathbf{t}), \qquad
  \mathbf{d} = R^\top\mathbf{d}_0
\f]

@section rpj_quotient_rule The Quotient Rule

For \f$r = a/b\f$, the full derivative is:
\f[
  \frac{\partial r}{\partial \theta}
  = \frac{1}{b}\frac{\partial a}{\partial \theta}
  - \frac{a}{b^2}\frac{\partial b}{\partial \theta}
\f]

The **consistent** Jacobian uses both terms. The **simplified** Jacobian
ignores \f$\partial b/\partial\theta\f$:
\f[
  \left.\frac{\partial r}{\partial \theta}\right|_{\text{simp}}
  = \frac{1}{b}\frac{\partial a}{\partial \theta}
\f]

@section rpj_translation Translation Jacobian

@subsection rpj_trans_numerator Numerator Dependence

The numerator:
\f[
  a = \mathbf{n}^\top (R\mathbf{p} + \mathbf{t} - \mathbf{q})
\f]

is **affine linear** in \f$\mathbf{t}\f$.

@subsection rpj_trans_denominator Denominator Dependence

The denominator:
\f[
  b = \mathbf{n}^\top (R\mathbf{d}_0)
\f]

does **not** depend on \f$\mathbf{t}\f$.

@subsection rpj_trans_jacobian Translation Jacobian (Both Consistent and Simplified)

Since \f$\partial b/\partial\mathbf{t} = 0\f$:
\f[
  \frac{\partial r}{\partial \mathbf{t}}
  = \frac{1}{b}\frac{\partial a}{\partial \mathbf{t}}
  = \frac{\mathbf{n}^\top}{b}
\f]

This is **constant** — it does not depend on \f$\mathbf{t}\f$.

@subsection rpj_trans_fd Consequence for Finite Differences

Finite differences are exact for linear functions:
\f[
  \frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon} = f'(x)
  \quad \text{(exact)}
\f]

Therefore:
- Large \f$\varepsilon\f$ values work (e.g., \f$10^{-2}\f$)
- Relative error reaches machine precision (\f$\sim 10^{-15}\f$)
- No visible plateau — flat line across all \f$\varepsilon\f$
- **Consistent and simplified Jacobians are identical**

@section rpj_rotation Rotation Jacobian

@subsection rpj_rot_dependence Both Terms Depend on Rotation

Numerator:
\f[
  a(R) = \mathbf{n}^\top (R\mathbf{p} + \mathbf{t} - \mathbf{q})
\f]

Denominator:
\f[
  b(R) = \mathbf{n}^\top (R\mathbf{d}_0)
\f]

Both depend on \f$R\f$, so the full quotient rule applies:
\f[
  \frac{\partial r}{\partial \boldsymbol{\omega}}
  = \frac{1}{b}\frac{\partial a}{\partial \boldsymbol{\omega}}
  - \frac{a}{b^2}\frac{\partial b}{\partial \boldsymbol{\omega}}
\f]

@subsection rpj_rot_missing The Missing Term

The difference between consistent and simplified:
\f[
  J_{\text{cons}} - J_{\text{simp}}
  = -\frac{a}{b^2}\frac{\partial b}{\partial \boldsymbol{\omega}}
\f]

This term is **not negligible** in general.

@subsection rpj_rot_db_dw Computing the Denominator Derivative

For right-multiplication perturbation \f$R \to R\exp(\widehat{\boldsymbol{\omega}})\f$:

**Forward case** (\f$\mathbf{d} = R\mathbf{d}_0\f$):

First-order variation:
\f[
  \mathbf{d}' \approx \mathbf{d} + R(\boldsymbol{\omega} \times \mathbf{d}_0)
  = \mathbf{d} - R(\mathbf{d}_0 \times \boldsymbol{\omega})
\f]

Therefore:
\f[
  \frac{\partial \mathbf{d}}{\partial \boldsymbol{\omega}} = -R[\mathbf{d}_0]_\times
\f]

And:
\f[
  \frac{\partial b}{\partial \omega_k}
  = \mathbf{n}^\top \frac{\partial \mathbf{d}}{\partial \omega_k}
  = -\mathbf{n}^\top R(\mathbf{d}_0 \times \mathbf{e}_k)
\f]

**Backward case** (\f$\mathbf{d} = R^\top\mathbf{d}_0\f$):

Under right-multiplication: \f$R'^\top = \exp(-\widehat{\boldsymbol{\omega}})R^\top\f$

First-order variation:
\f[
  \mathbf{d}' \approx \mathbf{d} - \boldsymbol{\omega} \times \mathbf{d}
\f]

Therefore:
\f[
  \frac{\partial \mathbf{d}}{\partial \boldsymbol{\omega}} = -[\mathbf{d}]_\times
\f]

And:
\f[
  \frac{\partial b}{\partial \omega_k}
  = -\mathbf{n}^\top (\mathbf{d} \times \mathbf{e}_k)
\f]

@section rpj_when_simplified When Simplified Fails

The missing term magnitude:
\f[
  \left| -\frac{a}{b^2}\frac{\partial b}{\partial \omega_k} \right|
\f]

becomes large when:

| Condition | Effect |
|-----------|--------|
| \f$\|a\|\f$ large | Far from convergence |
| \f$\|b\|\f$ small | Grazing angle (\f$\mathbf{n} \perp \mathbf{d}\f$) |
| \f$\|\partial b/\partial\omega_k\|\f$ large | Ray direction sensitive to rotation |

For random geometry, the simplified rotation Jacobian can show **~1% to 60% error**
compared to finite differences, while the consistent Jacobian matches to \f$\sim 10^{-10}\f$.

@section rpj_validation Validating Without Finite Differences

To verify the consistent Jacobian analytically:

1. Compute \f$a\f$ and \f$b\f$ at current pose
2. Compute \f$\partial b/\partial\omega_k\f$ analytically (see above)
3. Compute predicted difference:
   \f[
     \Delta_{\text{pred}}(k) = -\frac{a}{b^2}\frac{\partial b}{\partial \omega_k}
   \f]
4. Compute actual difference:
   \f[
     \Delta_{\text{act}}(k) = J_{\text{cons}}(\omega_k) - J_{\text{simp}}(\omega_k)
   \f]
5. Verify \f$\Delta_{\text{act}}(k) \approx \Delta_{\text{pred}}(k)\f$ for \f$k = 0,1,2\f$

@section rpj_fd_behavior Finite Difference Behavior Summary

| Component | Dependence | FD Behavior | Best \f$\varepsilon\f$ | Simplified OK? |
|-----------|------------|-------------|------------------------|----------------|
| Translation | Linear in \f$\mathbf{t}\f$ | Exact | Any | Yes |
| Rotation | Nonlinear in \f$R\f$ | Plateau | \f$10^{-6}\f$ to \f$10^{-8}\f$ | No |

@section rpj_summary Summary

- **Translation Jacobian:** Simplified = Consistent (denominator independent of \f$\mathbf{t}\f$)
- **Rotation Jacobian:** Simplified ≠ Consistent (must use full quotient rule)
- **The missing term** \f$-a \cdot (\partial b/\partial\boldsymbol{\omega}) / b^2\f$ can be large
- **FD validation:** Translation shows no plateau (linear); rotation shows plateau (nonlinear)
- **Always use consistent Jacobian** for rotation in production code
