@page effective_conditioning Effective Conditioning via Manifold Scaling

This document explains how to change the effective conditioning of the
SE(3) optimization problem in Ceres without modifying cost Jacobians.

@section cond_short_answer Short Answer

If your cost functors already return **correct Jacobians with respect to the 7D ambient pose**

\f[
  [q_x, q_y, q_z, q_w, t_x, t_y, t_z]
\f]

then **you do not need to modify them**.

To change the effective conditioning (what feels like *preconditioning* in Ceres),
you only need to modify:

1. the SE(3) **Plus** operation, and
2. the **chart Jacobian / pullback**

Ceres always builds the linear system in the local tangent space using these.

@section cond_ceres_internal What Ceres Does Internally

Let

- \f$x \in \mathbb{R}^7\f$ be the ambient pose
- \f$\delta \in \mathbb{R}^6\f$ be the tangent increment
- \f$\text{Plus}(x,\delta)\f$ be your manifold update
- \f$P(x) = \left.\frac{\partial \text{Plus}(x,\delta)}{\partial \delta}\right|_{\delta=0}\f$
  be the chart Jacobian (7x6)

If your cost functor provides an ambient Jacobian

\f[
  J_{\text{ambient}} = \frac{\partial r}{\partial x} \in \mathbb{R}^{1 \times 7},
\f]

then Ceres forms the local Jacobian as

\f[
  J_{\text{local}} = J_{\text{ambient}} \, P(x).
\f]

All Gauss-Newton / LM solves happen in this 6D local space.

@section cond_key_rule The Key Rule

If you change the Plus operation so that it applies a *scaled tangent*,

\f[
  \delta' = S \, \delta,
\f]

then the chart Jacobian must change consistently:

\f[
  P_{\text{new}}(x) = P_{\text{old}}(x) \, S.
\f]

Ceres will then automatically build

\f[
  J_{\text{local}} = J_{\text{ambient}} \, P_{\text{new}}(x),
\f]

without any change to your cost Jacobians.

@section cond_example Concrete Example: Scaling Rotation vs Translation

Let the tangent be ordered as

\f[
  \delta =
  \begin{bmatrix}
    \mathbf{v} \\
    \boldsymbol{\omega}
  \end{bmatrix}
  \in \mathbb{R}^6,
\f]

with translation \f$\mathbf{v}\f$ and rotation \f$\boldsymbol{\omega}\f$.

Choose a characteristic length \f$L\f$ (same units as translation) and define

\f[
  S =
  \begin{bmatrix}
    I_3 & 0 \\
    0 & L I_3
  \end{bmatrix}.
\f]

This makes "1 radian of rotation" comparable to "\f$L\f$ meters of translation".

@section cond_modified_plus Modified Plus Operation

The manifold update becomes

\f[
  \boldsymbol{\omega}' = L \boldsymbol{\omega}, \quad
  \mathbf{v}' = \mathbf{v},
\f]

\f[
  R_{\text{new}} = R \exp(\boldsymbol{\omega}'),
\f]

\f[
  \mathbf{t}_{\text{new}} =
  \mathbf{t} + R \, V(\boldsymbol{\omega}') \, \mathbf{v}'.
\f]

This is a pure change of variables, not a change of the objective.

@section cond_modified_chart Modified Chart Jacobian

At \f$\delta = 0\f$, the SE(3) chart Jacobian has the structure

\f[
  P(x) =
  \begin{bmatrix}
    \frac{\partial \mathbf{q}}{\partial \boldsymbol{\omega}} & 0 \\
    0 & \frac{\partial \mathbf{t}}{\partial \mathbf{v}}
  \end{bmatrix}.
\f]

After scaling, it becomes

\f[
  P_{\text{new}}(x) =
  \begin{bmatrix}
    \frac{\partial \mathbf{q}}{\partial \boldsymbol{\omega}} \, L & 0 \\
    0 & R
  \end{bmatrix}.
\f]

Only the **rotation columns** are scaled.
Nothing else changes.

@section cond_unchanged What Remains Unchanged

- All cost functors
- All ambient (1x7) Jacobians
- The quotient-rule correctness of your residual derivatives
- Your Jacobian pullback code (`J7 * P(x)`)
- The linear solver (including cuDSS)

You are changing only the *metric* in which Ceres measures steps on SE(3).

@section cond_consistency Important Consistency Check

Ceres **must** be using your manifold implementation
(`Plus` and `PlusJacobian`) for the 7D pose blocks.

If any default quaternion parameterization is still attached,
the pullback will be inconsistent.

@section cond_summary Summary

- Yes, you can keep your Jacobians.
- Modify **Plus** and **PlusJacobian** only.
- This is the correct and supported way to influence conditioning
  when using direct solvers in Ceres.
- Conceptually, this is variable scaling, not a hack.

If you want, the next step is choosing a good value for \f$L\f$
based on your mesh scale or correspondence statistics.
