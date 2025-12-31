@page se3_full_chart_jacobian SE(3) Full Chart Jacobian (Right-Multiplicative)

This document describes the full 7x6 chart Jacobian of the SE(3) Plus operation
away from the linearization point \f$\delta = 0\f$. It is provided strictly for
reference and mathematical completeness.

For Gauss-Newton optimization, the Jacobian is always evaluated at
\f$\delta = 0\f$, which yields the simplified form implemented in `SE3.h`.
The purpose of this document is to explain which terms vanish at the expansion
point and why.

This formulation uses the standard matrix Lie algebra parametrization of
\f$\mathfrak{so}(3)\f$, obtained by differentiating the orthogonality constraint
\f$R^\top R = I\f$, which yields skew-symmetric generators and is equivalent to
the axis-angle exponential map used in the implementation.

@section se3_full_conventions Conventions

Pose:
\f[
  x = (q, \mathbf{t}), \qquad q \in S^3,\ \mathbf{t} \in \mathbb{R}^3
\f]

Tangent increment (Lie algebra):
\f[
  \delta = (\mathbf{v}, \boldsymbol{\omega}) \in \mathbb{R}^6
\f]
where \f$\mathbf{v}\f$ is translation and \f$\boldsymbol{\omega}\f$ is an
axis-angle rotation vector (radians).

Right-multiplicative (body-frame) update:
\f[
  T_{\text{new}} = T \cdot \exp(\widehat{\delta})
\f]

Component form:
\f[
  \mathbf{q}_{\text{new}} = \mathbf{q} \otimes \exp(\boldsymbol{\omega}), \qquad
  \mathbf{t}_{\text{new}} = \mathbf{t} + R(\mathbf{q}) \, V(\boldsymbol{\omega}) \, \mathbf{v}
\f]

where \f$V(\boldsymbol{\omega})\f$ is the SO(3) left Jacobian (see @ref so3_left_jacobian).

@section se3_full_definition Definition of the Chart Jacobian

Define the Plus chart:
\f[
  x^{+}(\delta) = \mathrm{Plus}(x, \delta)
\f]

The chart Jacobian is:
\f[
  P(x,\delta) =
  \frac{\partial\,\mathrm{Plus}(x,\delta)}{\partial \delta}
  \in \mathbb{R}^{7\times6}
\f]

Block structure:
\f[
  P(x,\delta) =
  \begin{bmatrix}
    \frac{\partial \mathbf{q}_{\text{new}}}{\partial \mathbf{v}} &
    \frac{\partial \mathbf{q}_{\text{new}}}{\partial \boldsymbol{\omega}} \\
    \frac{\partial \mathbf{t}_{\text{new}}}{\partial \mathbf{v}} &
    \frac{\partial \mathbf{t}_{\text{new}}}{\partial \boldsymbol{\omega}}
  \end{bmatrix}
\f]



@section se3_full_zero_blocks Blocks That Are Always Zero

Quaternion with respect to translation:
\f[
  \frac{\partial \mathbf{q}_{\text{new}}}{\partial \mathbf{v}} = 0
\f]

Reason: the quaternion update depends only on
\f$\boldsymbol{\omega}\f$, not on \f$\mathbf{v}\f$.

@section se3_full_dt_dv Translation w.r.t. Translation

From:
\f[
  \mathbf{t}_{\text{new}} = \mathbf{t} + R(\mathbf{q})\,V(\boldsymbol{\omega})\,\mathbf{v}
\f]

we obtain:
\f[
  \frac{\partial \mathbf{t}_{\text{new}}}{\partial \mathbf{v}}
  = R(\mathbf{q})\,V(\boldsymbol{\omega})
\f]

At the expansion point \f$\boldsymbol{\omega}=0\f$:
\f[
  V(0) = I \quad \Rightarrow \quad
  \frac{\partial \mathbf{t}_{\text{new}}}{\partial \mathbf{v}} = R(\mathbf{q})
\f]

@section se3_full_dq_dw Quaternion w.r.t. Rotation

Quaternion update:
\f[
  \mathbf{q}_{\text{new}} = \mathbf{q} \otimes \delta\mathbf{q}, \qquad
  \delta\mathbf{q} = \exp(\boldsymbol{\omega})
\f]

Let \f$\mathbf{q} = [\mathbf{v}_q, s]\f$ and
\f$\delta\mathbf{q} = [\mathbf{v}_\delta, s_\delta]\f$.

Quaternion multiplication yields:
\f[
  \mathbf{q}_{\text{new}} =
  \begin{bmatrix}
    s\,\mathbf{v}_\delta + s_\delta\,\mathbf{v}_q + \mathbf{v}_q \times \mathbf{v}_\delta \\
    s\,s_\delta - \mathbf{v}_q^\top \mathbf{v}_\delta
  \end{bmatrix}
\f]

The Jacobian factorizes as:
\f[
  \frac{\partial \mathbf{q}_{\text{new}}}{\partial \boldsymbol{\omega}} =
  \frac{\partial \mathbf{q}_{\text{new}}}{\partial \delta\mathbf{q}}
  \frac{\partial \delta\mathbf{q}}{\partial \boldsymbol{\omega}}
\f]

At \f$\boldsymbol{\omega}=0\f$:
\f[
  \frac{\partial \mathbf{v}_q}{\partial \boldsymbol{\omega}}
  = \frac{1}{2}\left( s I + [\mathbf{v}_q]_\times \right), \qquad
  \frac{\partial s}{\partial \boldsymbol{\omega}}
  = -\frac{1}{2}\mathbf{v}_q^\top
\f]

@section se3_full_dt_dw Translation w.r.t. Rotation (Away from delta = 0)

From:
\f[
  \mathbf{t}_{\text{new}} = \mathbf{t} + R(\mathbf{q})\,V(\boldsymbol{\omega})\,\mathbf{v}
\f]

we obtain:
\f[
  \frac{\partial \mathbf{t}_{\text{new}}}{\partial \boldsymbol{\omega}}
  = R(\mathbf{q}) \frac{\partial V(\boldsymbol{\omega})}{\partial \boldsymbol{\omega}} \mathbf{v}
\f]

This block is proportional to \f$\mathbf{v}\f$ and is therefore nonzero
only away from the linearization point.

At \f$\delta=0\f$ (i.e. \f$\mathbf{v}=0\f$):
\f[
  \frac{\partial \mathbf{t}_{\text{new}}}{\partial \boldsymbol{\omega}} = 0
\f]

@section se3_full_final_form Final Forms

Full chart Jacobian:
\f[
  P(x,\delta) =
  \begin{bmatrix}
    0 &
    \frac{\partial (\mathbf{q}\otimes\exp(\boldsymbol{\omega}))}{\partial \boldsymbol{\omega}} \\
    R(\mathbf{q})V(\boldsymbol{\omega}) &
    R(\mathbf{q})\frac{\partial V(\boldsymbol{\omega})}{\partial \boldsymbol{\omega}}\mathbf{v}
  \end{bmatrix}
\f]

At the Gauss-Newton expansion point \f$\delta=0\f$:
\f[
  P(x,0) =
  \begin{bmatrix}
    0 & \frac{1}{2}(sI + [\mathbf{v}_q]_\times) \\
    R(\mathbf{q}) & 0
  \end{bmatrix}
\f]
This is the standard SE(3) chart Jacobian used in ICP and Gauss-Newton solvers.

@section se3_full_related Related Documents

- @ref so3_left_jacobian — Derivation and closed-form expression for \f$V(\boldsymbol{\omega})\f$
- `ICP/SE3.h` — Implementation of `se3Plus()` and `plusJacobian7x6()`
