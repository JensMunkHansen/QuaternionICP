@page so3_left_jacobian SO(3) Left Jacobian

The SO(3) left Jacobian \f$V(\boldsymbol{\omega})\f$ arises in the SE(3) exponential
map, coupling rotation and translation. This document derives its closed-form
expression and discusses numerical considerations.

@section so3lj_naming Naming Convention

The term "left Jacobian" refers to \f$V\f$ appearing on the **left** of
\f$\mathbf{v}\f$ in the translation formula \f$V\mathbf{v}\f$, not to
left-multiplication. This document uses the **right-multiplicative** (body-frame)
SE(3) convention consistent with the implementation in `SE3.h`.

@section so3lj_context Context: SE(3) Exponential Map

The SE(3) exponential map for a twist \f$\boldsymbol{\xi} = (\mathbf{v}, \boldsymbol{\omega})\f$:
\f[
  \exp(\widehat{\boldsymbol{\xi}}) =
  \begin{bmatrix}
    \exp([\boldsymbol{\omega}]_\times) & V(\boldsymbol{\omega})\,\mathbf{v} \\
    \mathbf{0}^\top & 1
  \end{bmatrix}
\f]

The translation component is \f$V(\boldsymbol{\omega})\,\mathbf{v}\f$, not simply
\f$\mathbf{v}\f$. The matrix \f$V\f$ accounts for the coupling between rotation
and translation during the exponential map.

@section so3lj_definition Definition

The SO(3) left Jacobian is:
\f[
  V(\boldsymbol{\omega}) = \sum_{k=0}^{\infty} \frac{1}{(k+1)!} [\boldsymbol{\omega}]_\times^k
\f]

This infinite series has a closed-form expression.

@section so3lj_closed_form Closed-Form Expression

Let \f$\theta = \|\boldsymbol{\omega}\|\f$ and define:
\f[
  [\boldsymbol{\omega}]_\times =
  \begin{bmatrix}
    0 & -\omega_z & \omega_y \\
    \omega_z & 0 & -\omega_x \\
    -\omega_y & \omega_x & 0
  \end{bmatrix}
\f]

Then:
\f[
  V(\boldsymbol{\omega}) = I
    + \frac{1 - \cos\theta}{\theta^2} [\boldsymbol{\omega}]_\times
    + \frac{\theta - \sin\theta}{\theta^3} [\boldsymbol{\omega}]_\times^2
\f]

Using the scalar coefficients:
\f[
  B = \frac{1 - \cos\theta}{\theta^2}, \qquad
  C = \frac{\theta - \sin\theta}{\theta^3} = \frac{1 - \sinc(\theta)}{\theta^2}
\f]

we write:
\f[
  V = I + B\,[\boldsymbol{\omega}]_\times + C\,[\boldsymbol{\omega}]_\times^2
\f]

@section so3lj_special_cases Special Cases

@subsection so3lj_identity At \f$\boldsymbol{\omega} = 0\f$

\f[
  V(\mathbf{0}) = I
\f]

The translation passes through unchanged.

@subsection so3lj_small_angle Small Angle Approximation

For \f$\|\boldsymbol{\omega}\| \ll 1\f$:
\f[
  V(\boldsymbol{\omega}) \approx I + \frac{1}{2}[\boldsymbol{\omega}]_\times
\f]

This first-order approximation is used when \f$\theta < 10^{-12}\f$ to avoid
numerical instability in the trigonometric coefficients.

@section so3lj_derivation Derivation

@subsection so3lj_series From the Matrix Exponential

The SE(3) exponential can be written as:
\f[
  \exp\left(\begin{bmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^\top & 0 \end{bmatrix}\right)
  = \sum_{k=0}^{\infty} \frac{1}{k!}
    \begin{bmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^\top & 0 \end{bmatrix}^k
\f]

The translation block extracts as:
\f[
  V\,\mathbf{v} = \sum_{k=0}^{\infty} \frac{1}{(k+1)!} [\boldsymbol{\omega}]_\times^k \mathbf{v}
\f]

@subsection so3lj_rodrigues Using Rodrigues' Formula

From the SO(3) exponential (Rodrigues):
\f[
  \exp([\boldsymbol{\omega}]_\times) = I + \frac{\sin\theta}{\theta}[\boldsymbol{\omega}]_\times
    + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times^2
\f]

The left Jacobian can be derived by integrating:
\f[
  V = \int_0^1 \exp(s\,[\boldsymbol{\omega}]_\times)\,ds
\f]

Substituting Rodrigues' formula and integrating term by term yields the
closed-form expression.

@section so3lj_numerical Numerical Implementation

@snippet ICP/SE3.h Vso3

The small-angle check avoids division by near-zero \f$\theta^2\f$.

@section so3lj_inverse Inverse (Right Jacobian)

The inverse of the left Jacobian is:
\f[
  V^{-1}(\boldsymbol{\omega}) = I
    - \frac{1}{2}[\boldsymbol{\omega}]_\times
    + \left(\frac{1}{\theta^2} - \frac{1 + \cos\theta}{2\theta\sin\theta}\right)
      [\boldsymbol{\omega}]_\times^2
\f]

This is used when converting from SE(3) to the Lie algebra (logarithm map).

@section so3lj_related Related Documents

- @ref se3_full_chart_jacobian — Uses \f$V(\boldsymbol{\omega})\f$ in SE(3) Plus Jacobian
- `ICP/SE3.h` — Implementation of `Vso3()` and `se3Plus()`

@section so3lj_references References

- Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge University Press.
- Solà, J. et al. (2018). *A micro Lie theory for state estimation in robotics*.
