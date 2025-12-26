# Ray-Projection Residual with Incidence Weighting w(c)
## Correct Jacobians (including dw/dc) for SE(3) right-multiplication

This document gives the correct analytic Jacobians for the ray-projection
point-to-plane residual with incidence-based weighting, matching your
GeometryWeighting struct.

It covers:
- residual definition
- SE(3) right-multiplication Jacobians (local increments)
- how to include weight w(c) and its derivative dw/dc
- explicit formulas for your supported weight modes:
    Abs      : w = |c|
    SqrtAbs  : w = sqrt(|c|)
with clamping and gating (tau)

This document does NOT discuss any simplified Jacobians.

-------------------------------------------------------------------------------
1) Definitions
-------------------------------------------------------------------------------

We define a scalar residual r(T) of the ray-projection form:

  r = w(c) * ( a / c )

where:
  a = n^T ( x - q )                (meters)
  c = n^T d                        (unitless)
  n is the (fixed) surface normal in the evaluation frame
  q is the (fixed) hit point in the evaluation frame

Forward case (source -> target frame evaluation):
  x = R * p + t
  d = R * d0

Here:
  p is source point in source-local coordinates
  d0 is the canonical ray direction in source-local (e.g. (0,0,-1))
  R,t are the pose mapping source -> target
  x,d are expressed in target frame

Note: you may also use the same formulas for backward residuals by changing
the expressions for x and d (see Section 6).

We will derive dr/ddelta for SE(3) right-multiplication increments delta.

-------------------------------------------------------------------------------
2) GeometryWeighting: gating, clamping, and modes
-------------------------------------------------------------------------------

Let tau in (0,1] be the gating/clamping parameter.

Given c = n^T d, define:

  ac_raw = abs(c)

Gating:
  if enable_gate and ac_raw < tau:
      w = 0
      and we treat the correspondence as rejected

Weight enable:
  if not enable_weight:
      w = 1

Otherwise clamp:
  ac = clamp(ac_raw, tau, 1)
  (i.e. ac = max(tau, min(1, ac_raw)))

Then weight modes:
  Abs     : w = ac
  SqrtAbs : w = sqrt(ac)

Important: If gating is enabled and ac_raw < tau, you typically want to skip
the residual entirely (preferred), or set residual and Jacobians to zero.
In that gated region, w is effectively constant zero and dw/dc is treated as 0.

-------------------------------------------------------------------------------
3) SE(3) right-multiplication perturbation and basic Jacobians
-------------------------------------------------------------------------------

Right-multiplication update:
  T_new = T * Exp(delta)
with delta = [ v ; w ] in R^6
  v: translation increment in local/body frame
  w: rotation increment in local/body frame

Forward model:
  x = R*p + t
  d = R*d0

Under right-multiplication, first-order variations give:

  dx/dv = R
  dx/dw = - R * [p]_x

  dd/dv = 0
  dd/dw = - R * [d0]_x

Here [u]_x is the 3x3 skew matrix such that [u]_x * y = u cross y.

Define 1x6 row Jacobians for a and c:

  a = n^T (x - q)
  c = n^T d

So:

  da/ddelta = n^T * dx/ddelta
            = [ n^T R ,  - n^T R [p]_x ]          (1x6)

  dc/ddelta = n^T * dd/ddelta
            = [ 0     ,  - n^T R [d0]_x ]         (1x6)

Key property:
  dc/dv = 0, so translation affects only a, not c.

-------------------------------------------------------------------------------
4) Full derivative of r = w(c) * a/c
-------------------------------------------------------------------------------

Residual:
  r = w(c) * (a/c)

Differentiate with respect to any parameter theta (component of delta):

  dr/dtheta = d/dtheta [ w(c) * a/c ]

Use product + quotient rules. Let:
  w  = w(c)
  wp = dw/dc

Then:

  dr/dtheta = (w/c) * da/dtheta
              + a * d/dtheta (w/c)

But w/c depends on theta only through c:

  d/dtheta (w/c) = (d/dc)(w/c) * dc/dtheta

Compute derivative with respect to c:

  d/dc (w/c) = (wp * c - w) / c^2

Therefore the correct Jacobian is:

  dr/dtheta = (w/c) * da/dtheta
              + a * ( (wp*c - w) / c^2 ) * dc/dtheta

In vector form (1x6):

  dr/ddelta = (w/c) * da/ddelta
              + a * ( (wp*c - w) / c^2 ) * dc/ddelta

Because dc/dv = 0, the second term contributes only to rotation columns.

-------------------------------------------------------------------------------
5) Computing w(c) and wp = dw/dc for your modes
-------------------------------------------------------------------------------

We must handle:
- abs(c)
- clamp to [tau, 1]
- gating at abs(c) < tau (optional)
- mode Abs or SqrtAbs
- enable_weight flag

Let s = sign(c), with sign(0) treated as 0 (but gated/clamped avoids exact 0).
For c != 0:
  d/dc abs(c) = sign(c) = s

Case A: enable_gate and abs(c) < tau
-----------------------------------
Interpretation: correspondence rejected.

Recommended behavior:
- skip adding residual block, OR
- set r = 0 and dr/ddelta = 0

If you do the latter:
  w = 0
  wp = 0
  r = 0
  dr/ddelta = 0

Case B: not enable_weight
-------------------------
Then weight is constant:
  w = 1
  wp = 0

So Jacobian reduces to:
  dr/ddelta = (1/c) * da/ddelta  +  a * ( (0*c - 1)/c^2 ) * dc/ddelta
            = (1/c) * da/ddelta  -  (a/c^2) * dc/ddelta

This is the "no incidence weighting" but still fully consistent derivative
(including the denominator derivative of a/c).

Case C: enable_weight and not gated
-----------------------------------
Define:
  ac_raw = abs(c)
  ac = clamp(ac_raw, tau, 1)

Clamping effect on wp:
- If ac_raw is strictly inside (tau, 1), then ac = abs(c) and derivative passes through.
- If clamped at tau or 1, then ac is constant and derivative is 0.

So:
  if ac_raw <= tau:  (this is gated out if enable_gate, otherwise clamped)
      ac = tau,  d(ac)/dc = 0
  else if ac_raw >= 1:
      ac = 1,    d(ac)/dc = 0
  else:
      ac = abs(c), d(ac)/dc = sign(c) = s

Now per mode:

Mode Abs:
  w = ac
  dw/dc = wp = d(ac)/dc

So:
  wp = 0                      when clamped
  wp = s                      when unclamped (tau < abs(c) < 1)

Mode SqrtAbs:
  w = sqrt(ac)
  dw/dc = wp = (1/(2*sqrt(ac))) * d(ac)/dc

So:
  wp = 0                                          when clamped
  wp = s / (2*sqrt(abs(c)))                       when unclamped

Note: When unclamped, ac = abs(c). When clamped at tau, use wp=0 (piecewise constant).
This matches your implementation semantics (clamp after gating).

-------------------------------------------------------------------------------
6) Backward residual (target -> source) under right-multiplication
-------------------------------------------------------------------------------

If you also evaluate a backward residual in the source frame, you typically use:

  x = R^T * (pT - t)          (point expressed in source)
  d = R^T * d0T               (ray direction expressed in source)

Then you still use the same weighted form:

  r = w(c) * a/c
  a = n^T (x - q)
  c = n^T d

But the SE(3) Jacobians for x and d change.

Under right-multiplication, R^T updates as:
  R_new^T = Exp(-w^) * R^T

First-order variations:
  d(R^T v)/dw = - [R^T v]_x     (3x3 mapping w -> delta)
  d(R^T v)/dv = 0              (for v being the rotation increment, not translation)

For x = R^T*(pT - t):
  Let u = (pT - t)
  x = R^T * u

Then:
  dx/dv = - R^T                 (because u depends on t and dt = R*v for right-mult;
                                in practice derive carefully from your chosen parameterization)
  dx/dw = - [x]_x

For d = R^T*d0T:
  dd/dv = 0
  dd/dw = - [d]_x

Practical recommendation:
- If you already have validated formulas from your Python reference, use those.
- Then reuse Sections 4 and 5 unchanged (only da/ddelta and dc/ddelta change).

The key message:
- The weighted derivative structure in Section 4 is universal.
- Only da/ddelta and dc/ddelta depend on whether you are forward or backward.

-------------------------------------------------------------------------------
7) Final Jacobian formula to implement (forward, right-multiplication)
-------------------------------------------------------------------------------

Given:
  a, c, w, wp, da/ddelta (1x6), dc/ddelta (1x6)

Compute:

  term1 = (w/c) * da/ddelta
  term2 = a * ((wp*c - w) / (c*c)) * dc/ddelta

  J = term1 + term2

Gating:
- If enable_gate and abs(c) < tau:
    set r = 0, J = 0 (or skip residual)
- Else compute with clamping rules for w and wp as above.

Translation columns:
- dc/dv = 0 => term2 does not affect translation
- so translation Jacobian is (w/c) * n^T R

Rotation columns:
- include both term1 and term2
- term2 captures the effect of changing incidence (c) on both the division and the weight

-------------------------------------------------------------------------------
8) Notes on nondifferentiabilities
-------------------------------------------------------------------------------

Your weighting uses abs(c) and clamp. Therefore:
- w(c) is not differentiable at c = 0
- d(abs(c))/dc has a sign discontinuity at c = 0
- clamp introduces kinks at abs(c) = tau and abs(c) = 1

Your implementation effectively defines a piecewise function. The Jacobian above
is therefore piecewise:
- use wp = 0 when clamped (piecewise constant region)
- use wp = sign(c) (Abs) or sign(c)/(2*sqrt(abs(c))) (SqrtAbs) when unclamped
- if gated, set residual/J to 0 (or omit residual)

This is standard practice and is consistent with robust gating behavior.

-------------------------------------------------------------------------------
9) Summary
-------------------------------------------------------------------------------

Residual:
  r = w(c) * a/c

with:
  a = n^T (R*p + t - q)
  c = n^T (R*d0)

Right-multiplication Jacobians:
  da/ddelta = [ n^T R , -n^T R [p]_x ]
  dc/ddelta = [ 0     , -n^T R [d0]_x ]

Full (fully consistent) derivative including dw/dc:
  dr/ddelta = (w/c) * da/ddelta + a * ((wp*c - w)/c^2) * dc/ddelta

Weight and derivative (unclamped region tau < abs(c) < 1):
  Abs     : w = abs(c),      wp = sign(c)
  SqrtAbs : w = sqrt(abs(c)),wp = sign(c)/(2*sqrt(abs(c)))

Clamped region (abs(c) <= tau or abs(c) >= 1):
  wp = 0  (piecewise constant), w = tau-or-1 transformed by mode

Gated region (enable_gate and abs(c) < tau):
  r = 0, J = 0 (or skip residual entirely)
