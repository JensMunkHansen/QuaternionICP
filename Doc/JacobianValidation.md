# Validating Left vs Right Jacobians on SO(3)

This note describes how to numerically validate Jacobians on SO(3) when using
left (space / fixed frame) and right (body / moving frame) perturbations.

All math is ASCII and assumes exp(hat(.)) parameterization.

---------------------------------------------------------------------

1. Problem Setup

Let R be a rotation matrix in SO(3).
Let f(R) be a vector-valued function.

Two common perturbation models:

Left perturbation (space frame):
  R(eps) = exp(eps * hat(delta)) * R

Right perturbation (body frame):
  R(eps) = R * exp(eps * hat(delta))

The Jacobians depend on which perturbation model is used.

---------------------------------------------------------------------

2. Finite-Difference Validation (Primary Test)

For a small eps and direction delta in R^3, define the central difference:

  df_num =
    ( f(R_plus) - f(R_minus) ) / (2 * eps)

Use central differences for accuracy.

---------------------------------------------------------------------

2.1 Left Jacobian Test

Perturbations:
  R_plus  = exp( eps * hat(delta) ) * R
  R_minus = exp(-eps * hat(delta) ) * R

Analytical prediction:
  df_ana = J_left(R) * delta

Validation:
  ||df_num - df_ana|| / max(1, ||df_num||) << 1

---------------------------------------------------------------------

2.2 Right Jacobian Test

Perturbations:
  R_plus  = R * exp( eps * hat(delta) )
  R_minus = R * exp(-eps * hat(delta) )

Analytical prediction:
  df_ana = J_right(R) * delta

---------------------------------------------------------------------

2.3 Practical Notes

- Use eps ~ 1e-6 to 1e-8 (double precision)
- Sweep eps over a decade and look for a plateau
- Test random R and random delta (unit and scaled)
- Compare directional derivatives, not full matrices

This catches sign errors, frame mixups, missing transposes, etc.

---------------------------------------------------------------------

3. Cross-Validation Between Left and Right Jacobians

Left and right perturbations represent the same physical motion if their
tangent vectors satisfy:

  delta_space = R * delta_body

This implies the Jacobians must satisfy:

  J_right(R) = J_left(R) * R

Equivalently:

  J_left(R) = J_right(R) * R^T

Numeric check:
  || J_right - J_left * R || ~= 0

If this fails but finite differences pass, your tangent convention differs
(e.g. reversed delta mapping, quaternion local parameterization, etc.).
An adjoint relationship still exists for the correct convention.

---------------------------------------------------------------------

4. Simple Sanity Test Function

Use a function with a known derivative:

  f(R) = R * p

where p is a fixed vector in R^3.

Left perturbation result:
  df = delta_space x (R * p)

Right perturbation result:
  df = R * (delta_body x p)

This test is extremely useful for debugging sign and frame errors.

---------------------------------------------------------------------

5. Validation Checklist

- Finite-difference test J_left
- Finite-difference test J_right
- Verify J_right = J_left * R
- Sanity check with f(R) = R * p

If all four pass, the Jacobians are correct.

---------------------------------------------------------------------
