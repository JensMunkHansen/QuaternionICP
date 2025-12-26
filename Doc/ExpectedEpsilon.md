# Interpretation of Finite-Difference Results for Translation vs Rotation
## Ray-Projection Residual on SE(3), Right-Multiplication

This document explains why finite-difference Jacobian validation behaves
very differently for translation and rotation in the ray-projection
residual, and why the observed results are correct and expected.

All discussion assumes:
- SE(3) poses
- right-multiplication updates: T_new = T * Exp(delta)
- central finite differences
- a scalar residual of the form r = a / b

-------------------------------------------------------------------------------
1) Residual definition
-------------------------------------------------------------------------------

The ray-projection residual is:

  r = a / b

with:

  a = n^T ( R * p + t - q )
  b = n^T ( R * d )

Here:
- R is rotation
- t is translation
- p is a source point
- q is a hit point
- n is a surface normal
- d is a ray direction (local, transformed by R)

-------------------------------------------------------------------------------
2) Translation dependence
-------------------------------------------------------------------------------

Examine how the residual depends on translation t.

The numerator:
  a(t) = n^T ( R * p + t - q )

This is affine linear in t.

The denominator:
  b(t) = n^T ( R * d )

This does NOT depend on t.

Therefore:

  dr/dt = (1 / b) * da/dt
        = n^T / b

Key properties:
- dr/dt is constant
- dr/dt does not depend on t
- there are no higher-order terms in t

-------------------------------------------------------------------------------
3) Consequence for finite differences (translation)
-------------------------------------------------------------------------------

Finite differences are exact for linear functions.

For a linear function f(x):

  ( f(x + eps) - f(x - eps) ) / (2 * eps) = f'(x)

independent of eps (up to floating-point roundoff).

Therefore, for translation:
- central finite differences recover dr/dt exactly
- there is no truncation error
- large eps values (e.g. 1e-2 or 3e-3) still work
- relative error reaches machine precision (~1e-15)

This explains the observations:
- J_cons(t) == J_simp(t)
- FD matches both to machine precision
- "best eps" appears large
- no visible plateau in the usual sense

This behavior is correct and expected.

-------------------------------------------------------------------------------
4) Rotation dependence
-------------------------------------------------------------------------------

Now examine rotation.

Both numerator and denominator depend on R:

  a(R) = n^T ( R * p + t - q )
  b(R) = n^T ( R * d )

Thus:
- r(R) = a(R) / b(R) is nonlinear in rotation
- higher-order terms are present

Applying the quotient rule gives:

  dr/dw = ( b * da/dw - a * db/dw ) / b^2

If the db/dw term is omitted, the Jacobian is incomplete.

-------------------------------------------------------------------------------
5) Consequence for finite differences (rotation)
-------------------------------------------------------------------------------

Because r is nonlinear in rotation:
- truncation error dominates for large eps
- floating-point error dominates for very small eps
- a genuine plateau appears for intermediate eps

Empirically:
- best eps is around 1e-6 to 1e-8
- consistent Jacobian matches FD at ~1e-12
- simplified Jacobian (ignoring db/dw) shows ~1e-2 error

This is exactly what theory predicts.

-------------------------------------------------------------------------------
6) Why translation and rotation behave differently
-------------------------------------------------------------------------------

Summary of differences:

Translation:
- residual is linear in t
- quotient rule reduces to simple form
- FD is exact for any reasonable eps
- no truncation error
- large eps acceptable

Rotation:
- residual is nonlinear in R
- quotient rule is essential
- FD requires eps sweep
- plateau appears at small eps
- simplified Jacobian is incorrect

Seeing different "best eps" values for translation and rotation is normal
and expected.

-------------------------------------------------------------------------------
7) Answer to the key question
-------------------------------------------------------------------------------

Question:
"Should the translation Jacobian show a plateau around 1e-6?"

Answer:
No.

Because the residual is exactly linear in translation, the finite-difference
approximation is exact. The usual truncation-vs-roundoff tradeoff does not
apply. The plateau degenerates into a flat line across many eps values.

-------------------------------------------------------------------------------
8) Final conclusion
-------------------------------------------------------------------------------

- The translation Jacobian behavior is correct.
- The rotation Jacobian behavior confirms the need for the quotient rule.
- The finite-difference validation is internally consistent.
- There is no bug or missing term for translation.

This is a textbook example of how linear and nonlinear components of an
SE(3) residual behave differently under finite-difference validation.
