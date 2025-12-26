# Verify simplified rotation Jacobian without FD
# Goal: check that (J_cons - J_simp) equals the missing quotient-rule term.

Residual:
  r = a / b

Definitions:
  a = n^T (x - q)
  b = n^T d

Forward case (source -> target):
  x = R*p + t
  d = R*d0

Backward case (target -> source):
  y = R^T (p - t)
  d = R^T*d0
  (replace x by y in 'a')

Rotation Jacobians:
  J_cons(w_k)  = dr/dw_k using quotient rule
  J_simp(w_k)  = dr/dw_k ignoring db/dw_k

Exact identity (rotation component k):
  J_cons(w_k) = J_simp(w_k) - a * (db/dw_k) / b^2

Equivalently:
  (J_cons - J_simp)(w_k) = - a * (db/dw_k) / b^2
  (J_simp - J_cons)(w_k) = + a * (db/dw_k) / b^2

So to validate "simplified" (without FD):
  1) Compute a and b at the current pose.
  2) Compute db/dw_k analytically.
  3) Compute predicted difference:
       Delta_pred(k) = - a * (db/dw_k) / (b*b)
  4) Compute actual difference:
       Delta_act(k)  = J_cons(w_k) - J_simp(w_k)
  5) Check Delta_act(k) ~= Delta_pred(k) for k=0,1,2.

------------------------------------------------------------
How to compute db/dw_k (right-multiplication, rotation only)
------------------------------------------------------------

We need db/dw where b = n^T d.

Case A: d = R*d0   (forward)
  Under right-multiplication: R_new = R*Exp(w^)
  First-order variation:
    d_new ~= d + (R*(w x d0)) = d + ( (R*w) x d )
  But the standard Jacobian form you already used is:
    dd/dw = - R * [d0]_x   (3x3 mapping w -> dd)
  Therefore:
    db/dw = n^T * dd/dw = n^T * ( - R*[d0]_x )

So per component k:
  db/dw_k = n^T * ( - R*(d0 x e_k) )
          = - n^T * ( R*(d0 x e_k) )

Case B: d = R^T*d0 (backward)
  Under right-multiplication: R_new^T = Exp(-w^)*R^T
  First-order variation:
    d_new ~= d - (w x d)
  Therefore:
    dd/dw = - [d]_x  (3x3 mapping w -> dd)
  Hence:
    db/dw = n^T * dd/dw = n^T * ( -[d]_x )

So per component k:
  db/dw_k = n^T * ( - (d x e_k) )
          = - n^T * ( d x e_k )

------------------------------------------------------------
Sanity checks (why simplified can be very wrong)
------------------------------------------------------------

Missing term magnitude:
  Missing(k) = - a * (db/dw_k) / b^2

This becomes large when:
- |a| is not small (not near convergence)
- |b| = |n^T d| is small (grazing) because 1/b^2 explodes
- |db/dw_k| is not small (direction sensitive to rotation)

That is exactly why simplified rotation can show huge FD disagreement on random geometry.
