# Finite-Difference Jacobian Validation on SE(3)  
## Right-Multiplication Perturbations Only

This note explains **how to correctly validate Jacobians using finite differences**
when working on **SE(3)** with **right-multiplication (moving-frame) updates**.

It focuses only on the **right-multiplication convention**, which is what is used by:
- Sophus (`T = T * Exp(δ)`)
- Ceres manifolds for `SE3`
- Most modern Lie-group–based solvers

The key point is that **translation perturbations live in the local (body) frame**, not the world frame.

---

## 1. SE(3) setup

A pose \(T \in \mathrm{SE}(3)\) acts on a point \(p\) as:

\[
x = R p + t
\]

with:
- \(R \in \mathrm{SO}(3)\)
- \(t \in \mathbb{R}^3\)

We use a **right-multiplication perturbation**:

\[
T(\delta) = T \exp(\delta^\wedge)
\]

where the Lie algebra perturbation is:

\[
\delta =
\begin{bmatrix}
v \\
\omega
\end{bmatrix}
\in \mathbb{R}^6
\]

---

## 2. Right-multiplication update rule

Write the SE(3) exponential as:

\[
\exp(\delta^\wedge) =
\begin{bmatrix}
\Delta R & \Delta t \\
0 & 1
\end{bmatrix}
\]

Then:

\[
T(\delta) = T \exp(\delta^\wedge)
=
\begin{bmatrix}
R \Delta R & t + R \Delta t \\
0 & 1
\end{bmatrix}
\]

So under **right-multiplication**:

- \(R' = R \Delta R\)
- \(t' = t + R \Delta t\)

This is the rule that **must** be respected when validating Jacobians.

---

## 3. Translation perturbations (the common pitfall)

To validate **translation Jacobian columns**, we apply a **pure translation perturbation**:

\[
\omega = 0
\]

For \(\omega = 0\):

- \(\Delta R = I\)
- \(\Delta t = v\)

Therefore:

\[
t' = t + R v
\]

**This is the crucial point.**

### ❌ Incorrect (world-frame translation)

\[
t' = t + \varepsilon e_k
\]

This corresponds to **left-multiplication**, not right-multiplication.

### ✅ Correct (right-multiplication translation)

\[
t' = t + R(\varepsilon e_k)
\]

The translation increment \(v\) lives in the **local/body frame** and must be rotated by \(R\).

---

## 4. Finite-difference Jacobian validation (right-multiplication)

Let \(r(T)\) be a scalar residual (e.g. a ray-projection residual).

We approximate derivatives using central differences:

\[
\frac{\partial r}{\partial \delta_i}
\approx
\frac{r(T(\,+\varepsilon e_i\,)) - r(T(\,-\varepsilon e_i\,))}{2\varepsilon}
\]

---

### 4.1 Translation components \(v_x, v_y, v_z\)

For \(k \in \{0,1,2\}\):

\[
R_\pm = R
\]

\[
t_\pm = t + R(\pm \varepsilon e_k)
\]

\[
J_{\text{FD}}[k]
\approx
\frac{r(R_+, t_+) - r(R_-, t_-)}{2\varepsilon}
\]

This validates the **translation columns** of a right-multiplication Jacobian.

---

### 4.2 Rotation components \(\omega_x, \omega_y, \omega_z\)

For \(k \in \{0,1,2\}\):

\[
R_\pm = R \exp((\pm \varepsilon e_k)^\wedge)
\]

\[
t_\pm = t
\]

\[
J_{\text{FD}}[3+k]
\approx
\frac{r(R_+, t_+) - r(R_-, t_-)}{2\varepsilon}
\]

This matches the standard SO(3) finite-difference check.

---

## 5. Why translation FD often “fails”

A very common mistake is to:
- use `Exp()` correctly for rotations, but
- perturb translation by adding directly to `t`

That mixes **right-multiplication rotation** with **left-multiplication translation**.

The result:
- rotation Jacobians validate correctly
- translation Jacobians do not

This is not a bug in the math — it is a **mismatch in perturbation conventions**.

---

## 6. Summary (right-multiplication only)

- Perturbations are applied as:  
  \(T \leftarrow T \exp(\delta^\wedge)\)
- Translation increments live in the **local/body frame**
- Finite-difference translation must use:  
  \(t \leftarrow t + R(\varepsilon e_k)\)
- Finite-difference rotation uses:  
  \(R \leftarrow R \exp((\varepsilon e_k)^\wedge)\)

---

## One-line takeaway

> **For right-multiplication on SE(3), finite-difference translation must be rotated by the current \(R\).**

This rule alone resolves most SE(3) Jacobian validation issues.
