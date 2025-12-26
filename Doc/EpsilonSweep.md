# Finite-Difference Jacobian Validation on SE(3)  
## Epsilon Sweeping and the Plateau Test (Right-Multiplication)

This document explains **why and how to sweep the finite-difference step size ε**
when validating analytic Jacobians on **SE(3)** using **right-multiplication
(moving-frame) perturbations**.

The goal is to reliably determine whether an analytic Jacobian is correct,
especially in the presence of rotations, translations, and nonlinear residuals
(e.g. ray-projection).

---

## 1. Why a single ε is not sufficient

Finite differences approximate a derivative as:

\[
\frac{\partial r}{\partial x}
\approx
\frac{r(x+\varepsilon) - r(x-\varepsilon)}{2\varepsilon}
\]

This approximation is affected by **two competing error sources**.

### 1.1 Truncation error (ε too large)

For large ε, higher-order terms in the Taylor expansion dominate:

\[
r(x+\varepsilon) = r(x) + \varepsilon r'(x) + O(\varepsilon^2)
\]

As a result:
- the FD estimate is biased
- nonlinear effects corrupt the derivative
- error scales like \(O(\varepsilon^2)\)

---

### 1.2 Floating-point error (ε too small)

For very small ε:
- subtraction causes catastrophic cancellation
- floating-point rounding dominates
- noise overwhelms the signal

This error scales roughly like:
\[
O\!\left(\frac{\epsilon_{\text{machine}}}{\varepsilon}\right)
\]

---

## 2. The plateau phenomenon

Because of these two opposing errors, plotting FD error vs ε typically yields:


error
  ^
  |\
  | \
  |  \      truncation-dominated
  |   \
  |    -------------------  ← plateau (sweet spot)
  |                     \
  |                      \  roundoff-dominated
  +-----------------------------> log10(ε)



- **Large ε** → truncation error dominates
- **Small ε** → floating-point error dominates
- **Intermediate ε** → errors balance → **plateau**

The plateau is the region where:
- the FD derivative is stable w.r.t. ε
- the approximation is trustworthy

---

## 3. Why the plateau matters

A correct analytic Jacobian must satisfy:

> **Its value lies on the finite-difference plateau.**

This gives a robust validation criterion:

- If no plateau exists → FD setup is wrong
- If plateau exists but does not match analytic Jacobian → analytic Jacobian is wrong
- If plateau exists and matches → Jacobian is correct

---

## 4. SE(3) right-multiplication recap (critical)

We use **right-multiplication perturbations**:

\[
T(\delta) = T \exp(\delta^\wedge), \quad
\delta =
\begin{bmatrix}
v \\
\omega
\end{bmatrix}
\]

with pose action:
\[
x = R p + t
\]

Under right-multiplication:
- \(R' = R \Delta R\)
- \(t' = t + R \Delta t\)

This directly affects how finite differences must be applied.

---

## 5. Correct finite-difference perturbations (right-multiplication)

Let \(r(T)\) be a scalar residual.

### 5.1 Translation components

To validate translation Jacobian columns \(v_x, v_y, v_z\):

For \(k \in \{0,1,2\}\):

\[
R_\pm = R
\]

\[
t_\pm = t + R(\pm \varepsilon e_k)
\]

\[
J^{\text{FD}}_k(\varepsilon)
\approx
\frac{r(R_+, t_+) - r(R_-, t_-)}{2\varepsilon}
\]

**Important:**  
Adding \(\varepsilon e_k\) directly to \(t\) is **incorrect** for right-multiplication.

---

### 5.2 Rotation components

To validate rotation Jacobian columns \(\omega_x, \omega_y, \omega_z\):

For \(k \in \{0,1,2\}\):

\[
R_\pm = R \exp((\pm \varepsilon e_k)^\wedge)
\]

\[
t_\pm = t
\]

\[
J^{\text{FD}}_{3+k}(\varepsilon)
\approx
\frac{r(R_+, t_+) - r(R_-, t_-)}{2\varepsilon}
\]

---

## 6. How to sweep ε in practice

### 6.1 Choose a logarithmic sweep

Typical range for double precision:

\[
\varepsilon \in [10^{-2}, 10^{-9}]
\]

Use logarithmic spacing, for example:

1e-2, 3e-3, 1e-3, 3e-4, 1e-4, ..., 1e-9

---

### 6.2 Compute FD Jacobians for each ε

For each ε and each parameter dimension:
1. Apply the correct SE(3) perturbation
2. Evaluate the residual
3. Compute the central difference

---

### 6.3 Compare against analytic Jacobian

Use an error metric such as:

- Absolute error:
  \[
  |J^{\text{FD}}(\varepsilon) - J^{\text{analytic}}|
  \]

- Relative error:
  \[
  \frac{|J^{\text{FD}}(\varepsilon) - J^{\text{analytic}}|}
       {\max(1, |J^{\text{analytic}}|)}
  \]

Plot or log this error vs \(\varepsilon\).

---

## 7. Interpreting the results

### Case A: No plateau
Likely causes:
- wrong perturbation rule (left vs right)
- incorrect translation update
- residual discontinuities or clipping

### Case B: Plateau exists but is offset
Likely causes:
- missing Jacobian terms
- sign errors
- wrong frame for cross products
- missing denominator derivative in ray-projection residuals

### Case C: Plateau matches analytic Jacobian
This is the desired outcome.
The Jacobian can be trusted.

---

## 8. Typical ε ranges that work well

For SE(3) with double precision:

| Component    | Typical plateau range |
|-------------|------------------------|
| Translation | \(10^{-6} \rightarrow 10^{-8}\) |
| Rotation    | \(10^{-6} \rightarrow 10^{-8}\) radians |

These are guidelines — always sweep.

---

## 9. Why sweeping is especially important for SE(3)

SE(3) Jacobian validation is more sensitive than Euclidean cases because:
- perturbations are applied via the exponential map
- translation depends on rotation (right-multiplication)
- residuals often include divisions (e.g. ray projection)
- normals and view directions amplify numerical noise

All of this makes single-ε checks unreliable.

---

## 10. Summary

- Finite differences involve a trade-off between truncation and roundoff errors
- Sweeping ε reveals a stable plateau region
- Correct analytic Jacobians lie on that plateau
- For right-multiplication on SE(3), translation FD **must** use \(t \leftarrow t + R(\varepsilon e_k)\)
- Sweeping ε is the most reliable way to validate SE(3) Jacobians

---

## One-line takeaway

> **Never trust a single ε; trust the plateau.**
