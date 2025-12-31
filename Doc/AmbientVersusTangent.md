@page se3_ambient_vs_tangent SE(3) Optimization: 7D Ambient vs 6D Tangent Parameterization

This document summarizes the practical advantages of using a **7D ambient
parameterization** (unit quaternion + translation) with a manifold constraint,
compared to optimizing directly in the **6D tangent space** of SE(3).

The context is Gauss–Newton / Levenberg–Marquardt optimization as used in ICP,
bundle adjustment, and nonlinear least squares solvers such as Ceres.

-----------------------------------------------------------------------

@section se3_background Background

An SE(3) pose can be represented in two common ways:

1. **6D tangent representation**
   - State lives in R^6 (v, w)
   - Updates applied via Exp/Log maps
   - No redundancy, minimal coordinates

2. **7D ambient representation**
   - State lives in R^7 (q, t), with q a unit quaternion
   - Optimization performed in R^7
   - A manifold (local parameterization) enforces the unit-norm constraint

Modern solvers (including Ceres) support the second approach natively.

-----------------------------------------------------------------------

@section se3_advantages_7d Advantages of the 7D Ambient Representation

## 1. Trigonometry-free state storage

The 7D state itself contains:
- quaternion components
- translation

There are **no trigonometric functions** involved in storing or evaluating the
state. Trigonometry appears only in:
- the Plus operation (local update)
- Jacobians at the linearization point

In contrast, a pure tangent-space formulation often requires repeated
Exp/Log evaluations to move between representations.

This makes the 7D approach:
- cheaper for residual evaluation
- simpler for debugging
- easier to reason about numerically

-----------------------------------------------------------------------

## 2. Clean separation of state and update

With the ambient approach:
- The **state** is always a valid pose
- The **update** lives in the tangent space
- The manifold defines how updates are applied

This separation has several benefits:
- Residuals are written directly in terms of (R, t)
- No ambiguity about which frame the state lives in
- The linearization point is explicit and stable

In contrast, tangent-only formulations often blur the distinction between:
- current estimate
- incremental update
- retraction back to the manifold

-----------------------------------------------------------------------

## 3. Better solver interoperability

Using a 7D ambient state aligns naturally with modern solvers:

- Ceres, g2o, GTSAM, Sophus-style APIs all support this model
- Automatic differentiation works out of the box
- Local parameterizations handle the pullback automatically

This reduces:
- custom solver glue code
- hand-written Log/Exp chains
- risk of subtle frame or sign errors

-----------------------------------------------------------------------

## 4. Numerically stable linearization

In the 7D approach:
- Linearization is always performed at delta = 0
- Jacobians are first-order accurate by construction
- Second-order coupling terms vanish naturally

This matches the mathematical assumptions of Gauss–Newton exactly.

In contrast, optimizing directly in the tangent space can:
- accidentally include higher-order terms
- mix linearization and retraction logic
- make it harder to reason about which Jacobian is being computed

-----------------------------------------------------------------------

## 5. Quaternion gauge freedom is harmless

The 7D representation has a known redundancy:
- q and -q represent the same rotation

This is not a disadvantage in practice:
- The manifold handles normalization
- The solver sees a smooth local chart
- Residuals depend only on R(q), not the sign of q

This redundancy is well-understood and does not degrade convergence.

-----------------------------------------------------------------------

## 6. Easier residual and Jacobian implementation

Most geometric residuals naturally use:
- R(q)
- t

Examples:
- point-to-plane ICP
- ray projection errors
- reprojection residuals

With a 7D state:
- Residuals are written directly in closed form
- Jacobians with respect to q and t are explicit
- The pullback to R^6 is handled once, centrally

This leads to:
- simpler code
- fewer mistakes
- easier maintenance

-----------------------------------------------------------------------

## 7. Conceptual alignment with Lie theory

Although the state is 7D, the optimization is still Lie-theoretic:

- Updates live in se(3)
- Retraction uses Exp
- Linearization occurs in the tangent space

The ambient representation is simply a **coordinate chart** on the manifold,
not a rejection of Lie theory.

This makes the approach mathematically sound and widely accepted.

-----------------------------------------------------------------------

@section se3_when_tangent When a Pure 6D Tangent Formulation Makes Sense

A pure tangent-space approach may still be useful when:
- implementing minimal solvers by hand
- deriving closed-form updates
- working in very constrained embedded environments
- writing symbolic derivations or proofs

For most practical optimization pipelines, however, the 7D ambient approach
with a manifold constraint is superior.

-----------------------------------------------------------------------

@section se3_summary Summary

Using a 7D ambient SE(3) representation with a manifold:

- avoids trigonometry in the state itself
- cleanly separates state and update
- integrates seamlessly with modern solvers
- matches Gauss–Newton assumptions
- simplifies residual and Jacobian code
- remains fully Lie-theoretic

For ICP and related problems, it is generally the most robust and maintainable
choice.
