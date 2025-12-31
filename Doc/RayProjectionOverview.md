@page ray_projection_overview Ray-Projection ICP Documentation

This page provides an overview of the point-to-plane ICP implementation using
ray-projection residuals, with links to detailed documentation.

@section rpo_intro Introduction

Point-to-plane ICP minimizes the distance from source points to target surfaces
along the surface normal direction. The ray-projection variant uses the residual:
\f[
  r = \frac{\mathbf{n}^\top(\mathbf{x} - \mathbf{q})}{\mathbf{n}^\top\mathbf{d}}
\f]

where:
- \f$\mathbf{x}\f$ is the transformed source point
- \f$\mathbf{q}\f$ is the target hit point
- \f$\mathbf{n}\f$ is the target surface normal
- \f$\mathbf{d}\f$ is the ray direction

This formulation requires careful handling of the Jacobians due to the quotient
structure and the SE(3) manifold.

@section rpo_foundations Mathematical Foundations

These documents establish the Lie group machinery used throughout:

- @ref so3_left_jacobian — The SO(3) left Jacobian \f$V(\boldsymbol{\omega})\f$
  coupling rotation and translation in the SE(3) exponential map

- @ref se3_full_chart_jacobian — The 7×6 chart Jacobian mapping tangent
  increments to ambient parameter changes

@section rpo_jacobians Ray-Projection Jacobians

Derivation of the residual Jacobians:

- @ref ray_projection_jacobian — Complete derivation showing why the quotient
  rule is essential for rotation but not for translation

@section rpo_validation Validation Methodology

How to validate Jacobians using finite differences:

- @ref epsilon_sweep_guide — The epsilon sweep technique for finding the
  FD plateau and validating analytic Jacobians

- @ref jacobian_validation — Left vs right perturbations on SO(3)

@section rpo_experimental Experimental Results

Analysis of validation experiments:

- @ref simplified_jacobian_analysis — Measured FD results showing the
  consistent Jacobian matches the plateau while the simplified does not

@section rpo_reading Suggested Reading Order

For someone new to the codebase:

1. This overview
2. @ref so3_left_jacobian — Understand V(ω)
3. @ref se3_full_chart_jacobian — Understand the Plus Jacobian
4. @ref ray_projection_jacobian — The core Jacobian derivation
5. @ref epsilon_sweep_guide — How to validate
6. @ref simplified_jacobian_analysis — See it in practice

@section rpo_implementation Implementation

The Jacobians are implemented in:

| Header | Contents |
|--------|----------|
| `ICP/SE3.h` | `Vso3()`, `se3Plus()`, `plusJacobian7x6()` |
| `ICP/JacobiansAmbient.h` | Ray-projection Jacobians (single-pose) |
| `ICP/JacobiansAmbientTwoPose.h` | Two-pose Jacobians |

Tests are in `ICPTest/JacobiansAmbientTest.cpp` with epsilon sweep validation.
