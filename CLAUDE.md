# QuaternionICP Development Notes

## Roadmap

1. **Jacobians.h** - Point-to-plane Jacobians for the 7D ambient parameterization
2. **JacobiansTest.cpp** - Validate against Python using finite differences
3. **ICPAmbient** - Inner loop (linearize, solve) + outer loop (iterate to convergence), matching Python behavior
4. **Ceres integration** - Supply custom SE(3) manifold for the solver
5. **Single pose first**, then extend to **two-pose** registration

## Conventions

- Quaternion storage: `[x, y, z, w]` (Eigen internal order)
- SE(3) pose: 7D vector `[qx, qy, qz, qw, tx, ty, tz]`
- Tangent space: 6D vector `[v_x, v_y, v_z, w_x, w_y, w_z]` (translation, rotation)
- Right-multiplication (body/moving frame): `T_new = T * Exp(delta^)`
