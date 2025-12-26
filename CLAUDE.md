# QuaternionICP Development Notes

## Environment

- Python virtual environment: `source ~/Environments/py313/bin/activate`
- CMake preset: `linux-gcc`
- Configure: `cmake --preset linux-gcc`
- Build: `cmake --build build/linux-gcc`
- Test: `ctest --test-dir build/linux-gcc -C Debug`

## C++ Build Rules

- Never compile standalone .cpp files with g++ directly
- Always use the CMake build system (ninja) - it handles include paths, library linking, and compiler flags
- To test new code: add to existing tests or create new CMake-registered tests, then build with `ninja`

## Roadmap

1. ~~**JacobiansAmbient.h**~~ - Point-to-plane Jacobians for the 7D ambient parameterization ✓
2. ~~**JacobiansAmbientTest.cpp**~~ - Validate against Python using finite differences ✓
3. **ICPSimple** - Inner loop (linearize, solve) + outer loop (iterate to convergence), matching Python behavior
   - ✓ Inner loop (`solveInner`) - matches Python for translation
   - ✗ Outer loop - exists but not validated
   - ✗ Levenberg-Marquardt damping
   - ✗ Line search
4. **Ceres integration** - Supply custom SE(3) manifold for the solver
5. **Single pose first**, then extend to **two-pose** registration
   - ✓ Single pose inner loop
   - ✗ Two-pose registration

## Conventions

- Quaternion storage: `[x, y, z, w]` (Eigen internal order)
- SE(3) pose: 7D vector `[qx, qy, qz, qw, tx, ty, tz]`
- Tangent space: 6D vector `[v_x, v_y, v_z, w_x, w_y, w_z]` (translation, rotation)
- Right-multiplication (body/moving frame): `T_new = T * Exp(delta^)`
