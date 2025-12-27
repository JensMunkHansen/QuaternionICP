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

### Completed
1. ~~**JacobiansAmbient.h**~~ - Point-to-plane Jacobians for the 7D ambient parameterization ✓
2. ~~**JacobiansAmbientTest.cpp**~~ - Validate against Python using finite differences ✓
3. ~~**ICPSimple**~~ - Inner loop + outer loop, LM damping, line search ✓
4. ~~**Ceres integration (single-pose)**~~ - Custom SE(3) manifold for solver ✓
5. ~~**Two-pose Jacobians**~~ - `JacobiansAmbientTwoPose.h` with consistent/simplified variants ✓
   - ✓ Forward and reverse ray cost functions
   - ✓ FD validation with epsilon sweep

### In Progress
6. **MultiICP Infrastructure**
   - ✗ AABB boxes for grids (bounding volume)
   - ✗ Compute potential edges using AABB overlap
   - ✗ Load EXR files and test two-file registration with SingleICP

7. **Code Refactoring**
   - ✗ Move random noise utilities out of SingleICP main (share with MultiICP)
   - ✗ Grid struct: add initial pose (`Pose7`) and running pose for multiview

8. **Ceres Two-Pose Integration**
   - ✗ Implement Ceres problem using `<1,7,7>` cost functions
   - ✗ Reuse `InnerParams`, extend `OuterParams` (add `SCHUR_ITERATIVE`)

### Testing & Validation
9. **MultiICP Testing**
   - ✗ Test multiview with 2 grids against SingleICP reference
   - ✗ Test with collection of up to 5000 grids
   - ✗ Performance optimization

## Conventions

- Quaternion storage: `[x, y, z, w]` (Eigen internal order)
- SE(3) pose: 7D vector `[qx, qy, qz, qw, tx, ty, tz]`
- Tangent space: 6D vector `[v_x, v_y, v_z, w_x, w_y, w_z]` (translation, rotation)
- Right-multiplication (body/moving frame): `T_new = T * Exp(delta^)`
