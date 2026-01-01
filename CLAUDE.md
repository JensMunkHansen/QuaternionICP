# QuaternionICP Development Notes

## Current TODO: Embree Backend Implementation

**Status:** Infrastructure complete, stub implementation needs to be finished.

**What's done:**
- `IntersectionBackend.h` - Abstract interface for ray-mesh intersection backends
- `GridSearchBackend.h` - GridSearch implementation (zero overhead, production ready)
- `EmbreeBackend.h` - Stub implementation (compiles but returns empty results)
- CMake: `USE_EMBREE` option, `find_package(embree 4)`, `ICP_USE_EMBREE` in Config.h
- Grid caches backend via `getBackend()` (lazy initialization)
- Correspondences.h uses backend abstraction

**What's needed:**
1. Implement `EmbreeBackend::build()` - extract triangles from Grid, build Embree BVH
2. Implement `EmbreeBackend::intersectParallel()` - ray tracing with Embree
3. Test with `-DUSE_EMBREE=ON` and verify results match GridSearch

**Key files:**
- `ICP/IntersectionBackend.h` - interface
- `ICP/GridSearchBackend.h` - working implementation + factory function
- `ICP/EmbreeBackend.h` - stub to be completed
- `ICP/Grid.h` - `getBackend()` method
- `ICP/Correspondences.h` - uses `tgtGrid.getBackend().intersectParallel(...)`

**Build with Embree:**
```bash
cmake --preset linux-gcc -DUSE_EMBREE=ON
cmake --build build/linux-gcc --config Release
```

---

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
6. ~~**MultiICP Infrastructure**~~ ✓
   - ✓ AABB boxes for grids (`Grid::AABB`, `computeWorldAABB()`)
   - ✓ Compute potential edges using AABB overlap
   - ✓ Load EXR files and test two-file registration
7. ~~**Grid Refactoring**~~ ✓
   - ✓ Grid struct: `initialPose` (Isometry3d) and `pose` (Pose7)
   - ✓ Intersection backend abstraction (`IntersectionBackend`, `GridSearchBackend`)
   - ✓ Embree support infrastructure (CMake option, stub backend)
8. ~~**Ceres Two-Pose Integration**~~ ✓

### Upcoming
9. **MultiICP Testing**
   - ✗ Test multiview with 2 grids against SingleICP reference
   - ✗ Test with collection of up to 5000 grids
   - ✗ Performance optimization

## Conventions

- Quaternion storage: `[x, y, z, w]` (Eigen internal order)
- SE(3) pose: 7D vector `[qx, qy, qz, qw, tx, ty, tz]`
- Tangent space: 6D vector `[v_x, v_y, v_z, w_x, w_y, w_z]` (translation, rotation)
- Right-multiplication (body/moving frame): `T_new = T * Exp(delta^)`
