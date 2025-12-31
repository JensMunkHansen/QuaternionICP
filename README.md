# QuaternionICP

Multi-view point cloud registration using ray-projected ICP with quaternion parameterization.

## Dependencies

**Required (built by `dependencies.sh`):**
- Eigen3 - Linear algebra
- Ceres Solver - Nonlinear least squares
- Abseil - C++ utilities (Ceres dependency)
- Sophus - Lie groups (SE3/SO3)
- tinyexr + miniz - EXR file loading
- GridSearch + SIMDMath - Spatial search
- args - Command-line parsing

**Optional:**
- TBB - Threading (`sudo apt install libtbb-dev` on Ubuntu) - enabled by default
- Catch2 - Testing framework (built by default)
- Intel MKL - LAPACK acceleration
- CUDA - GPU acceleration
- SuiteSparse - Sparse solvers

## Building

### 1. Build Dependencies

```bash
./dependencies.sh                      # GCC (default)
./dependencies.sh --compiler=clang     # Clang
```

Optional flags:
```bash
./dependencies.sh -DUSE_MKL=ON         # Intel MKL (requires: source /opt/intel/oneapi/setvars.sh)
./dependencies.sh -DUSE_TBB=ON         # TBB threading
./dependencies.sh -DUSE_CUDA=ON        # CUDA support
```

### 2. Build Project

```bash
cmake --preset linux-gcc
cmake --build build/linux-gcc
```

## Usage

### SingleICP

Register two depth grids:

```bash
SingleICPMain --source <source.exr> --target <target.exr>
```

### MultiICP

Register multiple grids from a folder:

```bash
MultiICPMain --grid-folder <folder>
```

Or register two grids:

```bash
MultiICPMain --source <source.exr> --target <target.exr>
```

Use `--help` for all options.
