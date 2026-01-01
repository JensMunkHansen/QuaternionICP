# QuaternionICP

Multi-view point cloud registration using ray-projected ICP with quaternion parameterization.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jensmunkhansen.github.io/QuaternionICP/)

## Dependencies

**Required (built by `dependencies.sh`):**
- Eigen3 - Linear algebra
- Ceres Solver - Nonlinear least squares
- Abseil - C++ utilities (Ceres dependency)
- Sophus - Lie groups (SE3/SO3)
- tinyexr + miniz - EXR file loading
- GridSearch + SIMDMath - Spatial search
- args - Command-line parsing
- Catch2 - Testing framework

**Optional:**
- Intel MKL - LAPACK/BLAS acceleration for Ceres
- TBB - Threading for Ceres (`sudo apt install libtbb-dev`)
- CUDA - GPU acceleration for Ceres
- SuiteSparse - Sparse Cholesky solvers

## Building

### 1. Build Dependencies

```bash
./dependencies.sh                      # Default build (GCC)
./dependencies.sh --compiler=clang     # Build with Clang
./dependencies.sh --help               # Show all options
```

**Optional features:**

| Flag | Description | Requirements |
|------|-------------|--------------|
| `-DUSE_MKL=ON` | Intel MKL for LAPACK/BLAS | `export MKLROOT=/opt/intel/oneapi/mkl/latest` |
| `-DUSE_TBB=ON` | TBB threading in Ceres | `sudo apt install libtbb-dev` |
| `-DUSE_CUDA=ON` | CUDA GPU acceleration | CUDA toolkit installed |
| `-DUSE_SUITESPARSE=ON` | Sparse solvers | Combined with CUDA for GPU sparse |

**Examples:**

```bash
# MKL + TBB (recommended for CPU performance)
export MKLROOT=/opt/intel/oneapi/mkl/latest
./dependencies.sh -DUSE_MKL=ON -DUSE_TBB=ON

# CUDA for GPU-accelerated dense solvers
./dependencies.sh -DUSE_CUDA=ON

# CUDA + SuiteSparse for GPU-accelerated sparse solvers
./dependencies.sh -DUSE_CUDA=ON -DUSE_SUITESPARSE=ON

# Full configuration (MKL, TBB, CUDA, SuiteSparse)
export MKLROOT=/opt/intel/oneapi/mkl/latest
./dependencies.sh -DUSE_MKL=ON -DUSE_TBB=ON -DUSE_CUDA=ON -DUSE_SUITESPARSE=ON
```

**Ceres Linear Solver Availability:**

| Linear Solver | Flags | Backend |
|---------------|-------|---------|
| DENSE_QR | (none) | Eigen |
| DENSE_SCHUR | (none) | Eigen |
| SPARSE_SCHUR | (none) | Eigen sparse |
| SPARSE_SCHUR | `-DUSE_SUITESPARSE=ON` | CHOLMOD (faster) |
| SPARSE_NORMAL_CHOLESKY | (none) | Eigen sparse |
| SPARSE_NORMAL_CHOLESKY | `-DUSE_SUITESPARSE=ON` | CHOLMOD (faster) |
| CUDA_DENSE_CHOLESKY | `-DUSE_CUDA=ON` | cuSOLVER |
| CUDA_SPARSE_CHOLESKY | `-DUSE_CUDA=ON -DUSE_SUITESPARSE=ON` | cuSPARSE + CHOLMOD |

### 2. Build Project

```bash
cmake --preset linux-gcc
cmake --build build/linux-gcc
```

**Available presets:**

| Preset | Description |
|--------|-------------|
| `linux-gcc` | GCC with MKL support (default) |
| `linux-gcc-nomkl` | GCC without MKL |
| `linux-clang` | Clang build |

### 3. Run Tests

```bash
ctest --test-dir build/linux-gcc -C Debug
```

## Important: MKL Configuration

If you built dependencies with `-DUSE_MKL=ON`, you **must** use a preset with MKL enabled (`linux-gcc`). The MKL shared libraries need to be in the binary's RUNPATH.

| Dependencies built with | Use preset |
|------------------------|------------|
| `-DUSE_MKL=ON` | `linux-gcc` (default) |
| No MKL flag | `linux-gcc-nomkl` |

Mismatched configurations will cause runtime errors like:
```
error while loading shared libraries: libmkl_intel_lp64.so.2: cannot open shared object file
```

## Usage

### SingleICP

Register two depth grids:

```bash
./build/linux-gcc/Debug/bin/SingleICPMain --source <source.exr> --target <target.exr>
```

### MultiICP

Register multiple grids from a folder:

```bash
./build/linux-gcc/Debug/bin/MultiICPMain --grid-folder <folder>
```

Or register two grids:

```bash
./build/linux-gcc/Debug/bin/MultiICPMain --source <source.exr> --target <target.exr>
```

Use `--help` for all options.

## Project Structure

```
QuaternionICP/
├── ICP/                 # Core ICP library
├── SingleICP/           # Single-pair registration executable
├── MultiICP/            # Multi-view registration executable
├── ICPTest/             # Unit tests
├── NativeDeps/          # Dependency build system
├── CMake/               # CMake modules
└── Doc/                 # Documentation (Doxygen)
```

## Documentation

Generate Doxygen documentation:

```bash
cmake --build build/linux-gcc --target doc
```

Output: `build/linux-gcc/doc/html/index.html`

## Continuous Integration

The CI workflow runs on Ubuntu 24.04 with a minimal dependency configuration:

| Component | CI Configuration |
|-----------|------------------|
| Intersection backend | Embree 4 (system `libembree-dev`) |
| Nonlinear solver | Ceres 2.2 (system `libceres-dev`) |
| Sparse solvers | SuiteSparse (via `libceres-dev`) |
| Intel MKL | Disabled |
| GridSearch | Disabled (requires private repo) |
| SIMDMath | Disabled (requires private repo) |

Dependencies built from source: Eigen3, Sophus, Catch2, args, miniz, tinyexr.

The CI uses `linux-gcc-ci` and `linux-clang-ci` presets which set `USE_EMBREE=ON` and `USE_GRIDSEARCH=OFF`.
