#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default compiler
COMPILER_CHOICE="gcc"
KEEP_BUILD=false

# Help function
show_help() {
    cat << 'EOF'
QuaternionICP Dependencies Builder

Usage: ./dependencies.sh [OPTIONS] [CMAKE_ARGS...]

Options:
  --help, -h            Show this help message
  --compiler=<gcc|clang> Select compiler (default: gcc)
  --keep-build          Keep build directory after install

Optional CMake flags (pass directly):
  -DUSE_MKL=ON          Enable Intel MKL for LAPACK/BLAS
                        Requires: source /opt/intel/oneapi/setvars.sh

  -DUSE_TBB=ON          Enable Intel TBB for parallel algorithms
                        Requires: source /opt/intel/oneapi/setvars.sh

  -DUSE_CUDA=ON         Enable CUDA support for GPU acceleration
                        Requires: CUDA toolkit installed

  -DUSE_SUITESPARSE=ON  Enable SuiteSparse for sparse Cholesky
                        Required for: SPARSE_NORMAL_CHOLESKY, CudaSparseCholesky
                        Combine with -DUSE_CUDA=ON for GPU-accelerated sparse

  -DDEPS_VERBOSE=ON     Show detailed build output

Examples:
  ./dependencies.sh
      Build with GCC, default options

  ./dependencies.sh --compiler=clang
      Build with Clang

  ./dependencies.sh -DUSE_MKL=ON -DUSE_TBB=ON
      Build with Intel MKL and TBB (run setvars.sh first)

  ./dependencies.sh -DUSE_CUDA=ON -DUSE_SUITESPARSE=ON
      Build with CUDA and SuiteSparse for GPU sparse solvers

Feature Matrix:
  Linear Solver              Requires
  ─────────────────────────  ────────────────────────
  DenseQR                    (always available)
  DenseSchur                 (always available)
  SparseSchur                USE_SUITESPARSE=ON
  SparseNormalCholesky       USE_SUITESPARSE=ON
  CudaDenseCholesky          USE_CUDA=ON
  CudaSparseCholesky         USE_CUDA=ON + USE_SUITESPARSE=ON

EOF
    exit 0
}

# Parse arguments
CMAKE_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            ;;
        --compiler)
            COMPILER_CHOICE="$2"
            shift 2
            ;;
        --compiler=*)
            COMPILER_CHOICE="${1#*=}"
            shift
            ;;
        --keep-build)
            KEEP_BUILD=true
            shift
            ;;
        *)
            CMAKE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set compiler based on choice
case "$COMPILER_CHOICE" in
    gcc)
        CC_COMPILER="gcc"
        CXX_COMPILER="g++"
        ;;
    clang)
        CC_COMPILER="clang"
        CXX_COMPILER="clang++"
        ;;
    *)
        echo "ERROR: Unknown compiler '$COMPILER_CHOICE'. Use 'gcc' or 'clang'."
        exit 1
        ;;
esac

# Detect platform (use CMake-style capitalized platform names)
PLATFORM="$(uname -s)"  # Linux, Darwin, Windows

INSTALL_DIR="${SCRIPT_DIR}/${PLATFORM}/${COMPILER_CHOICE}/install"
BUILD_DIR="${SCRIPT_DIR}/${PLATFORM}/${COMPILER_CHOICE}/build"

echo "=============================================="
echo "MultiRegistration Dependencies Builder"
echo "=============================================="

# Check for Intel oneAPI if MKL or TBB requested
if [[ "${CMAKE_ARGS[*]}" == *"USE_MKL=ON"* ]] || [[ "${CMAKE_ARGS[*]}" == *"USE_TBB=ON"* ]]; then
    if [ -z "$MKLROOT" ] || [ -z "$TBBROOT" ]; then
        echo ""
        echo "ERROR: MKL or TBB requested but oneAPI environment not set."
        echo "Please run first:"
        echo "  source /opt/intel/oneapi/setvars.sh"
        echo ""
        exit 1
    fi
    echo "Intel oneAPI detected:"
    echo "  MKLROOT: $MKLROOT"
    echo "  TBBROOT: $TBBROOT"
fi

echo ""
echo "Platform:          $PLATFORM"
echo "Compiler:          $COMPILER_CHOICE ($CC_COMPILER / $CXX_COMPILER)"
echo "Install directory: $INSTALL_DIR"
echo "Build directory:   $BUILD_DIR"
echo ""

# Create directories
mkdir -p "$INSTALL_DIR" "$BUILD_DIR"

# Configure NativeDeps
echo "Configuring dependencies..."
cmake -S "$SCRIPT_DIR/NativeDeps" -B "$BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC_COMPILER}" \
    -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
    -DBUILD_EIGEN3=ON \
    -DBUILD_CATCH2=ON \
    -DBUILD_ABSEIL=ON \
    -DBUILD_CERES=ON \
    -DBUILD_SOPHUS=ON \
    -DBUILD_ARGS=ON \
    -DBUILD_MINIZ=ON \
    -DBUILD_TINYEXR=ON \
    -DBUILD_SIMDMATH=ON \
    -DBUILD_GRIDSEARCH=ON \
    -DDEPS_VERBOSE=OFF \
    "${CMAKE_ARGS[@]}"

# Build all dependencies
echo ""
echo "Building dependencies (this may take a while)..."
cmake --build "$BUILD_DIR" --config Release --parallel

# Clean up build directory to save space
if [ "$KEEP_BUILD" = "false" ]; then
    echo ""
    echo "Cleaning up build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    echo "Build directory removed."
fi

echo ""
echo "=============================================="
echo "Dependencies installed successfully!"
echo "=============================================="
echo ""
echo "To use these dependencies, configure MultiViewICP with:"
echo "  cmake --preset linux-${COMPILER_CHOICE}"
echo ""
echo "Usage:"
echo "  ./dependencies.sh                      # GCC build (default)"
echo "  ./dependencies.sh --compiler=clang     # Clang build"
echo "  ./dependencies.sh --compiler=gcc       # GCC build (explicit)"
echo "  ./dependencies.sh --keep-build         # Keep build directory"
echo ""
echo "Optional CMake flags:"
echo "  -DUSE_MKL=ON         Enable Intel MKL (requires: source setvars.sh)"
echo "  -DUSE_TBB=ON         Enable TBB threading (requires: source setvars.sh)"
echo "  -DUSE_CUDA=ON        Enable CUDA support"
echo "  -DUSE_SUITESPARSE=ON Enable SuiteSparse"
echo "  -DDEPS_VERBOSE=ON    Show build output"
