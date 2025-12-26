#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect platform and compiler (use CMake-style capitalized platform names)
PLATFORM="$(uname -s)"  # Linux, Darwin, Windows
COMPILER="${CC:-gcc}"
COMPILER_NAME="${COMPILER##*/}"  # Extract basename

INSTALL_DIR="${SCRIPT_DIR}/${PLATFORM}/${COMPILER_NAME}/install"
BUILD_DIR="${SCRIPT_DIR}/${PLATFORM}/${COMPILER_NAME}/build"

echo "=============================================="
echo "MultiRegistration Dependencies Builder"
echo "=============================================="

# Check for Intel oneAPI if MKL or TBB requested
if [[ "$*" == *"USE_MKL=ON"* ]] || [[ "$*" == *"USE_TBB=ON"* ]]; then
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
echo "Compiler:          $COMPILER_NAME"
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
    -DCMAKE_C_COMPILER="${COMPILER}" \
    -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
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
    "$@"

# Build all dependencies
echo ""
echo "Building dependencies (this may take a while)..."
cmake --build "$BUILD_DIR" --config Release --parallel

echo ""
echo "=============================================="
echo "Dependencies installed successfully!"
echo "=============================================="
echo ""
echo "To use these dependencies, configure MultiViewICP with:"
echo "  cmake --preset linux-gcc"
echo ""
echo "Usage examples:"
echo "  CC=gcc CXX=g++ ./dependencies.sh           # GCC build (default)"
echo "  CC=clang CXX=clang++ ./dependencies.sh     # Clang build"
echo ""
echo "Optional flags for this script:"
echo "  -DUSE_MKL=ON        Enable Intel MKL (requires: source setvars.sh)"
echo "  -DUSE_TBB=ON        Enable TBB threading (requires: source setvars.sh)"
echo "  -DUSE_CUDA=ON       Enable CUDA support"
echo "  -DUSE_SUITESPARSE=ON Enable SuiteSparse"
echo "  -DDEPS_VERBOSE=ON   Show build output"
