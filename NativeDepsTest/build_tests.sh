#!/bin/bash
# Build and test all MKL/TBB configurations using CMake presets
#
# Usage:
#   ./build_tests.sh              # Build and test all
#   ./build_tests.sh --clean      # Clean and rebuild all
#   ./build_tests.sh --no-test    # Build only

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

CLEAN=0
RUN_TESTS=1
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=1 ;;
        --no-test) RUN_TESTS=0 ;;
        --help|-h)
            echo "Usage: $0 [--clean] [--no-test]"
            exit 0
            ;;
    esac
done

# Configure presets to test
PRESETS=(linux-gcc linux-gcc-mkl linux-gcc-tbb linux-gcc-mkl-tbb)

for preset in "${PRESETS[@]}"; do
    echo ""
    echo "======== $preset ========"

    [[ $CLEAN -eq 1 ]] && rm -rf "build/$preset"

    cmake --preset "$preset"
    cmake --build "build/$preset" --config Release

    [[ $RUN_TESTS -eq 1 ]] && ctest --test-dir "build/$preset" -C Release --output-on-failure
done

echo ""
echo "All configurations built and tested successfully!"
