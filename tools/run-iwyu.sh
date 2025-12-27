#!/bin/bash
# Run include-what-you-use on GridSearch
# Usage: ./run-iwyu.sh [gcc|clang] [--fix] [--verify]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MAPPING_FILE="$SCRIPT_DIR/iwyu.imp"

# Default to clang
TOOLCHAIN="${1:-clang}"
if [[ "$TOOLCHAIN" == "--"* ]]; then
    # First arg is a flag, use default toolchain
    TOOLCHAIN="clang"
else
    shift 2>/dev/null || true
fi

# Parse remaining arguments
FIX_MODE=false
VERIFY_MODE=false
for arg in "$@"; do
    case $arg in
        --fix) FIX_MODE=true ;;
        --verify) VERIFY_MODE=true; FIX_MODE=true ;;
    esac
done

case "$TOOLCHAIN" in
    gcc)   BUILD_DIR="$PROJECT_ROOT/build/linux-gcc" ;;
    clang) BUILD_DIR="$PROJECT_ROOT/build/linux-clang" ;;
    *)
        echo "Usage: $0 [gcc|clang] [--fix] [--verify]"
        echo "  gcc     - Use build/linux-gcc"
        echo "  clang   - Use build/linux-clang (default)"
        echo "  --fix   - Apply suggested changes"
        echo "  --verify - Apply fixes then build both gcc and clang"
        exit 1
        ;;
esac

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
    echo "Error: $BUILD_DIR/compile_commands.json not found"
    echo "Run: cmake --preset linux-${TOOLCHAIN}"
    exit 1
fi

echo "IWYU analysis using $TOOLCHAIN build"
echo ""

if [[ "$FIX_MODE" == "true" ]]; then
    echo "Applying IWYU suggestions..."
    iwyu_tool -p "$BUILD_DIR" -- \
        -Xiwyu --mapping_file="$MAPPING_FILE" \
        -Xiwyu --no_fwd_decls 2>&1 | fix_include --nosafe_headers || true

    if [[ "$VERIFY_MODE" == "true" ]]; then
        echo ""
        echo "=== Verifying builds ==="

        echo ""
        echo "Building with clang..."
        if cmake --build "$PROJECT_ROOT/build/linux-clang" --parallel; then
            echo "Clang build: OK"
        else
            echo "Clang build: FAILED"
            exit 1
        fi

        echo ""
        echo "Building with gcc..."
        if cmake --build "$PROJECT_ROOT/build/linux-gcc" --parallel; then
            echo "GCC build: OK"
        else
            echo "GCC build: FAILED"
            exit 1
        fi

        echo ""
        echo "=== All builds passed ==="
    fi
else
    iwyu_tool -p "$BUILD_DIR" -- \
        -Xiwyu --mapping_file="$MAPPING_FILE" \
        -Xiwyu --no_fwd_decls 2>&1 | tee "$PROJECT_ROOT/iwyu-report.txt"
    echo ""
    echo "Report: iwyu-report.txt"
    echo "Apply:  $0 $TOOLCHAIN --fix"
    echo "Verify: $0 $TOOLCHAIN --verify"
fi
