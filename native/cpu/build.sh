#!/usr/bin/env bash
#
# Build the ZillaPHP native CPU library.
#
# Detects OpenBLAS automatically. If found, matmul uses cblas_sgemm
# (5–10× faster). If not, falls back to a naive triple loop.
#
set -e

cd "$(dirname "$0")"

OPENBLAS_CFLAGS=""
OPENBLAS_LIBS=""
USE_OPENBLAS=""

# Try pkg-config first
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openblas 2>/dev/null; then
    OPENBLAS_CFLAGS=$(pkg-config --cflags openblas)
    OPENBLAS_LIBS=$(pkg-config --libs openblas)
    USE_OPENBLAS="-DUSE_OPENBLAS"
    echo "OpenBLAS found via pkg-config"
# Fall back to direct library check
elif [ -f /usr/lib/x86_64-linux-gnu/libopenblas.so ] || \
     [ -f /usr/lib/libopenblas.so ] || \
     ldconfig -p 2>/dev/null | grep -q libopenblas; then
    OPENBLAS_LIBS="-lopenblas"
    USE_OPENBLAS="-DUSE_OPENBLAS"
    echo "OpenBLAS found via ldconfig"
else
    echo "OpenBLAS not found — compiling without it (matmul will be slow)"
    echo "Install with: sudo apt install libopenblas-dev"
fi

# shellcheck disable=SC2086
gcc -O3 -march=native -ffast-math -shared -fPIC \
    $USE_OPENBLAS $OPENBLAS_CFLAGS \
    -o libzilla_cpu.so zilla_cpu.c \
    $OPENBLAS_LIBS \
    -lm

echo "Built: $(pwd)/libzilla_cpu.so"