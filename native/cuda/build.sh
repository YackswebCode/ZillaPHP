#!/usr/bin/env bash
#
# Build the CUDA library. Requires nvcc in PATH.
#
# On Google Colab:
#   !apt-get install -y nvidia-cuda-toolkit  (or use the preinstalled nvcc)
#   !cd native/cuda && ./build.sh
#
set -e

cd "$(dirname "$0")"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found in PATH."
    echo "Install CUDA Toolkit or run this on a machine with a GPU."
    exit 1
fi

echo "nvcc version:"
nvcc --version | head -4
echo

nvcc -O3 -shared -Xcompiler -fPIC \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -o libzilla_cuda.so \
    matmul.cu elementwise.cu \
    -lcudart

echo "Built: $(pwd)/libzilla_cuda.so"