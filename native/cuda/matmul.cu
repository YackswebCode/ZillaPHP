/*
 * matmul.cu — CUDA matrix multiplication kernel.
 *
 * Computes C[M,N] = A[M,K] @ B[K,N]  (row-major).
 *
 * Compile on a machine with CUDA:
 *   nvcc -O3 -shared -Xcompiler -fPIC -o libzilla_cuda.so *.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

#define TILE 16

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < (K + TILE - 1) / TILE; k++) {
        // Load A tile
        if (row < M && (k * TILE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + k * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile (transposed access for coalescing)
        if ((k * TILE + threadIdx.y) < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(k * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_f32(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA matmul error: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"