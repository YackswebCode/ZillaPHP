/*
 * matmul.cu — CUDA matrix multiplication kernel.
 *
 * Computes C[M,N] = A[M,K] @ B[K,N]  (row-major).
 *
 * Compile on a machine with CUDA:
 *   nvcc -O3 -shared -Xcompiler -fPIC -o libzilla_cuda.so *.cu
 *
 * NOTE: The entire file is guarded by __CUDACC__ so that editors
 * (VS Code, clangd) using a non-CUDA compiler do not report errors
 * on CUDA-only keywords. nvcc defines __CUDACC__ automatically.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

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
    int kTiles = (K + TILE - 1) / TILE;

    for (int k = 0; k < kTiles; k++) {
        int aCol = k * TILE + threadIdx.x;
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int bRow = k * TILE + threadIdx.y;
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
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
    const float* A_host,
    const float* B_host,
    float*       C_host,
    int M, int K, int N
) {
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *A_dev = NULL, *B_dev = NULL, *C_dev = NULL;
    cudaError_t err;

    err = cudaMalloc((void**)&A_dev, sizeA);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc A failed: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&B_dev, sizeB);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc B failed: %s\n", cudaGetErrorString(err));
        cudaFree(A_dev); return;
    }
    err = cudaMalloc((void**)&C_dev, sizeC);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc C failed: %s\n", cudaGetErrorString(err));
        cudaFree(A_dev); cudaFree(B_dev); return;
    }

    cudaMemcpy(A_dev, A_host, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, sizeB, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_kernel<<<grid, block>>>(A_dev, B_dev, C_dev, M, K, N);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA matmul kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(A_dev); cudaFree(B_dev); cudaFree(C_dev);
        return;
    }

    cudaMemcpy(C_host, C_dev, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
}

} // extern "C"

#else
/* Not compiled with nvcc — this file is CUDA-only. */
#endif /* __CUDACC__ */