/*
 * matmul.cu — CUDA matrix multiplication kernel.
 *
 * Uses persistent scratch buffers to avoid the ~40ms/call cudaMalloc
 * penalty on virtualized GPUs (Colab's T4).
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

/* -------- Persistent scratch buffers -------- */
static float* g_A  = NULL;
static float* g_B  = NULL;
static float* g_C  = NULL;
static size_t g_cap = 0;   // capacity in BYTES

static void ensure_capacity(size_t need) {
    if (need <= g_cap) return;

    // Round up to next 16 MB to reduce reallocations
    size_t newcap = ((need + (16u << 20) - 1) / (16u << 20)) * (16u << 20);

    if (g_A) cudaFree(g_A);
    if (g_B) cudaFree(g_B);
    if (g_C) cudaFree(g_C);

    cudaError_t e1 = cudaMalloc((void**)&g_A, newcap);
    cudaError_t e2 = cudaMalloc((void**)&g_B, newcap);
    cudaError_t e3 = cudaMalloc((void**)&g_C, newcap);

    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        fprintf(stderr, "CUDA buffer alloc failed (%zu bytes): %s\n",
                newcap, cudaGetErrorString(cudaGetLastError()));
        g_cap = 0;
        return;
    }
    g_cap = newcap;
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
    size_t need  = sizeA > sizeB ? sizeA : sizeB;
    if (sizeC > need) need = sizeC;

    ensure_capacity(need);
    if (g_cap == 0) return;

    cudaMemcpy(g_A, A_host, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(g_B, B_host, sizeB, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_kernel<<<grid, block>>>(g_A, g_B, g_C, M, K, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA matmul kernel error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(C_host, g_C, sizeC, cudaMemcpyDeviceToHost);
}

} // extern "C"

#else
/* Not compiled with nvcc — this file is CUDA-only. */
#endif /* __CUDACC__ */