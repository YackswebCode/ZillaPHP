/*
 * elementwise.cu — CUDA elementwise ops.
 * Uses persistent scratch buffers for the same reason as matmul.cu.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

extern "C" {

#define ELEM_BLOCK 256

__global__ void relu_kernel(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
}

__global__ void add_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
__global__ void sub_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] - B[i];
}
__global__ void mul_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * B[i];
}
__global__ void div_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] / B[i];
}

/* -------- Persistent scratch buffers -------- */
static float* g_X  = NULL;
static float* g_Y  = NULL;
static float* g_Z  = NULL;
static size_t g_cap = 0;   // bytes

static void ensure_capacity(size_t need) {
    if (need <= g_cap) return;

    size_t newcap = ((need + (16u << 20) - 1) / (16u << 20)) * (16u << 20);

    if (g_X) cudaFree(g_X);
    if (g_Y) cudaFree(g_Y);
    if (g_Z) cudaFree(g_Z);

    cudaError_t e1 = cudaMalloc((void**)&g_X, newcap);
    cudaError_t e2 = cudaMalloc((void**)&g_Y, newcap);
    cudaError_t e3 = cudaMalloc((void**)&g_Z, newcap);

    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        fprintf(stderr, "CUDA elementwise alloc failed\n");
        g_cap = 0;
        return;
    }
    g_cap = newcap;
}

static void check_err(const char* tag) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA %s error: %s\n", tag, cudaGetErrorString(err));
}

void relu_f32_cuda(const float* X_host, float* Y_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    ensure_capacity(sz);
    if (g_cap == 0) return;

    cudaMemcpy(g_X, X_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    relu_kernel<<<grid, ELEM_BLOCK>>>(g_X, g_Y, N);
    cudaDeviceSynchronize();
    check_err("relu");

    cudaMemcpy(Y_host, g_Y, sz, cudaMemcpyDeviceToHost);
}

void add_f32_cuda(const float* A_host, const float* B_host, float* C_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    ensure_capacity(sz);
    if (g_cap == 0) return;

    cudaMemcpy(g_X, A_host, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_Y, B_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    add_kernel<<<grid, ELEM_BLOCK>>>(g_X, g_Y, g_Z, N);
    cudaDeviceSynchronize();
    check_err("add");

    cudaMemcpy(C_host, g_Z, sz, cudaMemcpyDeviceToHost);
}

void sub_f32_cuda(const float* A_host, const float* B_host, float* C_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    ensure_capacity(sz);
    if (g_cap == 0) return;

    cudaMemcpy(g_X, A_host, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_Y, B_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    sub_kernel<<<grid, ELEM_BLOCK>>>(g_X, g_Y, g_Z, N);
    cudaDeviceSynchronize();
    check_err("sub");

    cudaMemcpy(C_host, g_Z, sz, cudaMemcpyDeviceToHost);
}

void mul_f32_cuda(const float* A_host, const float* B_host, float* C_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    ensure_capacity(sz);
    if (g_cap == 0) return;

    cudaMemcpy(g_X, A_host, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_Y, B_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    mul_kernel<<<grid, ELEM_BLOCK>>>(g_X, g_Y, g_Z, N);
    cudaDeviceSynchronize();
    check_err("mul");

    cudaMemcpy(C_host, g_Z, sz, cudaMemcpyDeviceToHost);
}

void div_f32_cuda(const float* A_host, const float* B_host, float* C_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    ensure_capacity(sz);
    if (g_cap == 0) return;

    cudaMemcpy(g_X, A_host, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_Y, B_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    div_kernel<<<grid, ELEM_BLOCK>>>(g_X, g_Y, g_Z, N);
    cudaDeviceSynchronize();
    check_err("div");

    cudaMemcpy(C_host, g_Z, sz, cudaMemcpyDeviceToHost);
}

} // extern "C"

#else
/* Not compiled with nvcc — this file is CUDA-only. */
#endif /* __CUDACC__ */