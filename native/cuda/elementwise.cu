/*
 * elementwise.cu — CUDA elementwise ops.
 */

#include <cuda_runtime.h>
#include <stdio.h>

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

void relu_f32_cuda(const float* X, float* Y, int N) {
    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    relu_kernel<<<grid, ELEM_BLOCK>>>(X, Y, N);
}

void add_f32_cuda(const float* A, const float* B, float* C, int N) {
    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    add_kernel<<<grid, ELEM_BLOCK>>>(A, B, C, N);
}

void sub_f32_cuda(const float* A, const float* B, float* C, int N) {
    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    sub_kernel<<<grid, ELEM_BLOCK>>>(A, B, C, N);
}

void mul_f32_cuda(const float* A, const float* B, float* C, int N) {
    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    mul_kernel<<<grid, ELEM_BLOCK>>>(A, B, C, N);
}

void div_f32_cuda(const float* A, const float* B, float* C, int N) {
    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    div_kernel<<<grid, ELEM_BLOCK>>>(A, B, C, N);
}

} // extern "C"