/*
 * elementwise.cu — CUDA elementwise ops.
 *
 * Same __CUDACC__ guard as matmul.cu so editors do not report
 * CUDA-keyword errors.
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

static void check_err(const char* tag) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA %s error: %s\n", tag, cudaGetErrorString(err));
    }
}

void relu_f32_cuda(const float* X_host, float* Y_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    float *X_d, *Y_d;
    cudaMalloc((void**)&X_d, sz);
    cudaMalloc((void**)&Y_d, sz);
    cudaMemcpy(X_d, X_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    relu_kernel<<<grid, ELEM_BLOCK>>>(X_d, Y_d, N);
    cudaDeviceSynchronize();
    check_err("relu");

    cudaMemcpy(Y_host, Y_d, sz, cudaMemcpyDeviceToHost);
    cudaFree(X_d);
    cudaFree(Y_d);
}

void add_f32_cuda(const float* A_host, const float* B_host, float* C_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, sz);
    cudaMalloc((void**)&B_d, sz);
    cudaMalloc((void**)&C_d, sz);
    cudaMemcpy(A_d, A_host, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    add_kernel<<<grid, ELEM_BLOCK>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    check_err("add");

    cudaMemcpy(C_host, C_d, sz, cudaMemcpyDeviceToHost);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

void sub_f32_cuda(const float* A_host, const float* B_host, float* C_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, sz);
    cudaMalloc((void**)&B_d, sz);
    cudaMalloc((void**)&C_d, sz);
    cudaMemcpy(A_d, A_host, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    sub_kernel<<<grid, ELEM_BLOCK>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    check_err("sub");

    cudaMemcpy(C_host, C_d, sz, cudaMemcpyDeviceToHost);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

void mul_f32_cuda(const float* A_host, const float* B_host, float* C_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, sz);
    cudaMalloc((void**)&B_d, sz);
    cudaMalloc((void**)&C_d, sz);
    cudaMemcpy(A_d, A_host, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    mul_kernel<<<grid, ELEM_BLOCK>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    check_err("mul");

    cudaMemcpy(C_host, C_d, sz, cudaMemcpyDeviceToHost);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

void div_f32_cuda(const float* A_host, const float* B_host, float* C_host, int N) {
    size_t sz = (size_t)N * sizeof(float);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, sz);
    cudaMalloc((void**)&B_d, sz);
    cudaMalloc((void**)&C_d, sz);
    cudaMemcpy(A_d, A_host, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_host, sz, cudaMemcpyHostToDevice);

    int grid = (N + ELEM_BLOCK - 1) / ELEM_BLOCK;
    div_kernel<<<grid, ELEM_BLOCK>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    check_err("div");

    cudaMemcpy(C_host, C_d, sz, cudaMemcpyDeviceToHost);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

} // extern "C"

#else
/* Not compiled with nvcc — this file is CUDA-only. */
#endif /* __CUDACC__ */