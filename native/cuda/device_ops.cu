/*
 * device_ops.cu — kernel variants that operate entirely on device buffers.
 *
 * Each function takes integer buffer handles from buffer_registry.cu.
 * No memcpy inside. No host round-trip.
 *
 * Callers should:
 *   1. zilla_buffer_alloc + zilla_buffer_upload for inputs (once)
 *   2. zilla_buffer_alloc for outputs (once)
 *   3. call these * _dev functions in a loop
 *   4. zilla_buffer_download to fetch results (once)
 *   5. zilla_buffer_free all handles
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

extern "C" {

/* Provided by buffer_registry.cu */
float* zilla_buffer_ptr(int id);

/* Persistent cuBLAS handle */
static cublasHandle_t g_cublas_dev = NULL;

static cublasHandle_t get_cublas(void) {
    if (g_cublas_dev == NULL) {
        if (cublasCreate(&g_cublas_dev) != CUBLAS_STATUS_SUCCESS) {
            return NULL;
        }
        cublasSetStream(g_cublas_dev, NULL);
    }
    return g_cublas_dev;
}

/* ---------- matmul_dev ------------------------------------------------
 *
 *   C = A @ B     all on-device. Row-major, mapped to cuBLAS via transpose.
 *   Returns 0 on success, -1 on failure.
 */
int matmul_dev(int a_id, int b_id, int c_id, int M, int K, int N) {
    float* A = zilla_buffer_ptr(a_id);
    float* B = zilla_buffer_ptr(b_id);
    float* C = zilla_buffer_ptr(c_id);
    if (!A || !B || !C) {
        fprintf(stderr, "matmul_dev: invalid buffer handle\n");
        return -1;
    }

    cublasHandle_t h = get_cublas();
    if (!h) return -1;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasStatus_t st = cublasSgemm(
        h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "matmul_dev: cublasSgemm failed (%d)\n", (int)st);
        return -1;
    }
    return 0;
}

/* ---------- Elementwise kernels on device pointers ------------------- */

#define ELEM_BLOCK 256

__global__ void add_dev_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
__global__ void sub_dev_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] - B[i];
}
__global__ void mul_dev_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * B[i];
}
__global__ void div_dev_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] / B[i];
}
__global__ void relu_dev_kernel(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
}

int add_dev(int a_id, int b_id, int c_id, int n) {
    float* A = zilla_buffer_ptr(a_id);
    float* B = zilla_buffer_ptr(b_id);
    float* C = zilla_buffer_ptr(c_id);
    if (!A || !B || !C) return -1;
    int grid = (n + ELEM_BLOCK - 1) / ELEM_BLOCK;
    add_dev_kernel<<<grid, ELEM_BLOCK>>>(A, B, C, n);
    return 0;
}

int sub_dev(int a_id, int b_id, int c_id, int n) {
    float* A = zilla_buffer_ptr(a_id);
    float* B = zilla_buffer_ptr(b_id);
    float* C = zilla_buffer_ptr(c_id);
    if (!A || !B || !C) return -1;
    int grid = (n + ELEM_BLOCK - 1) / ELEM_BLOCK;
    sub_dev_kernel<<<grid, ELEM_BLOCK>>>(A, B, C, n);
    return 0;
}

int mul_dev(int a_id, int b_id, int c_id, int n) {
    float* A = zilla_buffer_ptr(a_id);
    float* B = zilla_buffer_ptr(b_id);
    float* C = zilla_buffer_ptr(c_id);
    if (!A || !B || !C) return -1;
    int grid = (n + ELEM_BLOCK - 1) / ELEM_BLOCK;
    mul_dev_kernel<<<grid, ELEM_BLOCK>>>(A, B, C, n);
    return 0;
}

int div_dev(int a_id, int b_id, int c_id, int n) {
    float* A = zilla_buffer_ptr(a_id);
    float* B = zilla_buffer_ptr(b_id);
    float* C = zilla_buffer_ptr(c_id);
    if (!A || !B || !C) return -1;
    int grid = (n + ELEM_BLOCK - 1) / ELEM_BLOCK;
    div_dev_kernel<<<grid, ELEM_BLOCK>>>(A, B, C, n);
    return 0;
}

int relu_dev(int x_id, int y_id, int n) {
    float* X = zilla_buffer_ptr(x_id);
    float* Y = zilla_buffer_ptr(y_id);
    if (!X || !Y) return -1;
    int grid = (n + ELEM_BLOCK - 1) / ELEM_BLOCK;
    relu_dev_kernel<<<grid, ELEM_BLOCK>>>(X, Y, n);
    return 0;
}

} // extern "C"

#else
#endif /* __CUDACC__ */