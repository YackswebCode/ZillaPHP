/*
 * attention.cu — Transformer attention primitives.
 *
 *   softmax_rows_fwd_dev    row-wise softmax over [N, M]
 *   softmax_rows_bwd_dev    softmax backward (uses saved output)
 *   causal_mask_dev         in-place -inf mask on upper triangle
 *   scale_dev               multiply every element by a scalar
 *
 * These + existing matmul kernels are enough to build full attention
 * forward and backward in PHP.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

extern "C" {

float* zilla_buffer_ptr(int id);

#define ATTN_TB 256

/* ================================================================
 * Row-wise softmax (numerically stable, shared-memory reduction)
 *   Input / output layout: [N, M] row-major
 *   One thread block per row
 * ================================================================ */

__global__ void softmax_rows_fwd_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int M
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* rowIn = x + (size_t)row * M;
    float* rowOut = y + (size_t)row * M;

    __shared__ float smax[ATTN_TB];
    __shared__ float ssum[ATTN_TB];

    // ---- Find max ----
    float localMax = -1e30f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        if (rowIn[i] > localMax) localMax = rowIn[i];
    }
    smax[threadIdx.x] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smax[threadIdx.x + s] > smax[threadIdx.x])
            smax[threadIdx.x] = smax[threadIdx.x + s];
        __syncthreads();
    }
    float mx = smax[0];
    __syncthreads();

    // ---- exp + sum ----
    float localSum = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float e = expf(rowIn[i] - mx);
        rowOut[i] = e;
        localSum += e;
    }
    ssum[threadIdx.x] = localSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    float sum = ssum[0];
    __syncthreads();

    // ---- normalize ----
    float inv = 1.0f / sum;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        rowOut[i] *= inv;
    }
}

int softmax_rows_fwd_dev(int x_id, int y_id, int N, int M) {
    const float* x = zilla_buffer_ptr(x_id);
    float* y = zilla_buffer_ptr(y_id);
    if (!x || !y) return -1;
    softmax_rows_fwd_kernel<<<N, ATTN_TB>>>(x, y, N, M);
    return 0;
}

/* ================================================================
 * Softmax backward
 *   Given saved output P (post-softmax) and dP, computes
 *     dS[i,j] = P[i,j] * (dP[i,j] - sum_k dP[i,k] * P[i,k])
 *   Layouts: all [N, M]
 * ================================================================ */

__global__ void softmax_rows_bwd_kernel(
    const float* __restrict__ P,
    const float* __restrict__ dP,
    float* __restrict__ dS,
    int N, int M
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* pRow  = P  + (size_t)row * M;
    const float* dpRow = dP + (size_t)row * M;
    float* dsRow = dS + (size_t)row * M;

    __shared__ float sdot[ATTN_TB];

    // dot = sum_k dP[i,k] * P[i,k]
    float localDot = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        localDot += dpRow[i] * pRow[i];
    }
    sdot[threadIdx.x] = localDot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdot[threadIdx.x] += sdot[threadIdx.x + s];
        __syncthreads();
    }
    float dot = sdot[0];
    __syncthreads();

    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        dsRow[i] = pRow[i] * (dpRow[i] - dot);
    }
}

int softmax_rows_bwd_dev(int P_id, int dP_id, int dS_id, int N, int M) {
    const float* P  = zilla_buffer_ptr(P_id);
    const float* dP = zilla_buffer_ptr(dP_id);
    float* dS = zilla_buffer_ptr(dS_id);
    if (!P || !dP || !dS) return -1;
    softmax_rows_bwd_kernel<<<N, ATTN_TB>>>(P, dP, dS, N, M);
    return 0;
}

/* ================================================================
 * Causal mask: in-place add -1e9 to positions where col > row.
 *   Layout: [N, M]  (typically N == M for square attention)
 * ================================================================ */

__global__ void causal_mask_kernel(
    float* __restrict__ x,
    int N, int M,
    float maskValue
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= M) return;
    if (col > row) {
        x[(size_t)row * M + col] += maskValue;
    }
}

int causal_mask_dev(int x_id, int N, int M, float maskValue) {
    float* x = zilla_buffer_ptr(x_id);
    if (!x) return -1;
    int gridX = (M + ATTN_TB - 1) / ATTN_TB;
    dim3 grid(gridX, N);
    causal_mask_kernel<<<grid, ATTN_TB>>>(x, N, M, maskValue);
    return 0;
}

/* ================================================================
 * Scale: multiply every element by a scalar (in place)
 *   Used to divide scores by 1/sqrt(d_k).
 * ================================================================ */

__global__ void scale_kernel(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

int scale_dev(int x_id, float s, int n) {
    float* x = zilla_buffer_ptr(x_id);
    if (!x) return -1;
    int grid = (n + ATTN_TB - 1) / ATTN_TB;
    scale_kernel<<<grid, ATTN_TB>>>(x, s, n);
    return 0;
}

/* ================================================================
 * Softmax + causal mask fused (avoids a separate pass)
 *   In-place: applies mask, then softmax over rows.
 * ================================================================ */

__global__ void causal_softmax_kernel(
    float* __restrict__ x,
    int N, int M,
    float maskValue
) {
    int row = blockIdx.x;
    if (row >= N) return;

    float* rowData = x + (size_t)row * M;

    __shared__ float smax[ATTN_TB];
    __shared__ float ssum[ATTN_TB];

    // Mask: set row[i] to maskValue for i > row
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        if (i > row) rowData[i] = maskValue;
    }
    __syncthreads();

    // max
    float localMax = -1e30f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        if (rowData[i] > localMax) localMax = rowData[i];
    }
    smax[threadIdx.x] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smax[threadIdx.x + s] > smax[threadIdx.x])
            smax[threadIdx.x] = smax[threadIdx.x + s];
        __syncthreads();
    }
    float mx = smax[0];
    __syncthreads();

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float e = expf(rowData[i] - mx);
        rowData[i] = e;
        localSum += e;
    }
    ssum[threadIdx.x] = localSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    float sum = ssum[0];
    __syncthreads();

    float inv = 1.0f / sum;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        rowData[i] *= inv;
    }
}

int causal_softmax_dev(int x_id, int N, int M, float maskValue) {
    float* x = zilla_buffer_ptr(x_id);
    if (!x) return -1;
    causal_softmax_kernel<<<N, ATTN_TB>>>(x, N, M, maskValue);
    return 0;
}

} // extern "C"

#else
#endif /* __CUDACC__ */