/*
 * training_ops.cu — device-resident training primitives.
 *
 * All operations work on buffer IDs from buffer_registry.cu.
 * No host round-trips inside any operation.
 *
 * Softmax-CE targets are stored as float32 (0.0 .. C-1.0) to avoid
 * needing a separate int32 buffer type in the registry. The kernel
 * casts on the fly.
 *
 * Row-major cuBLAS conventions used here:
 *
 *   matmul      C[M,N] = A[M,K] @ B[K,N]     (existing, device_ops.cu)
 *   matmul_tn   C[M,N] = Aᵀ[M,K] @ B[K,N]    A stored [K,M]
 *   matmul_nt   C[M,N] = A[M,K] @ Bᵀ[K,N]    B stored [N,K]
 *
 * Derivation for matmul_tn (row-major with cuBLAS col-major mapping):
 *   Let A_storage be [K,M] row-major = A_cm [M,K] col-major.
 *   We want C[i,j] = Σ_k A_stored[k,i] * B[k,j].
 *   C_cm[j,i] = Σ_k B_cm[j,k] * A_cm[i,k]
 *   → C_cm = B_cm @ A_cmᵀ
 *   → cublasSgemm(OP_N, OP_T, N, M, K, ... B_ptr, N, A_ptr, M, ... C_ptr, N)
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>

extern "C" {

float* zilla_buffer_ptr(int id);

#define TB 256

/* ================================================================
 * Bias broadcast add (in place):  X[i, c] += bias[c]
 * ================================================================ */

__global__ void add_bias_kernel(float* X, const float* b, int n, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    X[i] += b[i % C];
}

int add_bias_dev(int x_id, int b_id, int B, int C) {
    float* X = zilla_buffer_ptr(x_id);
    const float* bias = zilla_buffer_ptr(b_id);
    if (!X || !bias) return -1;
    int n = B * C;
    int grid = (n + TB - 1) / TB;
    add_bias_kernel<<<grid, TB>>>(X, bias, n, C);
    return 0;
}

/* ================================================================
 * ReLU forward with saved mask (for backward)
 * ================================================================ */

__global__ void relu_fwd_kernel(const float* X, float* Y, float* mask, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = X[i];
    if (v > 0.0f) { Y[i] = v; mask[i] = 1.0f; }
    else          { Y[i] = 0.0f; mask[i] = 0.0f; }
}

int relu_fwd_dev(int x_id, int y_id, int mask_id, int n) {
    const float* X = zilla_buffer_ptr(x_id);
    float* Y = zilla_buffer_ptr(y_id);
    float* m = zilla_buffer_ptr(mask_id);
    if (!X || !Y || !m) return -1;
    int grid = (n + TB - 1) / TB;
    relu_fwd_kernel<<<grid, TB>>>(X, Y, m, n);
    return 0;
}

/* ================================================================
 * ReLU backward:  dX[i] = dY[i] * mask[i]
 * ================================================================ */

__global__ void relu_bwd_kernel(const float* dY, const float* mask, float* dX, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dX[i] = dY[i] * mask[i];
}

int relu_bwd_dev(int dy_id, int mask_id, int dx_id, int n) {
    const float* dY = zilla_buffer_ptr(dy_id);
    const float* m  = zilla_buffer_ptr(mask_id);
    float* dX = zilla_buffer_ptr(dx_id);
    if (!dY || !m || !dX) return -1;
    int grid = (n + TB - 1) / TB;
    relu_bwd_kernel<<<grid, TB>>>(dY, m, dX, n);
    return 0;
}

/* ================================================================
 * Fused softmax + cross-entropy (forward + dLogits)
 *
 * One thread block per sample. Uses shared memory for probs.
 * Assumes C <= 128 (fits in shared mem easily).
 *
 * targets stored as float32 (cast to int32 inside).
 * Loss atomically accumulated into *outLoss.
 * ================================================================ */

__global__ void softmax_ce_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ targets_f,
    float* __restrict__ dLogits,
    float* __restrict__ outLoss,
    int B, int C
) {
    int i = blockIdx.x;
    if (i >= B) return;

    extern __shared__ float probs[];

    // Single-thread computation (C is small, e.g. 10). Simpler + correct.
    if (threadIdx.x == 0) {
        const float* row = logits + (size_t)i * C;

        // max
        float mx = row[0];
        for (int c = 1; c < C; c++) if (row[c] > mx) mx = row[c];

        // exp + sum
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            float e = expf(row[c] - mx);
            probs[c] = e;
            sum += e;
        }

        // loss
        int tgt = (int) targets_f[i];
        float p_tgt = probs[tgt] / sum;
        atomicAdd(outLoss, -logf(p_tgt));

        // dLogits
        float inv = 1.0f / (float)B;
        float* out = dLogits + (size_t)i * C;
        for (int c = 0; c < C; c++) {
            float p = probs[c] / sum;
            float oh = (c == tgt) ? 1.0f : 0.0f;
            out[c] = (p - oh) * inv;
        }
    }
}

int softmax_ce_dev(int logits_id, int targets_id, int dlogits_id, int loss_id, int B, int C) {
    const float* logits = zilla_buffer_ptr(logits_id);
    const float* targets = zilla_buffer_ptr(targets_id);
    float* dLogits = zilla_buffer_ptr(dlogits_id);
    float* loss = zilla_buffer_ptr(loss_id);
    if (!logits || !targets || !dLogits || !loss) return -1;

    softmax_ce_kernel<<<B, 1, C * sizeof(float)>>>(logits, targets, dLogits, loss, B, C);
    return 0;
}

/* ================================================================
 * Bias gradient:  db[c] = Σ_i dY[i, c]
 * ================================================================ */

__global__ void bias_grad_kernel(const float* dY, float* db, int B, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s = 0.0f;
    for (int i = 0; i < B; i++) s += dY[(size_t)i * C + c];
    db[c] = s;
}

int bias_grad_dev(int dy_id, int db_id, int B, int C) {
    const float* dY = zilla_buffer_ptr(dy_id);
    float* db = zilla_buffer_ptr(db_id);
    if (!dY || !db) return -1;
    int grid = (C + TB - 1) / TB;
    bias_grad_kernel<<<grid, TB>>>(dY, db, B, C);
    return 0;
}

/* ================================================================
 * SGD update:  W -= lr * dW     (in place)
 * ================================================================ */

__global__ void sgd_kernel(float* W, const float* dW, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    W[i] -= lr * dW[i];
}

int sgd_update_dev(int w_id, int dw_id, float lr, int n) {
    float* W = zilla_buffer_ptr(w_id);
    const float* dW = zilla_buffer_ptr(dw_id);
    if (!W || !dW) return -1;
    int grid = (n + TB - 1) / TB;
    sgd_kernel<<<grid, TB>>>(W, dW, lr, n);
    return 0;
}

/* ================================================================
 * Zero a buffer
 * ================================================================ */

__global__ void zero_kernel(float* X, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) X[i] = 0.0f;
}

int zero_dev(int id, int n) {
    float* X = zilla_buffer_ptr(id);
    if (!X) return -1;
    int grid = (n + TB - 1) / TB;
    zero_kernel<<<grid, TB>>>(X, n);
    return 0;
}

/* ================================================================
 * matmul with A transposed:  C = Aᵀ @ B
 *   A stored [K, M], B stored [K, N], C stored [M, N]
 * ================================================================ */

static cublasHandle_t get_h(void) {
    static cublasHandle_t h = NULL;
    if (h == NULL) cublasCreate(&h);
    return h;
}

int matmul_tn_dev(int a_id, int b_id, int c_id, int M, int K, int N) {
    float* A = zilla_buffer_ptr(a_id);
    float* B = zilla_buffer_ptr(b_id);
    float* C = zilla_buffer_ptr(c_id);
    if (!A || !B || !C) return -1;

    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasSgemm(
        get_h(),
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        B, N,
        A, M,
        &beta,
        C, N
    );
    return (st == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

/* ================================================================
 * matmul with B transposed:  C = A @ Bᵀ
 *   A stored [M, K], B stored [N, K], C stored [M, N]
 * ================================================================ */

int matmul_nt_dev(int a_id, int b_id, int c_id, int M, int K, int N) {
    float* A = zilla_buffer_ptr(a_id);
    float* B = zilla_buffer_ptr(b_id);
    float* C = zilla_buffer_ptr(c_id);
    if (!A || !B || !C) return -1;

    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasSgemm(
        get_h(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, K,
        A, K,
        &beta,
        C, N
    );
    return (st == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

/* ================================================================
 * D2D copy
 * ================================================================ */

int copy_dev(int src_id, int dst_id, int n) {
    float* s = zilla_buffer_ptr(src_id);
    float* d = zilla_buffer_ptr(dst_id);
    if (!s || !d) return -1;
    cudaError_t e = cudaMemcpy(d, s, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice);
    return (e == cudaSuccess) ? 0 : -1;
}

} // extern "C"

#else
#endif /* __CUDACC__ */