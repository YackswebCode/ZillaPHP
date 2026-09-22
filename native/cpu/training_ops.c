/*
 * training_ops.c — CPU-side training primitives that operate on
 * persistent buffer handles. Mirrors the CUDA training_ops.cu API
 * so the same PHP code works on either backend.
 *
 * matmul operations use cblas_sgemm from OpenBLAS when available;
 * otherwise they fall back to a straightforward triple loop.
 */

#include <stddef.h>
#include <math.h>
#include <string.h>

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

float* zilla_cpu_buffer_ptr(int id);

/* ================================================================
 * matmul row-major (row × row)
 *   C[M,N] = A[M,K] @ B[K,N]
 * ================================================================ */

int matmul_dev(int a_id, int b_id, int c_id, int M, int K, int N) {
    const float* A = zilla_cpu_buffer_ptr(a_id);
    const float* B = zilla_cpu_buffer_ptr(b_id);
    float* C = zilla_cpu_buffer_ptr(c_id);
    if (!A || !B || !C) return -1;

#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,
                      B, N,
                0.0f, C, N);
#else
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i*K + k] * B[k*N + j];
            C[i*N + j] = s;
        }
#endif
    return 0;
}

/* ================================================================
 * matmul_tn:  C[M,N] = Aᵀ @ B,  A stored [K,M], B stored [K,N]
 * ================================================================ */

int matmul_tn_dev(int a_id, int b_id, int c_id, int M, int K, int N) {
    const float* A = zilla_cpu_buffer_ptr(a_id);   // [K,M]
    const float* B = zilla_cpu_buffer_ptr(b_id);   // [K,N]
    float* C = zilla_cpu_buffer_ptr(c_id);         // [M,N]
    if (!A || !B || !C) return -1;

#ifdef USE_OPENBLAS
    /* Row-major: cblas_sgemm(..., TransA, NoTrans, M, N, K, alpha,
     *                        A, lda = M (columns of A storage),
     *                        B, ldb = N,
     *                        beta, C, ldc = N); */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, M,
                      B, N,
                0.0f, C, N);
#else
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[k*M + i] * B[k*N + j];
            C[i*N + j] = s;
        }
#endif
    return 0;
}

/* ================================================================
 * matmul_nt:  C[M,N] = A @ Bᵀ,  A stored [M,K], B stored [N,K]
 * ================================================================ */

int matmul_nt_dev(int a_id, int b_id, int c_id, int M, int K, int N) {
    const float* A = zilla_cpu_buffer_ptr(a_id);   // [M,K]
    const float* B = zilla_cpu_buffer_ptr(b_id);   // [N,K]
    float* C = zilla_cpu_buffer_ptr(c_id);         // [M,N]
    if (!A || !B || !C) return -1;

#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, A, K,
                      B, K,
                0.0f, C, N);
#else
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i*K + k] * B[j*K + k];
            C[i*N + j] = s;
        }
#endif
    return 0;
}

/* ================================================================
 * add_bias_dev:  X[i,c] += bias[c]
 * ================================================================ */

int add_bias_dev(int x_id, int b_id, int B, int C) {
    float* X = zilla_cpu_buffer_ptr(x_id);
    const float* b = zilla_cpu_buffer_ptr(b_id);
    if (!X || !b) return -1;
    int n = B * C;
    for (int i = 0; i < n; i++) X[i] += b[i % C];
    return 0;
}

/* ================================================================
 * relu_fwd_dev:  Y = max(X,0), mask[i] = (X[i] > 0)
 * ================================================================ */

int relu_fwd_dev(int x_id, int y_id, int mask_id, int n) {
    const float* X = zilla_cpu_buffer_ptr(x_id);
    float* Y = zilla_cpu_buffer_ptr(y_id);
    float* m = zilla_cpu_buffer_ptr(mask_id);
    if (!X || !Y || !m) return -1;
    for (int i = 0; i < n; i++) {
        float v = X[i];
        if (v > 0.0f) { Y[i] = v; m[i] = 1.0f; }
        else          { Y[i] = 0.0f; m[i] = 0.0f; }
    }
    return 0;
}

/* ================================================================
 * relu_bwd_dev:  dX = dY * mask
 * ================================================================ */

int relu_bwd_dev(int dy_id, int mask_id, int dx_id, int n) {
    const float* dY = zilla_cpu_buffer_ptr(dy_id);
    const float* m  = zilla_cpu_buffer_ptr(mask_id);
    float* dX = zilla_cpu_buffer_ptr(dx_id);
    if (!dY || !m || !dX) return -1;
    for (int i = 0; i < n; i++) dX[i] = dY[i] * m[i];
    return 0;
}

/* ================================================================
 * Fused softmax + cross-entropy.
 *
 *   loss += -log(softmax(logits)[i, tgt_i])
 *   dLogits[i,c] = (softmax - onehot) / B
 *
 * Targets are stored as float32 (cast to int inside).
 * Loss is added into *loss_ptr.
 * ================================================================ */

int softmax_ce_dev(int logits_id, int targets_id, int dlogits_id,
                   int loss_id, int B, int C)
{
    const float* logits = zilla_cpu_buffer_ptr(logits_id);
    const float* targets = zilla_cpu_buffer_ptr(targets_id);
    float* dLogits = zilla_cpu_buffer_ptr(dlogits_id);
    float* loss_ptr = zilla_cpu_buffer_ptr(loss_id);
    if (!logits || !targets || !dLogits || !loss_ptr) return -1;

    double total = 0.0;
    for (int i = 0; i < B; i++) {
        const float* row = logits + (size_t)i * C;
        int tgt = (int) targets[i];

        /* max */
        float mx = row[0];
        for (int c = 1; c < C; c++) if (row[c] > mx) mx = row[c];

        /* exp + sum */
        float sum = 0.0f;
        float probs[128];
        if (C > 128) return -2;
        for (int c = 0; c < C; c++) {
            float e = expf(row[c] - mx);
            probs[c] = e;
            sum += e;
        }

        /* loss */
        float p_tgt = probs[tgt] / sum;
        total += -logf(p_tgt);

        /* dLogits */
        float inv = 1.0f / (float)B;
        float* out = dLogits + (size_t)i * C;
        for (int c = 0; c < C; c++) {
            float p = probs[c] / sum;
            float oh = (c == tgt) ? 1.0f : 0.0f;
            out[c] = (p - oh) * inv;
        }
    }

    *loss_ptr += (float) total;
    return 0;
}

/* ================================================================
 * bias_grad_dev:  db[c] = Σ_i dY[i,c]
 * ================================================================ */

int bias_grad_dev(int dy_id, int db_id, int B, int C) {
    const float* dY = zilla_cpu_buffer_ptr(dy_id);
    float* db = zilla_cpu_buffer_ptr(db_id);
    if (!dY || !db) return -1;
    for (int c = 0; c < C; c++) db[c] = 0.0f;
    for (int i = 0; i < B; i++)
        for (int c = 0; c < C; c++)
            db[c] += dY[i*C + c];
    return 0;
}

/* ================================================================
 * sgd_update_dev:  W -= lr * dW
 * ================================================================ */

int sgd_update_dev(int w_id, int dw_id, float lr, int n) {
    float* W = zilla_cpu_buffer_ptr(w_id);
    const float* dW = zilla_cpu_buffer_ptr(dw_id);
    if (!W || !dW) return -1;
    for (int i = 0; i < n; i++) W[i] -= lr * dW[i];
    return 0;
}

/* ================================================================
 * zero_dev
 * ================================================================ */

int zero_dev(int id, int n) {
    float* X = zilla_cpu_buffer_ptr(id);
    if (!X) return -1;
    memset(X, 0, (size_t)n * sizeof(float));
    return 0;
}