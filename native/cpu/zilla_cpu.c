/*
 * zilla_cpu.c — Native CPU kernels for ZillaPHP.
 *
 * Compile with -DUSE_OPENBLAS to link against cblas_sgemm. Falls back
 * to a naive triple-loop matmul when OpenBLAS is not available.
 *
 * Build:
 *   ./build.sh
 */

#include <stddef.h>
#include <math.h>

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

/* ==================================================================
 * matmul_f32
 *
 *   C[M,N] = A[M,K] @ B[K,N]     (row-major)
 *
 * Uses cblas_sgemm when OpenBLAS is available. Otherwise falls back
 * to a cache-friendly triple loop.
 * ================================================================== */

void matmul_f32(const float* A, const float* B, float* C, int M, int K, int N) {

#ifdef USE_OPENBLAS
    /*
     * cblas_sgemm computes:
     *     C := alpha * op(A) * op(B) + beta * C
     *
     * For row-major matrices A[M,K] and B[K,N] producing C[M,N]:
     *     alpha = 1.0, beta = 0.0
     *     lda   = K   (leading dim of A = number of columns)
     *     ldb   = N
     *     ldc   = N
     */
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0f, A, K,
              B, N,
        0.0f, C, N
    );

#else
    /* Naive fallback (still works, just slower). */
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    for (int i = 0; i < M; i++) {
        const float* Ai = A + (size_t)i * K;
        float*       Ci = C + (size_t)i * N;
        for (int k = 0; k < K; k++) {
            float aik = Ai[k];
            const float* Bk = B + (size_t)k * N;
            for (int j = 0; j < N; j++) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
#endif
}

/* ==================================================================
 * Element-wise kernels
 * ================================================================== */

void relu_f32(const float* X, float* Y, int N) {
    for (int i = 0; i < N; i++) Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
}

void add_f32(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}

void sub_f32(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] - B[i];
}

void mul_f32(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] * B[i];
}

void div_f32(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] / B[i];
}

/* ==================================================================
 * Reductions
 * ================================================================== */

float sum_f32(const float* X, int N) {
    float s = 0.0f;
    for (int i = 0; i < N; i++) s += X[i];
    return s;
}