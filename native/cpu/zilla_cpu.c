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

/* ==================================================================
 * Conv2D (forward + backward)
 *
 *   input  : [C, H, W]
 *   weight : [outC, C, kH, kW]
 *   bias   : [outC]
 *   output : [outC, outH, outW]
 *
 *   outH = (H + 2*padding - kH) / stride + 1
 *   outW = (W + 2*padding - kW) / stride + 1
 * ================================================================== */

void conv2d_forward_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int C, int H, int W,
    int outC, int kH, int kW,
    int stride, int padding
) {
    int outH = (H + 2 * padding - kH) / stride + 1;
    int outW = (W + 2 * padding - kW) / stride + 1;

    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float sum = bias[oc];
                for (int c = 0; c < C; c++) {
                    const float* inBase = input + (size_t)c * H * W;
                    const float* wBase  = weight + ((size_t)oc * C + c) * kH * kW;
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * stride - padding + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * stride - padding + kw;
                            if (iw < 0 || iw >= W) continue;
                            sum += inBase[ih * W + iw] * wBase[kh * kW + kw];
                        }
                    }
                }
                output[((size_t)oc * outH + oh) * outW + ow] = sum;
            }
        }
    }
}

void conv2d_backward_f32(
    const float* input,
    const float* weight,
    const float* gradOutput,
    float* gradInput,
    float* gradWeight,
    float* gradBias,
    int C, int H, int W,
    int outC, int kH, int kW,
    int stride, int padding
) {
    int outH = (H + 2 * padding - kH) / stride + 1;
    int outW = (W + 2 * padding - kW) / stride + 1;

    /* zero gradInput and gradWeight */
    for (size_t i = 0; i < (size_t)C * H * W; i++) gradInput[i] = 0.0f;
    for (size_t i = 0; i < (size_t)outC * C * kH * kW; i++) gradWeight[i] = 0.0f;
    for (int i = 0; i < outC; i++) gradBias[i] = 0.0f;

    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float g = gradOutput[((size_t)oc * outH + oh) * outW + ow];
                gradBias[oc] += g;

                for (int c = 0; c < C; c++) {
                    const float* inBase = input + (size_t)c * H * W;
                    float* giBase = gradInput + (size_t)c * H * W;
                    float* gwBase = gradWeight + ((size_t)oc * C + c) * kH * kW;

                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * stride - padding + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * stride - padding + kw;
                            if (iw < 0 || iw >= W) continue;

                            size_t iIdx = (size_t)ih * W + iw;
                            size_t wIdx = (size_t)kh * kW + kw;

                            giBase[iIdx]  += g * weight[((size_t)oc * C + c) * kH * kW + wIdx];
                            gwBase[wIdx]  += g * inBase[iIdx];
                        }
                    }
                }
            }
        }
    }
}

/* ==================================================================
 * MaxPool2D (forward + backward)
 * ================================================================== */

void maxpool2d_forward_f32(
    const float* input,
    float* output,
    int* argmax,
    int C, int H, int W,
    int kH, int kW, int stride
) {
    int outH = (H - kH) / stride + 1;
    int outW = (W - kW) / stride + 1;

    for (int c = 0; c < C; c++) {
        const float* inBase = input + (size_t)c * H * W;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float best = -1e30f;
                int bestIdx = 0;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        size_t idx = (size_t)ih * W + iw;
                        float v = inBase[idx];
                        if (v > best) { best = v; bestIdx = (int)idx; }
                    }
                }
                size_t oIdx = ((size_t)c * outH + oh) * outW + ow;
                output[oIdx] = best;
                argmax[oIdx] = bestIdx;
            }
        }
    }
}

void maxpool2d_backward_f32(
    const float* gradOutput,
    const int* argmax,
    float* gradInput,
    int C, int H, int W,
    int outH, int outW
) {
    for (size_t i = 0; i < (size_t)C * H * W; i++) gradInput[i] = 0.0f;

    for (int c = 0; c < C; c++) {
        float* giBase = gradInput + (size_t)c * H * W;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                size_t oIdx = ((size_t)c * outH + oh) * outW + ow;
                giBase[argmax[oIdx]] += gradOutput[oIdx];
            }
        }
    }
}