/*
 * audio.c — Native audio kernels for ZillaPHP.
 *
 * Currently provides:
 *   - Hann window generation
 *   - Radix-2 FFT (in-place, iterative Cooley-Tukey)
 *   - STFT magnitude
 *
 * The mel filterbank and log() are applied in PHP because the mel
 * matmul reuses the existing OpenBLAS-backed matmul_f32, and the
 * resulting [nMels, nFrames] tensor is small.
 *
 * Build: see build.sh — this file is compiled into libzilla_cpu.so.
 */

#include <stddef.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ==================================================================
 * Hann window
 *
 *   w[n] = 0.5 - 0.5 * cos(2 * pi * n / (N - 1))
 * ================================================================== */

void hann_window_f32(float* w, int N) {
    if (N <= 1) {
        if (N == 1) w[0] = 1.0f;
        return;
    }
    for (int n = 0; n < N; n++) {
        w[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / (float)(N - 1));
    }
}

/* ==================================================================
 * Iterative radix-2 FFT (in-place, Cooley-Tukey).
 *
 * N must be a power of two. O(N log N).
 * ================================================================== */

static void bit_reverse_permute(float* re, float* im, int N) {
    int j = 0;
    for (int i = 0; i < N - 1; i++) {
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
        int m = N >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

void fft_radix2_f32(float* re, float* im, int N) {
    bit_reverse_permute(re, im, N);

    for (int len = 2; len <= N; len <<= 1) {
        int half = len >> 1;
        float angleStep = -2.0f * (float)M_PI / (float)len;
        float wReStep = cosf(angleStep);
        float wImStep = sinf(angleStep);

        for (int i = 0; i < N; i += len) {
            float wRe = 1.0f;
            float wIm = 0.0f;
            for (int j = 0; j < half; j++) {
                int a = i + j;
                int b = a + half;

                /* Complex multiply:  t = W * x[b]
                 *   tRe = wRe*re[b] - wIm*im[b]
                 *   tIm = wRe*im[b] + wIm*re[b]     <-- fixed
                 */
                float tRe = wRe * re[b] - wIm * im[b];
                float tIm = wRe * im[b] + wIm * re[b];

                re[b] = re[a] - tRe;
                im[b] = im[a] - tIm;
                re[a] += tRe;
                im[a] += tIm;

                float newWRe = wRe * wReStep - wIm * wImStep;
                float newWIm = wRe * wImStep + wIm * wReStep;
                wRe = newWRe;
                wIm = newWIm;
            }
        }
    }
}

/* ==================================================================
 * STFT magnitude.
 *
 *   input       : [inputLen]        audio samples, normalized to [-1, 1]
 *   window      : [fftSize]         pre-computed Hann window
 *   fftSize     : must be a power of 2
 *   hopSize     : usually fftSize / 4
 *   output      : [nBins, nFrames]  real magnitude spectrum, row-major
 *                                   output[bin * nFrames + frame]
 *   outNBins    : set to fftSize/2 + 1
 *   outNFrames  : set to computed frame count
 *
 * Zero-pads the final frame if inputLen isn't a multiple of hopSize.
 * ================================================================== */

void stft_magnitude_f32(
    const float* input, int inputLen,
    const float* window, int fftSize, int hopSize,
    float* output, int* outNBins, int* outNFrames
) {
    if (fftSize < 2 || (fftSize & (fftSize - 1)) != 0) {
        /* fftSize must be a power of two */
        *outNBins = 0; *outNFrames = 0;
        return;
    }
    if (fftSize > 16384) {
        /* hard cap to avoid stack overflow on scratch buffers */
        *outNBins = 0; *outNFrames = 0;
        return;
    }

    int nBins = fftSize / 2 + 1;

    int nFrames = (inputLen < fftSize)
        ? 1
        : 1 + (inputLen - fftSize) / hopSize;
    if (nFrames <= 0) nFrames = 1;

    /* Scratch — sized to the max fftSize we allow */
    static float re[16384];
    static float im[16384];

    for (int f = 0; f < nFrames; f++) {
        int start = f * hopSize;

        for (int i = 0; i < fftSize; i++) {
            int idx = start + i;
            float s = (idx < inputLen) ? input[idx] : 0.0f;
            re[i] = s * window[i];
            im[i] = 0.0f;
        }

        fft_radix2_f32(re, im, fftSize);

        for (int b = 0; b < nBins; b++) {
            output[b * nFrames + f] = sqrtf(re[b] * re[b] + im[b] * im[b]);
        }
    }

    *outNBins = nBins;
    *outNFrames = nFrames;
}