/*
 * audio.cu — CUDA audio kernels: Hann window + STFT via cuFFT.
 *
 * Batches all frames of a waveform into a single cuFFT R2C transform.
 * For 1-second 16 kHz audio with FFT=512, HOP=160, that's 97 frames
 * in one batched call.
 *
 * Persistent buffers + persistent cuFFT plan keep launch overhead low.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

/* ---------------- Hann window ---------------- */

__global__ void hann_window_kernel(float* w, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (N <= 1) { w[i] = (N == 1) ? 1.0f : 0.0f; return; }
    const float pi = 3.14159265358979323846f;
    w[i] = 0.5f - 0.5f * cosf(2.0f * pi * (float)i / (float)(N - 1));
}

void hann_window_cuda(float* w_host, int N) {
    float* d_w = nullptr;
    if (cudaMalloc((void**)&d_w, N * sizeof(float)) != cudaSuccess) return;
    int block = 256, grid = (N + block - 1) / block;
    hann_window_kernel<<<grid, block>>>(d_w, N);
    cudaDeviceSynchronize();
    cudaMemcpy(w_host, d_w, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_w);
}

/* ---------------- Frame + window ---------------- */

__global__ void frame_extract_kernel(
    const float* __restrict__ input, int inputLen,
    const float* __restrict__ window, int fftSize, int hopSize,
    float* __restrict__ framed, int nFrames
) {
    int f = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFrames || i >= fftSize) return;
    int idx = f * hopSize + i;
    float s = (idx < inputLen) ? input[idx] : 0.0f;
    framed[f * fftSize + i] = s * window[i];
}

/* ---------------- Magnitude ---------------- */

__global__ void magnitude_kernel(
    const cufftComplex* __restrict__ spec,
    float* __restrict__ mag,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float re = spec[i].x, im = spec[i].y;
    mag[i] = sqrtf(re * re + im * im);
}

/* ---------------- Persistent state ---------------- */

static float*        d_framed   = nullptr;
static float*        d_window   = nullptr;
static cufftComplex* d_spectrum = nullptr;
static float*        d_magnitude= nullptr;
static size_t        g_framed_cap = 0;
static size_t        g_spec_cap   = 0;
static int           g_plan_fftSize = 0;
static int           g_plan_nFrames = 0;
static cufftHandle   g_plan = 0;

static void ensure_framed(size_t need) {
    if (need <= g_framed_cap) return;
    if (d_framed) cudaFree(d_framed);
    if (cudaMalloc((void**)&d_framed, need) != cudaSuccess) { g_framed_cap = 0; return; }
    g_framed_cap = need;
}

static void ensure_window(int fftSize) {
    if (d_window) cudaFree(d_window);
    if (cudaMalloc((void**)&d_window, fftSize * sizeof(float)) != cudaSuccess) {
        d_window = nullptr; return;
    }
    int block = 256, grid = (fftSize + block - 1) / block;
    hann_window_kernel<<<grid, block>>>(d_window, fftSize);
}

static void ensure_spectrum(int nFrames, int fftSize) {
    int nBins = fftSize / 2 + 1;
    size_t needComplex = (size_t)nFrames * nBins * sizeof(cufftComplex);
    size_t needMag     = (size_t)nFrames * nBins * sizeof(float);

    if (needComplex > g_spec_cap) {
        if (d_spectrum) cudaFree(d_spectrum);
        if (cudaMalloc((void**)&d_spectrum, needComplex) != cudaSuccess) {
            g_spec_cap = 0; return;
        }
        g_spec_cap = needComplex;
    }
    if (d_magnitude) cudaFree(d_magnitude);
    cudaMalloc((void**)&d_magnitude, needMag);
}

static void ensure_plan(int fftSize, int nFrames) {
    if (g_plan != 0 && g_plan_fftSize == fftSize && g_plan_nFrames == nFrames) return;
    if (g_plan != 0) cufftDestroy(g_plan);
    if (cufftPlan1d(&g_plan, fftSize, CUFFT_R2C, nFrames) != CUFFT_SUCCESS) {
        g_plan = 0; return;
    }
    g_plan_fftSize = fftSize;
    g_plan_nFrames = nFrames;
}

/* ---------------- Public entry point ---------------- */

void stft_magnitude_cuda(
    const float* input, int inputLen,
    const float* /*window_host_unused*/,
    int fftSize, int hopSize,
    float* output, int* outNBins, int* outNFrames
) {
    if (fftSize < 2 || (fftSize & (fftSize - 1)) != 0 || fftSize > 16384) {
        *outNBins = 0; *outNFrames = 0;
        return;
    }

    int nBins   = fftSize / 2 + 1;
    int nFrames = (inputLen < fftSize) ? 1 : 1 + (inputLen - fftSize) / hopSize;
    if (nFrames <= 0) nFrames = 1;

    size_t framedSize = (size_t)nFrames * fftSize * sizeof(float);
    ensure_framed(framedSize);
    ensure_window(fftSize);
    ensure_spectrum(nFrames, fftSize);
    ensure_plan(fftSize, nFrames);
    if (g_framed_cap == 0 || d_window == nullptr || g_spec_cap == 0 || g_plan == 0) {
        *outNBins = 0; *outNFrames = 0;
        return;
    }

    // Input → device
    float* d_input = nullptr;
    cudaMalloc((void**)&d_input, inputLen * sizeof(float));
    cudaMemcpy(d_input, input, inputLen * sizeof(float), cudaMemcpyHostToDevice);

    // Frame + window
    {
        int block = 256;
        dim3 grid((fftSize + block - 1) / block, nFrames);
        frame_extract_kernel<<<grid, block>>>(
            d_input, inputLen, d_window, fftSize, hopSize, d_framed, nFrames
        );
    }

    // Batched cuFFT R2C
    cufftExecR2C(g_plan, d_framed, d_spectrum);

    // Magnitude
    {
        int total = nFrames * nBins;
        int block = 256, grid = (total + block - 1) / block;
        magnitude_kernel<<<grid, block>>>(d_spectrum, d_magnitude, total);
    }
    cudaDeviceSynchronize();

    // Copy out and transpose [nFrames, nBins] → [nBins, nFrames]
    float* h_mag = (float*)malloc((size_t)nFrames * nBins * sizeof(float));
    cudaMemcpy(h_mag, d_magnitude, (size_t)nFrames * nBins * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int f = 0; f < nFrames; f++)
        for (int b = 0; b < nBins; b++)
            output[b * nFrames + f] = h_mag[f * nBins + b];

    free(h_mag);
    cudaFree(d_input);

    *outNBins   = nBins;
    *outNFrames = nFrames;
}

} // extern "C"

#else
#endif /* __CUDACC__ */