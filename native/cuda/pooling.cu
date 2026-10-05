/*
 * pooling.cu — CUDA MaxPool2D forward + backward.
 *
 * Layout:
 *   input  : [C, H, W]
 *   output : [C, outH, outW]
 *   argmax : int32 flat index into input (across all channels)
 *
 *   outH = (H - kH) / stride + 1
 *   outW = (W - kW) / stride + 1
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

float* zilla_buffer_ptr(int id);

#define POOL_TB 256

/* ================================================================
 * Forward — one thread per output element
 * ================================================================ */

__global__ void maxpool2d_fwd_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ argmax,
    int C, int H, int W,
    int outH, int outW,
    int kH, int kW, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c  = idx / (outW * outH);

    const float* in_c = input + (size_t)c * H * W;
    int base_flat = c * H * W;

    float best = -1e30f;
    int bestFlat = base_flat;

    for (int kh = 0; kh < kH; kh++) {
        int ih = oh * stride + kh;
        for (int kw = 0; kw < kW; kw++) {
            int iw = ow * stride + kw;
            int local = ih * W + iw;
            float v = in_c[local];
            if (v > best) {
                best = v;
                bestFlat = base_flat + local;
            }
        }
    }

    output[idx] = best;
    argmax[idx] = bestFlat;
}

int maxpool2d_fwd_dev(
    int input_id, int output_id, int argmax_id,
    int C, int H, int W,
    int outH, int outW,
    int kH, int kW, int stride
) {
    const float* input = zilla_buffer_ptr(input_id);
    float* output = zilla_buffer_ptr(output_id);
    int* argmax   = (int*) zilla_buffer_ptr(argmax_id);
    if (!input || !output || !argmax) return -1;

    int total = C * outH * outW;
    int grid  = (total + POOL_TB - 1) / POOL_TB;

    maxpool2d_fwd_kernel<<<grid, POOL_TB>>>(
        input, output, argmax,
        C, H, W, outH, outW, kH, kW, stride
    );
    return 0;
}

/* ================================================================
 * Backward — scatter gradient from argmax positions
 *
 * Caller must zero gradInput first (use zero_dev).
 * ================================================================ */

__global__ void maxpool2d_bwd_kernel(
    const float* __restrict__ gradOutput,
    const int* __restrict__ argmax,
    float* __restrict__ gradInput,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int target = argmax[idx];
    atomicAdd(&gradInput[target], gradOutput[idx]);
}

int maxpool2d_bwd_dev(
    int gradOut_id, int argmax_id, int gradInput_id,
    int C, int outH, int outW
) {
    const float* gradOut = zilla_buffer_ptr(gradOut_id);
    const int* argmax    = (const int*) zilla_buffer_ptr(argmax_id);
    float* gradInput = zilla_buffer_ptr(gradInput_id);
    if (!gradOut || !argmax || !gradInput) return -1;

    int total = C * outH * outW;
    int grid  = (total + POOL_TB - 1) / POOL_TB;

    // NOTE: gradInput must be zeroed by the caller before this kernel.
    maxpool2d_bwd_kernel<<<grid, POOL_TB>>>(
        gradOut, argmax, gradInput, total
    );
    return 0;
}

} // extern "C"

#else
#endif /* __CUDACC__ */