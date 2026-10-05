/*
 * pooling_batched.cu — CUDA MaxPool2D for batched [B, C, H, W] tensors.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

float* zilla_buffer_ptr(int id);

#define POOL_BTB 256

/* ================================================================
 * Forward — one thread per output element (b, c, oh, ow)
 * argmax[b,c,oh,ow] = flat index into input[b, :, :, :]
 * ================================================================ */

__global__ void maxpool2d_batched_fwd_kernel(
    const float* __restrict__ input,   // [B, C, H, W]
    float* __restrict__ output,        // [B, C, outH, outW]
    int* __restrict__ argmax,          // [B, C, outH, outW]
    int B, int C, int H, int W,
    int outH, int outW,
    int kH, int kW, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c  = (idx / (outW * outH)) % C;
    int b  = idx / (outW * outH * C);

    const float* in_bc = input + ((size_t)b * C + c) * H * W;
    int base = (size_t)b * C * H * W + c * H * W;

    float best = -1e30f;
    int bestFlat = base;

    for (int kh = 0; kh < kH; kh++) {
        int ih = oh * stride + kh;
        for (int kw = 0; kw < kW; kw++) {
            int iw = ow * stride + kw;
            int local = ih * W + iw;
            float v = in_bc[local];
            if (v > best) {
                best = v;
                bestFlat = base + local;
            }
        }
    }

    output[idx] = best;
    argmax[idx] = bestFlat;
}

int maxpool2d_batched_fwd_dev(
    int input_id, int output_id, int argmax_id,
    int B, int C, int H, int W,
    int outH, int outW,
    int kH, int kW, int stride
) {
    const float* input = zilla_buffer_ptr(input_id);
    float* output = zilla_buffer_ptr(output_id);
    int* argmax   = (int*) zilla_buffer_ptr(argmax_id);
    if (!input || !output || !argmax) return -1;

    int total = B * C * outH * outW;
    int grid  = (total + POOL_BTB - 1) / POOL_BTB;

    maxpool2d_batched_fwd_kernel<<<grid, POOL_BTB>>>(
        input, output, argmax,
        B, C, H, W, outH, outW, kH, kW, stride
    );
    return 0;
}

/* ================================================================
 * Backward — scatter gradient from argmax positions.
 * Caller must zero gradInput first.
 * ================================================================ */

__global__ void maxpool2d_batched_bwd_kernel(
    const float* __restrict__ gradOutput,   // [B, C, outH, outW]
    const int* __restrict__ argmax,         // [B, C, outH, outW]
    float* __restrict__ gradInput,          // [B, C, H, W]
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int target = argmax[idx];
    atomicAdd(&gradInput[target], gradOutput[idx]);
}

int maxpool2d_batched_bwd_dev(
    int gradOut_id, int argmax_id, int gradInput_id,
    int B, int C, int outH, int outW
) {
    const float* gradOut = zilla_buffer_ptr(gradOut_id);
    const int* argmax    = (const int*) zilla_buffer_ptr(argmax_id);
    float* gradInput = zilla_buffer_ptr(gradInput_id);
    if (!gradOut || !argmax || !gradInput) return -1;

    int total = B * C * outH * outW;
    int grid  = (total + POOL_BTB - 1) / POOL_BTB;

    maxpool2d_batched_bwd_kernel<<<grid, POOL_BTB>>>(
        gradOut, argmax, gradInput, total
    );
    return 0;
}

} // extern "C"

#else
#endif /* __CUDACC__ */