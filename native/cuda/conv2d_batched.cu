/*
 * conv2d_batched.cu — CUDA Conv2D for batched [B, C, H, W] tensors.
 *
 * Layouts:
 *   input   : [B, C, H, W]
 *   weight  : [outC, C, kH, kW]      (shared across batch)
 *   bias    : [outC]
 *   output  : [B, outC, outH, outW]
 *   dOut    : [B, outC, outH, outW]
 *   dInput  : [B, C, H, W]
 *   dWeight : [outC, C, kH, kW]      (reduced over batch)
 *   dBias   : [outC]                 (reduced over batch)
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

float* zilla_buffer_ptr(int id);

#define CONV_BTB 256

/* ================================================================
 * Forward — one thread per output element (b, oc, oh, ow)
 * ================================================================ */

__global__ void conv2d_batched_fwd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C, int H, int W,
    int outC, int outH, int outW,
    int kH, int kW,
    int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * outC * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int oc = (idx / (outW * outH)) % outC;
    int b  = idx / (outW * outH * outC);

    const float* in_b = input + (size_t)b * C * H * W;
    float sum = bias[oc];

    for (int c = 0; c < C; c++) {
        const float* in_bc = in_b + (size_t)c * H * W;
        const float* w_oc  = weight + ((size_t)oc * C + c) * kH * kW;
        for (int kh = 0; kh < kH; kh++) {
            int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kW; kw++) {
                int iw = ow * stride - padding + kw;
                if (iw < 0 || iw >= W) continue;
                sum += in_bc[ih * W + iw] * w_oc[kh * kW + kw];
            }
        }
    }

    output[idx] = sum;
}

int conv2d_batched_fwd_dev(
    int input_id, int weight_id, int bias_id, int output_id,
    int B, int C, int H, int W,
    int outC, int outH, int outW,
    int kH, int kW,
    int stride, int padding
) {
    const float* input  = zilla_buffer_ptr(input_id);
    const float* weight = zilla_buffer_ptr(weight_id);
    const float* bias   = zilla_buffer_ptr(bias_id);
    float* output = zilla_buffer_ptr(output_id);
    if (!input || !weight || !bias || !output) return -1;

    int total = B * outC * outH * outW;
    int grid  = (total + CONV_BTB - 1) / CONV_BTB;

    conv2d_batched_fwd_kernel<<<grid, CONV_BTB>>>(
        input, weight, bias, output,
        B, C, H, W, outC, outH, outW, kH, kW, stride, padding
    );
    return 0;
}

/* ================================================================
 * Backward — gradient w.r.t. input
 * ================================================================ */

__global__ void conv2d_batched_bwd_input_kernel(
    const float* __restrict__ gradOutput,   // [B, outC, outH, outW]
    const float* __restrict__ weight,       // [outC, C, kH, kW]
    float* __restrict__ gradInput,          // [B, C, H, W]
    int B, int C, int H, int W,
    int outC, int outH, int outW,
    int kH, int kW,
    int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;

    int iw = idx % W;
    int ih = (idx / W) % H;
    int c  = (idx / (W * H)) % C;
    int b  = idx / (W * H * C);

    float sum = 0.0f;
    const float* g_b = gradOutput + (size_t)b * outC * outH * outW;

    for (int oc = 0; oc < outC; oc++) {
        const float* g_boc = g_b + (size_t)oc * outH * outW;
        const float* w_oc  = weight + ((size_t)oc * C + c) * kH * kW;

        for (int kh = 0; kh < kH; kh++) {
            int oh_num = ih + padding - kh;
            if (oh_num < 0) continue;
            if (oh_num % stride != 0) continue;
            int oh = oh_num / stride;
            if (oh >= outH) continue;

            for (int kw = 0; kw < kW; kw++) {
                int ow_num = iw + padding - kw;
                if (ow_num < 0) continue;
                if (ow_num % stride != 0) continue;
                int ow = ow_num / stride;
                if (ow >= outW) continue;

                sum += g_boc[oh * outW + ow] * w_oc[kh * kW + kw];
            }
        }
    }

    gradInput[idx] = sum;
}

/* ================================================================
 * Backward — gradient w.r.t. weight (reduced over batch)
 *
 *   dWeight[oc, c, kh, kw] = Σ_b Σ_oh Σ_ow
 *       input[b, c, oh*s-p+kh, ow*s-p+kw] * dOut[b, oc, oh, ow]
 *
 * One block per weight element; block threads split the (b, oh, ow) work.
 * ================================================================ */

__global__ void conv2d_batched_bwd_weight_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gradOutput,
    float* __restrict__ gradWeight,
    int B, int C, int H, int W,
    int outC, int outH, int outW,
    int kH, int kW,
    int stride, int padding
) {
    // blockIdx.x = weight index in [outC * C * kH * kW]
    int wIdx = blockIdx.x;
    if (wIdx >= outC * C * kH * kW) return;

    int kw = wIdx % kW;
    int kh = (wIdx / kW) % kH;
    int c  = (wIdx / (kW * kH)) % C;
    int oc = wIdx / (kW * kH * C);

    float localSum = 0.0f;

    for (int b = 0; b < B; b++) {
        const float* in_bc = input + ((size_t)b * C + c) * H * W;
        const float* g_boc = gradOutput + ((size_t)b * outC + oc) * outH * outW;

        for (int oh = 0; oh < outH; oh++) {
            int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= H) continue;
            for (int ow = threadIdx.x; ow < outW; ow += blockDim.x) {
                int iw = ow * stride - padding + kw;
                if (iw < 0 || iw >= W) continue;
                localSum += in_bc[ih * W + iw] * g_boc[oh * outW + ow];
            }
        }
    }

    // Reduce within block via shared memory
    __shared__ float sdata[CONV_BTB];
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) gradWeight[wIdx] = sdata[0];
}

/* ================================================================
 * Backward — gradient w.r.t. bias (reduced over batch and spatial)
 * ================================================================ */

__global__ void conv2d_batched_bwd_bias_kernel(
    const float* __restrict__ gradOutput,
    float* __restrict__ gradBias,
    int B, int outC, int outH, int outW
) {
    int oc = blockIdx.x;
    if (oc >= outC) return;

    float localSum = 0.0f;
    int spatial = outH * outW;

    for (int b = 0; b < B; b++) {
        const float* g = gradOutput + ((size_t)b * outC + oc) * spatial;
        for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
            localSum += g[i];
        }
    }

    __shared__ float sdata[CONV_BTB];
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) gradBias[oc] = sdata[0];
}

int conv2d_batched_bwd_dev(
    int input_id, int weight_id, int gradOut_id,
    int gradInput_id, int gradWeight_id, int gradBias_id,
    int B, int C, int H, int W,
    int outC, int outH, int outW,
    int kH, int kW,
    int stride, int padding
) {
    const float* input    = zilla_buffer_ptr(input_id);
    const float* weight   = zilla_buffer_ptr(weight_id);
    const float* gradOut  = zilla_buffer_ptr(gradOut_id);
    float* gradInput  = zilla_buffer_ptr(gradInput_id);
    float* gradWeight = zilla_buffer_ptr(gradWeight_id);
    float* gradBias   = zilla_buffer_ptr(gradBias_id);
    if (!input || !weight || !gradOut || !gradInput || !gradWeight || !gradBias) return -1;

    // ---- dInput ----
    int totalIn = B * C * H * W;
    int gridIn  = (totalIn + CONV_BTB - 1) / CONV_BTB;
    conv2d_batched_bwd_input_kernel<<<gridIn, CONV_BTB>>>(
        gradOut, weight, gradInput,
        B, C, H, W, outC, outH, outW, kH, kW, stride, padding
    );

    // ---- dWeight: one block per weight element ----
    int totalW = outC * C * kH * kW;
    conv2d_batched_bwd_weight_kernel<<<totalW, CONV_BTB>>>(
        input, gradOut, gradWeight,
        B, C, H, W, outC, outH, outW, kH, kW, stride, padding
    );

    // ---- dBias: one block per output channel ----
    conv2d_batched_bwd_bias_kernel<<<outC, CONV_BTB>>>(
        gradOut, gradBias, B, outC, outH, outW
    );

    return 0;
}

} // extern "C"

#else
#endif /* __CUDACC__ */