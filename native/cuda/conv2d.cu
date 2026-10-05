/*
 * conv2d.cu — CUDA Conv2D forward + backward.
 *
 * Layouts (matching the CPU implementation in native/cpu/):
 *   input   : [C, H, W]
 *   weight  : [outC, C, kH, kW]
 *   bias    : [outC]
 *   output  : [outC, outH, outW]
 *
 *   outH = (H + 2*padding - kH) / stride + 1
 *   outW = (W + 2*padding - kW) / stride + 1
 *
 * All operations work on persistent buffer IDs from buffer_registry.cu.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

float* zilla_buffer_ptr(int id);

#define CONV_TB 256

/* ================================================================
 * Forward
 * ================================================================ */

__global__ void conv2d_fwd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int C, int H, int W,
    int outC, int outH, int outW,
    int kH, int kW,
    int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outC * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int oc = idx / (outW * outH);

    float sum = bias[oc];

    for (int c = 0; c < C; c++) {
        const float* in_c = input + (size_t)c * H * W;
        const float* w_oc = weight + ((size_t)oc * C + c) * kH * kW;

        for (int kh = 0; kh < kH; kh++) {
            int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= H) continue;

            for (int kw = 0; kw < kW; kw++) {
                int iw = ow * stride - padding + kw;
                if (iw < 0 || iw >= W) continue;

                sum += in_c[ih * W + iw] * w_oc[kh * kW + kw];
            }
        }
    }

    output[idx] = sum;
}

int conv2d_fwd_dev(
    int input_id, int weight_id, int bias_id, int output_id,
    int C, int H, int W,
    int outC, int outH, int outW,
    int kH, int kW,
    int stride, int padding
) {
    const float* input  = zilla_buffer_ptr(input_id);
    const float* weight = zilla_buffer_ptr(weight_id);
    const float* bias   = zilla_buffer_ptr(bias_id);
    float* output = zilla_buffer_ptr(output_id);
    if (!input || !weight || !bias || !output) return -1;

    int total = outC * outH * outW;
    int grid  = (total + CONV_TB - 1) / CONV_TB;

    conv2d_fwd_kernel<<<grid, CONV_TB>>>(
        input, weight, bias, output,
        C, H, W, outC, outH, outW,
        kH, kW, stride, padding
    );
    return 0;
}

/* ================================================================
 * Backward — gradient w.r.t. input
 *
 *   dInput[c, ih, iw] = sum over (oc, kh, kw) of
 *       weight[oc, c, kh, kw] * dOutput[oc, oh, ow]
 *   where (oh, ow) satisfies: ih = oh*stride - padding + kh
 * ================================================================ */

__global__ void conv2d_bwd_input_kernel(
    const float* __restrict__ gradOutput,
    const float* __restrict__ weight,
    float* __restrict__ gradInput,
    int C, int H, int W,
    int outC, int outH, int outW,
    int kH, int kW,
    int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx >= total) return;

    int iw = idx % W;
    int ih = (idx / W) % H;
    int c  = idx / (W * H);

    float sum = 0.0f;

    for (int oc = 0; oc < outC; oc++) {
        const float* g_oc = gradOutput + (size_t)oc * outH * outW;
        const float* w_oc = weight + ((size_t)oc * C + c) * kH * kW;

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

                float g = g_oc[oh * outW + ow];
                float w = w_oc[kh * kW + kw];
                sum += g * w;
            }
        }
    }

    gradInput[idx] = sum;
}

/* ================================================================
 * Backward — gradient w.r.t. weight
 *
 *   dWeight[oc, c, kh, kw] = sum over (oh, ow) of
 *       input[c, oh*stride - padding + kh, ow*stride - padding + kw]
 *       * dOutput[oc, oh, ow]
 * ================================================================ */

__global__ void conv2d_bwd_weight_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gradOutput,
    float* __restrict__ gradWeight,
    int C, int H, int W,
    int outC, int outH, int outW,
    int kH, int kW,
    int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outC * C * kH * kW;
    if (idx >= total) return;

    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int c  = (idx / (kW * kH)) % C;
    int oc = idx / (kW * kH * C);

    const float* in_c = input + (size_t)c * H * W;
    const float* g_oc = gradOutput + (size_t)oc * outH * outW;

    float sum = 0.0f;

    for (int oh = 0; oh < outH; oh++) {
        int ih = oh * stride - padding + kh;
        if (ih < 0 || ih >= H) continue;

        for (int ow = 0; ow < outW; ow++) {
            int iw = ow * stride - padding + kw;
            if (iw < 0 || iw >= W) continue;

            float x = in_c[ih * W + iw];
            float g = g_oc[oh * outW + ow];
            sum += x * g;
        }
    }

    gradWeight[idx] = sum;
}

/* ================================================================
 * Backward — gradient w.r.t. bias
 *   dBias[oc] = sum over (oh, ow) of dOutput[oc, oh, ow]
 * ================================================================ */

__global__ void conv2d_bwd_bias_kernel(
    const float* __restrict__ gradOutput,
    float* __restrict__ gradBias,
    int outC, int outH, int outW
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= outC) return;

    const float* g = gradOutput + (size_t)oc * outH * outW;
    float sum = 0.0f;
    int n = outH * outW;
    for (int i = 0; i < n; i++) sum += g[i];
    gradBias[oc] = sum;
}

int conv2d_bwd_dev(
    int input_id, int weight_id, int gradOut_id,
    int gradInput_id, int gradWeight_id, int gradBias_id,
    int C, int H, int W,
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

    // gradInput
    int totalIn = C * H * W;
    int gridIn  = (totalIn + CONV_TB - 1) / CONV_TB;
    conv2d_bwd_input_kernel<<<gridIn, CONV_TB>>>(
        gradOut, weight, gradInput,
        C, H, W, outC, outH, outW,
        kH, kW, stride, padding
    );

    // gradWeight
    int totalW = outC * C * kH * kW;
    int gridW  = (totalW + CONV_TB - 1) / CONV_TB;
    conv2d_bwd_weight_kernel<<<gridW, CONV_TB>>>(
        input, gradOut, gradWeight,
        C, H, W, outC, outH, outW,
        kH, kW, stride, padding
    );

    // gradBias
    int gridB = (outC + CONV_TB - 1) / CONV_TB;
    conv2d_bwd_bias_kernel<<<gridB, CONV_TB>>>(
        gradOut, gradBias, outC, outH, outW
    );

    return 0;
}

} // extern "C"

#else
#endif /* __CUDACC__ */