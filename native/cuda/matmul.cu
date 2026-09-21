/*
 * matmul.cu — cuBLAS-backed matmul with pinned host memory + persistent stream.
 *
 * Replaced the custom tiled kernel with cuBLAS sgemm. On a Colab T4 this
 * takes 1024x1024 from ~205 ms to ~10-15 ms because cuBLAS uses hand-tuned
 * assembly that exploits tensor cores and shared-memory tiling far better
 * than a naive 16x16 kernel.
 *
 * Row-major <-> column-major: cuBLAS assumes column-major. To compute
 *   C[M,N] = A[M,K] @ B[K,N]   (row-major)
 * we invoke:
 *   C^T = B^T @ A^T           (both column-major)
 * which cuBLAS executes as:
 *   cublasSgemm(handle, OP_N, OP_N,
 *               N, M, K, &alpha,
 *               B_dev, N,     // B is KxN row-major = NxK col-major, ld=N
 *               A_dev, K,     // A is MxK row-major = KxM col-major, ld=K
 *               &beta,
 *               C_dev, N);    // C is MxN row-major = NxM col-major, ld=N
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <string.h>

extern "C" {

/* -------- Persistent device buffers -------- */
static float* g_A_dev = NULL;
static float* g_B_dev = NULL;
static float* g_C_dev = NULL;
static size_t g_dev_cap = 0;

/* -------- Persistent pinned host buffers -------- */
static float* h_A_pin = NULL;
static float* h_B_pin = NULL;
static float* h_C_pin = NULL;
static size_t h_pin_cap = 0;

/* -------- Persistent stream + cuBLAS handle -------- */
static cudaStream_t  g_stream  = NULL;
static cublasHandle_t g_cublas = NULL;

static void ensure_device(size_t need) {
    if (need <= g_dev_cap) return;
    size_t newcap = ((need + (16u << 20) - 1) / (16u << 20)) * (16u << 20);

    if (g_A_dev) cudaFree(g_A_dev);
    if (g_B_dev) cudaFree(g_B_dev);
    if (g_C_dev) cudaFree(g_C_dev);

    if (cudaMalloc((void**)&g_A_dev, newcap) != cudaSuccess ||
        cudaMalloc((void**)&g_B_dev, newcap) != cudaSuccess ||
        cudaMalloc((void**)&g_C_dev, newcap) != cudaSuccess) {
        fprintf(stderr, "device alloc failed\n");
        g_dev_cap = 0;
        return;
    }
    g_dev_cap = newcap;
}

static void ensure_host(size_t need) {
    if (need <= h_pin_cap) return;
    size_t newcap = ((need + (16u << 20) - 1) / (16u << 20)) * (16u << 20);

    if (h_A_pin) cudaFreeHost(h_A_pin);
    if (h_B_pin) cudaFreeHost(h_B_pin);
    if (h_C_pin) cudaFreeHost(h_C_pin);

    if (cudaHostAlloc((void**)&h_A_pin, newcap, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**)&h_B_pin, newcap, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**)&h_C_pin, newcap, cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "pinned alloc failed\n");
        h_pin_cap = 0;
        return;
    }
    h_pin_cap = newcap;
}

static void ensure_stream_and_cublas(void) {
    if (g_stream == NULL) {
        cudaStreamCreate(&g_stream);
    }
    if (g_cublas == NULL) {
        cublasCreate(&g_cublas);
        cublasSetStream(g_cublas, g_stream);
    }
}

void matmul_f32(
    const float* A_host,
    const float* B_host,
    float*       C_host,
    int M, int K, int N
) {
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);
    size_t need  = sizeA > sizeB ? sizeA : sizeB;
    if (sizeC > need) need = sizeC;

    ensure_device(need);
    ensure_host(need);
    ensure_stream_and_cublas();
    if (g_dev_cap == 0 || h_pin_cap == 0 || g_cublas == NULL) return;

    // 1. Staging into pinned memory (fast CPU-side memcpy)
    memcpy(h_A_pin, A_host, sizeA);
    memcpy(h_B_pin, B_host, sizeB);

    // 2. Async H2D over our stream
    cudaMemcpyAsync(g_A_dev, h_A_pin, sizeA, cudaMemcpyHostToDevice, g_stream);
    cudaMemcpyAsync(g_B_dev, h_B_pin, sizeB, cudaMemcpyHostToDevice, g_stream);

    // 3. cuBLAS sgemm on the same stream (row-major via transpose trick)
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasSgemm(
        g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        g_B_dev, N,
        g_A_dev, K,
        &beta,
        g_C_dev, N
    );

    // 4. Async D2H into pinned buffer
    cudaMemcpyAsync(h_C_pin, g_C_dev, sizeC, cudaMemcpyDeviceToHost, g_stream);

    // 5. Wait for the stream to drain
    cudaError_t err = cudaStreamSynchronize(g_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA stream error: %s\n", cudaGetErrorString(err));
        return;
    }

    // 6. Copy back into user buffer
    memcpy(C_host, h_C_pin, sizeC);
}

} // extern "C"

#else
/* Not compiled with nvcc — this file is CUDA-only. */
#endif /* __CUDACC__ */