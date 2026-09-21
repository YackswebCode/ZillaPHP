/*
 * buffer_registry.cu — GPU memory registry for ZillaPHP persistent tensors.
 *
 * Instead of transferring tensors between CPU and GPU on every operation,
 * we keep them resident on the GPU and refer to them by integer handle.
 * PHP stores the handle; the actual device pointer stays here.
 *
 * Each buffer is capped at 1 GB and there are 1024 slots — plenty for
 * any current model.
 */

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

extern "C" {

#define ZILLA_MAX_BUFFERS 1024

typedef struct {
    float* ptr;
    size_t bytes;
    int    in_use;
} zilla_buffer_t;

static zilla_buffer_t g_buffers[ZILLA_MAX_BUFFERS];
static int g_registry_initialized = 0;

static void init_registry(void) {
    if (g_registry_initialized) return;
    memset(g_buffers, 0, sizeof(g_buffers));
    g_registry_initialized = 1;
}

/* Allocate a fresh device buffer of n_floats floats.
 * Returns a positive handle on success, -1 on failure. */
int zilla_buffer_alloc(size_t n_floats) {
    init_registry();
    if (n_floats == 0) return -1;

    size_t bytes = n_floats * sizeof(float);

    for (int i = 1; i < ZILLA_MAX_BUFFERS; i++) {
        if (!g_buffers[i].in_use) {
            float* p = NULL;
            if (cudaMalloc((void**)&p, bytes) != cudaSuccess) {
                fprintf(stderr, "zilla_buffer_alloc: cudaMalloc failed (%zu bytes)\n", bytes);
                cudaGetLastError();  // clear sticky error
                return -1;
            }
            g_buffers[i].ptr    = p;
            g_buffers[i].bytes  = bytes;
            g_buffers[i].in_use = 1;
            return i;
        }
    }
    fprintf(stderr, "zilla_buffer_alloc: no free slots\n");
    return -1;
}

/* Free a buffer. Idempotent: freeing an unused handle is a no-op. */
void zilla_buffer_free(int id) {
    init_registry();
    if (id <= 0 || id >= ZILLA_MAX_BUFFERS) return;
    if (!g_buffers[id].in_use) return;

    cudaFree(g_buffers[id].ptr);
    g_buffers[id].ptr    = NULL;
    g_buffers[id].bytes  = 0;
    g_buffers[id].in_use = 0;
}

/* Upload host floats into an existing device buffer.
 * Returns 0 on success, -1 on failure. */
int zilla_buffer_upload(int id, const float* host, size_t n_floats) {
    init_registry();
    if (id <= 0 || id >= ZILLA_MAX_BUFFERS) return -1;
    if (!g_buffers[id].in_use) return -1;

    size_t bytes = n_floats * sizeof(float);
    if (bytes > g_buffers[id].bytes) {
        fprintf(stderr, "zilla_buffer_upload: %zu bytes exceeds buffer capacity\n", bytes);
        return -1;
    }

    cudaError_t e = cudaMemcpy(g_buffers[id].ptr, host, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        fprintf(stderr, "zilla_buffer_upload: %s\n", cudaGetErrorString(e));
        cudaGetLastError();
        return -1;
    }
    return 0;
}

/* Download device floats into a host buffer.
 * Returns 0 on success, -1 on failure. */
int zilla_buffer_download(int id, float* host, size_t n_floats) {
    init_registry();
    if (id <= 0 || id >= ZILLA_MAX_BUFFERS) return -1;
    if (!g_buffers[id].in_use) return -1;

    size_t bytes = n_floats * sizeof(float);
    if (bytes > g_buffers[id].bytes) return -1;

    cudaError_t e = cudaMemcpy(host, g_buffers[id].ptr, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        fprintf(stderr, "zilla_buffer_download: %s\n", cudaGetErrorString(e));
        cudaGetLastError();
        return -1;
    }
    return 0;
}

/* Synchronize the default stream. Useful for timing. */
void zilla_sync(void) {
    cudaDeviceSynchronize();
}

/* Return the raw device pointer for a handle (for internal use).
 * Returns NULL if the handle is invalid. */
float* zilla_buffer_ptr(int id) {
    init_registry();
    if (id <= 0 || id >= ZILLA_MAX_BUFFERS) return NULL;
    if (!g_buffers[id].in_use) return NULL;
    return g_buffers[id].ptr;
}

} // extern "C"

#else
/* Not compiled with nvcc. */
#endif /* __CUDACC__ */