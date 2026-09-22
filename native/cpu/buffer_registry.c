/*
 * buffer_registry.c — CPU-side persistent float buffers.
 *
 * Same idea as the CUDA version: PHP holds integer handles, C holds
 * the actual malloc'd pointers. This eliminates repeated toC/fromC
 * conversions during training loops.
 *
 * Buffers are plain heap memory. No alignment tricks. OpenBLAS will
 * use whatever SIMD path it needs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ZILLA_MAX_BUFFERS 1024

typedef struct {
    float*  ptr;
    size_t  n_floats;
    int     in_use;
} zilla_cpu_buffer_t;

static zilla_cpu_buffer_t g_buffers[ZILLA_MAX_BUFFERS];
static int g_registry_initialized = 0;

static void ensure_init(void) {
    if (g_registry_initialized) return;
    memset(g_buffers, 0, sizeof(g_buffers));
    g_registry_initialized = 1;
}

int zilla_cpu_buffer_alloc(size_t n_floats) {
    ensure_init();
    if (n_floats == 0) return -1;

    for (int i = 1; i < ZILLA_MAX_BUFFERS; i++) {
        if (!g_buffers[i].in_use) {
            float* p = (float*) aligned_alloc(64, ((n_floats * sizeof(float)) + 63) & ~63UL);
            if (!p) {
                fprintf(stderr, "zilla_cpu_buffer_alloc: malloc failed (%zu floats)\n", n_floats);
                return -1;
            }
            g_buffers[i].ptr = p;
            g_buffers[i].n_floats = n_floats;
            g_buffers[i].in_use = 1;
            return i;
        }
    }
    fprintf(stderr, "zilla_cpu_buffer_alloc: no free slots\n");
    return -1;
}

void zilla_cpu_buffer_free(int id) {
    ensure_init();
    if (id <= 0 || id >= ZILLA_MAX_BUFFERS) return;
    if (!g_buffers[id].in_use) return;
    free(g_buffers[id].ptr);
    g_buffers[id].ptr = NULL;
    g_buffers[id].n_floats = 0;
    g_buffers[id].in_use = 0;
}

int zilla_cpu_buffer_upload(int id, const float* host, size_t n_floats) {
    ensure_init();
    if (id <= 0 || id >= ZILLA_MAX_BUFFERS) return -1;
    if (!g_buffers[id].in_use) return -1;
    if (n_floats > g_buffers[id].n_floats) return -1;

    memcpy(g_buffers[id].ptr, host, n_floats * sizeof(float));
    return 0;
}

int zilla_cpu_buffer_download(int id, float* host, size_t n_floats) {
    ensure_init();
    if (id <= 0 || id >= ZILLA_MAX_BUFFERS) return -1;
    if (!g_buffers[id].in_use) return -1;
    if (n_floats > g_buffers[id].n_floats) return -1;

    memcpy(host, g_buffers[id].ptr, n_floats * sizeof(float));
    return 0;
}

float* zilla_cpu_buffer_ptr(int id) {
    ensure_init();
    if (id <= 0 || id >= ZILLA_MAX_BUFFERS) return NULL;
    if (!g_buffers[id].in_use) return NULL;
    return g_buffers[id].ptr;
}