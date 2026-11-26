/* CPU and GPU launch support for the VTI and TTI RTM sources.
   With CPU_REF the sources compile with g++ and the kernels run on the host in double.
   Model files and written results stay little-endian float32.
   GPU builds keep the CUDA or HIP runtime, float arithmetic, and the original launch. */

#ifndef CPU_BACKEND_H
#define CPU_BACKEND_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(CPU_REF)

#include <time.h>
#include <unordered_set>

typedef enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 } cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
} cudaMemcpyKind;

struct rtm_dim3 {
    unsigned int x, y, z;
};

static struct rtm_dim3 threadIdx, blockIdx, blockDim, gridDim;

/* The sources copy one wavefield pointer onto the other and later free both names.
   The second free of an address is skipped. The overwritten allocation is reclaimed when the process exits. */
static std::unordered_set<void *> rtm_live;

static inline const char *cudaGetErrorString(cudaError_t error)
{
    return error == cudaSuccess ? "success" : "cpu runtime error";
}

static inline cudaError_t cudaGetLastError(void)
{
    return cudaSuccess;
}

static inline cudaError_t cudaSetDevice(int device)
{
    (void)device;
    return cudaSuccess;
}

static inline cudaError_t cudaDeviceReset(void)
{
    return cudaSuccess;
}

template <typename T>
static inline cudaError_t cudaMalloc(T **pointer, size_t bytes)
{
    *pointer = (T *)malloc(bytes);
    if (*pointer != NULL)
        rtm_live.insert(*pointer);
    return (*pointer != NULL || bytes == 0) ? cudaSuccess : cudaErrorMemoryAllocation;
}

static inline cudaError_t cudaFree(void *pointer)
{
    if (rtm_live.erase(pointer) > 0)
        free(pointer);
    return cudaSuccess;
}

static inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
    (void)kind;
    if (bytes > 0)
        memcpy(dst, src, bytes);
    return cudaSuccess;
}

static inline cudaError_t cudaMemset(void *pointer, int value, size_t bytes)
{
    memset(pointer, value, bytes);
    return cudaSuccess;
}

#define RTM_GLOBAL
#define RTM_DEVICE
#define RTM_CONSTANT

/* grid and block are the same 1-D bounds as the GPU launch.
   Kernel arguments in this repository are variables, so repeating them per thread is the same call. */
#define RTM_LAUNCH(fn, grid, block, ...)                                      \
    do {                                                                      \
        const int _rtm_grid = (int)(grid);                                    \
        const int _rtm_block = (int)(block);                                  \
        blockDim.x = (unsigned int)_rtm_block;                                \
        blockDim.y = 1;                                                       \
        blockDim.z = 1;                                                       \
        gridDim.x = (unsigned int)_rtm_grid;                                  \
        gridDim.y = 1;                                                        \
        gridDim.z = 1;                                                        \
        for (int _rtm_b = 0; _rtm_b < _rtm_grid; ++_rtm_b) {                   \
            blockIdx.x = (unsigned int)_rtm_b;                                \
            blockIdx.y = 0;                                                   \
            blockIdx.z = 0;                                                   \
            for (int _rtm_t = 0; _rtm_t < _rtm_block; ++_rtm_t) {              \
                threadIdx.x = (unsigned int)_rtm_t;                           \
                threadIdx.y = 0;                                              \
                threadIdx.z = 0;                                              \
                fn(__VA_ARGS__);                                              \
            }                                                                 \
        }                                                                     \
    } while (0)

#elif defined(__HIPCC__)

#include <hip/hip_runtime.h>
#define RTM_GLOBAL __global__
#define RTM_DEVICE __device__
#define RTM_CONSTANT __constant__
#define RTM_LAUNCH(fn, grid, block, ...) fn<<<grid, block>>>(__VA_ARGS__)

#else

#include <cuda_runtime.h>
#define RTM_GLOBAL __global__
#define RTM_DEVICE __device__
#define RTM_CONSTANT __constant__
#define RTM_LAUNCH(fn, grid, block, ...) fn<<<grid, block>>>(__VA_ARGS__)

#endif

/* Last assignment wins. Returns -1 when a value is missing or is not a positive integer. */
static inline int rtm_set_int(int argc, char **argv, const char *key, int *value)
{
    const size_t key_len = strlen(key);
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], key, key_len) != 0 || argv[i][key_len] != '=')
            continue;
        int parsed = 0;
        char tail = '\0';
        if (sscanf(argv[i] + key_len + 1, "%d%c", &parsed, &tail) != 1 || parsed < 1)
            status = -1;
        else
            *value = parsed;
    }
    return status;
}

/* Returns the argv index of the first argument that is not key=value for a known key. */
static inline int rtm_unknown_option(int argc, char **argv, const char **keys)
{
    for (int i = 1; i < argc; ++i) {
        int known = 0;
        for (int k = 0; keys[k] != NULL; ++k) {
            const size_t key_len = strlen(keys[k]);
            if (strncmp(argv[i], keys[k], key_len) == 0 && argv[i][key_len] == '=') {
                known = 1;
                break;
            }
        }
        if (!known)
            return i;
    }
    return 0;
}

/* On-disk samples are float32. CPU_REF arithmetic below uses double. */
typedef float rtm_file_real;

#if defined(CPU_REF)

#include <math.h>

#ifdef sqrtf
#undef sqrtf
#endif
#ifdef cosf
#undef cosf
#endif
#ifdef sinf
#undef sinf
#endif
#ifdef acosf
#undef acosf
#endif

#define float double
#define sqrtf(x) sqrt((x))
#define cosf(x) cos((x))
#define sinf(x) sin((x))
#define acosf(x) acos((x))

static_assert(sizeof(float) == sizeof(double), "CPU reference arithmetic is double");
static_assert(sizeof(rtm_file_real) == 4, "model files stay float32");

#endif

static inline size_t rtm_read_samples(FILE *fp, float *dst, size_t count)
{
#if defined(CPU_REF)
    rtm_file_real chunk[4096];
    size_t done = 0;
    while (done < count) {
        size_t n = count - done;
        if (n > 4096)
            n = 4096;
        n = fread(chunk, sizeof(rtm_file_real), n, fp);
        if (n == 0)
            break;
        for (size_t i = 0; i < n; ++i)
            dst[done + i] = (float)chunk[i];
        done += n;
    }
    return done;
#else
    return fread(dst, sizeof(rtm_file_real), count, fp);
#endif
}

static inline size_t rtm_write_samples(FILE *fp, const float *src, size_t count)
{
#if defined(CPU_REF)
    rtm_file_real chunk[4096];
    size_t done = 0;
    while (done < count) {
        size_t n = count - done;
        if (n > 4096)
            n = 4096;
        for (size_t i = 0; i < n; ++i)
            chunk[i] = (rtm_file_real)src[done + i];
        if (fwrite(chunk, sizeof(rtm_file_real), n, fp) != n)
            break;
        done += n;
    }
    return done;
#else
    return fwrite(src, sizeof(rtm_file_real), count, fp);
#endif
}

#endif
