#ifndef LIBTTL_H
#define LIBTTL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TTL_API_VERSION_MAJOR 1u
#define TTL_API_VERSION_MINOR 3u
#define TTL_API_VERSION_PATCH 0u
#define TTL_API_VERSION_ENCODE(major, minor, patch) \
    (((uint32_t)(major) << 24) | ((uint32_t)(minor) << 16) | (uint32_t)(patch))
#define TTL_API_VERSION TTL_API_VERSION_ENCODE( \
    TTL_API_VERSION_MAJOR, TTL_API_VERSION_MINOR, TTL_API_VERSION_PATCH)

typedef enum ttl_status {
    TTL_STATUS_OK = 0,
    TTL_STATUS_INVALID_ARGUMENT = 1,
    TTL_STATUS_OUT_OF_RANGE = 2,
    TTL_STATUS_UNSUPPORTED = 3,
    TTL_STATUS_BACKEND_ERROR = 4
} ttl_status_t;

typedef enum ttl_dtype {
    TTL_DTYPE_BOOL = 0,
    TTL_DTYPE_INT8 = 1,
    TTL_DTYPE_INT32 = 2,
    TTL_DTYPE_FLOAT16 = 3,
    TTL_DTYPE_FLOAT32 = 4,
    TTL_DTYPE_FLOAT4_E2M1 = 5,
    TTL_DTYPE_FLOAT8_E4M3 = 6,
    TTL_DTYPE_FLOAT8_E8M0 = 7
} ttl_dtype_t;

typedef uint64_t ttl_capability_set_t;

/* Dynamic storage shared by one workgroup (CUDA block, Metal threadgroup). */
#define TTL_CAPABILITY_DYNAMIC_WORKGROUP_MEMORY (UINT64_C(1) << 0)
/* Distinct TTL stream handles can use independently scheduled native queues. */
#define TTL_CAPABILITY_INDEPENDENT_STREAMS (UINT64_C(1) << 1)
/* The backend can instantiate a final-kernel DAG as one native graph. */
#define TTL_CAPABILITY_NATIVE_GRAPH_EXECUTION (UINT64_C(1) << 2)
#define TTL_CAPABILITY_KNOWN_MASK \
    (TTL_CAPABILITY_DYNAMIC_WORKGROUP_MEMORY | \
     TTL_CAPABILITY_INDEPENDENT_STREAMS | \
     TTL_CAPABILITY_NATIVE_GRAPH_EXECUTION)

typedef struct ttl_device ttl_device_t;
typedef struct ttl_tensor ttl_tensor_t;
typedef struct ttl_stream ttl_stream_t;
typedef struct ttl_timer ttl_timer_t;
typedef struct ttl_module ttl_module_t;
typedef struct ttl_kernel ttl_kernel_t;
typedef struct ttl_program ttl_program_t;
typedef struct ttl_program_execution ttl_program_execution_t;

typedef struct ttl_close_result {
    bool is_close;
    size_t mismatch_count;
    size_t worst_linear_index;
    double max_absolute_error;
    double max_relative_error;
} ttl_close_result_t;

typedef struct ttl_scalar {
    ttl_dtype_t dtype;
    union {
        int32_t int32;
        float float32;
    } value;
} ttl_scalar_t;

typedef struct ttl_program_binding {
    const char *name;
    ttl_tensor_t *tensor;
} ttl_program_binding_t;

/* Error text is owned by libttl until the next call on the same thread. */
uint32_t ttl_api_version(void);
const char *ttl_last_error(void);
const char *ttl_dtype_name(ttl_dtype_t dtype);
size_t ttl_dtype_bit_width(ttl_dtype_t dtype);
ttl_status_t ttl_dtype_storage_bytes(
    ttl_dtype_t dtype, size_t element_count, size_t *bytes);

/* Devices are accelerator-only. Host storage has no device handle. */
ttl_status_t ttl_device_count(size_t *count);
ttl_status_t ttl_device_open(size_t position, ttl_device_t **device);
ttl_status_t ttl_device_capabilities(
    const ttl_device_t *device, ttl_capability_set_t *capabilities);
void ttl_device_destroy(ttl_device_t *device);

ttl_status_t ttl_tensor_empty_host(
    ttl_dtype_t dtype, size_t rank, const int64_t *shape,
    ttl_tensor_t **tensor);
ttl_status_t ttl_tensor_empty(
    const ttl_device_t *device, ttl_dtype_t dtype, size_t rank,
    const int64_t *shape, ttl_tensor_t **tensor);
void ttl_tensor_destroy(ttl_tensor_t *tensor);

/* Read and write are synchronous host-boundary operations. Packed float4
 * stores even linear elements in the low nibble and odd elements in the high
 * nibble. The unused high nibble of an odd-sized tensor must be zero. */
ttl_status_t ttl_tensor_write(
    ttl_tensor_t *tensor, const void *source, size_t bytes);
ttl_status_t ttl_tensor_read(
    const ttl_tensor_t *tensor, void *destination, size_t bytes);
ttl_status_t ttl_tensor_copy(
    ttl_tensor_t *destination, const ttl_tensor_t *source,
    ttl_stream_t *stream);

ttl_status_t ttl_tensor_rank(const ttl_tensor_t *tensor, size_t *rank);
ttl_status_t ttl_tensor_extent(
    const ttl_tensor_t *tensor, size_t dimension, int64_t *extent);
ttl_status_t ttl_tensor_stride(
    const ttl_tensor_t *tensor, size_t dimension, int64_t *stride);
ttl_status_t ttl_tensor_dtype(
    const ttl_tensor_t *tensor, ttl_dtype_t *dtype);
ttl_status_t ttl_tensor_byte_size(
    const ttl_tensor_t *tensor, size_t *bytes);
ttl_status_t ttl_tensor_is_host(
    const ttl_tensor_t *tensor, bool *is_host);
ttl_status_t ttl_tensor_is_contiguous(
    const ttl_tensor_t *tensor, bool *is_contiguous);

/* Verification-only: copies values to the host and compares them on the CPU. */
ttl_status_t ttl_close(
    const ttl_tensor_t *actual,
    const ttl_tensor_t *reference,
    double relative_tolerance,
    double absolute_tolerance,
    ttl_close_result_t *result);

/* libttl uses explicit streams and has no ambient current-stream state. */
ttl_status_t ttl_stream_create(
    const ttl_device_t *device, ttl_stream_t **stream);
ttl_status_t ttl_stream_wait(
    ttl_stream_t *waiting, const ttl_stream_t *dependency);
ttl_status_t ttl_stream_synchronize(ttl_stream_t *stream);
void ttl_stream_destroy(ttl_stream_t *stream);

/* Measures a completed interval of work submitted to one explicit stream. */
ttl_status_t ttl_timer_create(
    const ttl_device_t *device, ttl_timer_t **timer);
ttl_status_t ttl_timer_begin(ttl_timer_t *timer, ttl_stream_t *stream);
ttl_status_t ttl_timer_end(
    ttl_timer_t *timer, ttl_stream_t *stream, double *milliseconds);
void ttl_timer_destroy(ttl_timer_t *timer);

/* A kernel-module directory contains module.json, private launch.json, and a
 * backend-native device artifact. */
ttl_status_t ttl_module_load(const char *path, ttl_module_t **module);
ttl_status_t ttl_module_required_capabilities(
    const ttl_module_t *module, ttl_capability_set_t *capabilities);
void ttl_module_destroy(ttl_module_t *module);

ttl_status_t ttl_kernel_prepare(
    const ttl_module_t *module,
    const ttl_device_t *device,
    ttl_kernel_t **kernel);
void ttl_kernel_destroy(ttl_kernel_t *kernel);

ttl_status_t ttl_kernel_launch(
    ttl_kernel_t *kernel,
    ttl_tensor_t *const *tensors,
    size_t tensor_count,
    const ttl_scalar_t *scalars,
    size_t scalar_count,
    ttl_stream_t *stream);

/* A program directory contains program.json and one final kernel package per
 * referenced function under functions/FUNCTION. */
ttl_status_t ttl_program_load(const char *path, ttl_program_t **program);
void ttl_program_destroy(ttl_program_t *program);

ttl_status_t ttl_program_prepare(
    const ttl_program_t *program,
    const ttl_device_t *device,
    const ttl_program_binding_t *bindings,
    size_t binding_count,
    ttl_program_execution_t **execution);
void ttl_program_execution_destroy(ttl_program_execution_t *execution);

ttl_status_t ttl_program_launch(
    ttl_program_execution_t *execution, ttl_stream_t *stream);

#ifdef __cplusplus
}
#endif

#endif
