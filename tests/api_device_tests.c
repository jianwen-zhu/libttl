#include "libttl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ELEMENTS 64
#define STRESS_LAUNCHES 24

static int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fprintf(stderr, "%s:%d: check failed: %s\n",                     \
                    __FILE__, __LINE__, #condition);                            \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

#define EXPECT_STATUS(expression, expected)                                     \
    do {                                                                        \
        const ttl_status_t status_ = (expression);                              \
        if (status_ != (expected)) {                                            \
            fprintf(stderr, "%s:%d: %s returned %d, expected %d: %s\n",       \
                    __FILE__, __LINE__, #expression, (int)status_,              \
                    (int)(expected), ttl_last_error());                         \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

#define EXPECT_OK(expression) EXPECT_STATUS(expression, TTL_STATUS_OK)

static int make_path(
    char *destination, size_t capacity, const char *root, const char *name) {
    const int written = snprintf(destination, capacity, "%s/%s", root, name);
    if (written < 0 || (size_t)written >= capacity) {
        fprintf(stderr, "test path is too long: %s/%s\n", root, name);
        ++failures;
        return 0;
    }
    return 1;
}

static void check_malformed_modules(const char *root) {
    static const char *const names[] = {
        "bad_schema_version",
        "duplicate_tensor",
        "unknown_binding",
        "artifact_traversal",
    };
    char path[4096];
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        ttl_module_t *module = NULL;
        if (!make_path(path, sizeof(path), root, names[i])) return;
        EXPECT_STATUS(ttl_module_load(path, &module), TTL_STATUS_INVALID_ARGUMENT);
        CHECK(module == NULL);
    }
}

static void check_malformed_programs(const char *root) {
    static const char *const names[] = {
        "dead_call",
        "mutable",
        "redefined_value",
        "use_before_definition",
    };
    char path[4096];
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        ttl_program_t *program = NULL;
        if (!make_path(path, sizeof(path), root, names[i])) return;
        EXPECT_STATUS(
            ttl_program_load(path, &program), TTL_STATUS_INVALID_ARGUMENT);
        CHECK(program == NULL);
    }
}

static void check_device_contract(
    ttl_device_t *device, ttl_capability_set_t capabilities) {
    ttl_capability_set_t queried = UINT64_MAX;
    EXPECT_OK(ttl_device_capabilities(device, &queried));
    CHECK(queried == capabilities);
    EXPECT_STATUS(
        ttl_device_capabilities(NULL, &queried), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_device_capabilities(device, NULL), TTL_STATUS_INVALID_ARGUMENT);

#if defined(TTL_TEST_CUDA)
    CHECK(capabilities == TTL_CAPABILITY_KNOWN_MASK);
#else
    CHECK(capabilities == 0);
#endif
}

static void check_tensor_stream_timer(
    ttl_device_t *device, ttl_stream_t *first, ttl_stream_t *second) {
    const int64_t shape[] = {ELEMENTS};
    const int64_t short_shape[] = {ELEMENTS - 1};
    float source[ELEMENTS];
    float result[ELEMENTS];
    for (size_t i = 0; i < ELEMENTS; ++i) source[i] = (float)i - 17.0f;
    memset(result, 0, sizeof(result));

    ttl_tensor_t *host_source = NULL;
    ttl_tensor_t *device_value = NULL;
    ttl_tensor_t *device_direct = NULL;
    ttl_tensor_t *host_result = NULL;
    ttl_tensor_t *wrong_shape = NULL;
    ttl_tensor_t *wrong_dtype = NULL;
    ttl_timer_t *timer = NULL;
    double milliseconds = -1.0;

    EXPECT_STATUS(
        ttl_tensor_empty(NULL, TTL_DTYPE_FLOAT32, 1, shape, &device_value),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, shape, &host_source));
    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_FLOAT32, 1, shape, &device_value));
    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_FLOAT32, 1, shape, &device_direct));
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, shape, &host_result));
    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_FLOAT32, 1, short_shape, &wrong_shape));
    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_INT32, 1, shape, &wrong_dtype));
    EXPECT_OK(ttl_tensor_write(host_source, source, sizeof(source)));

    bool is_host = true;
    EXPECT_OK(ttl_tensor_is_host(device_value, &is_host));
    CHECK(!is_host);
    EXPECT_STATUS(
        ttl_tensor_copy(NULL, host_source, first), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_copy(device_value, NULL, first), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_copy(device_value, host_source, NULL),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_copy(wrong_shape, host_source, first),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_copy(wrong_dtype, host_source, first),
        TTL_STATUS_INVALID_ARGUMENT);

    EXPECT_STATUS(ttl_stream_wait(NULL, first), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(ttl_stream_wait(second, NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(ttl_stream_synchronize(NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(ttl_timer_create(NULL, &timer), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(ttl_timer_create(device, NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_timer_create(device, &timer));
    EXPECT_STATUS(ttl_timer_begin(NULL, second), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(ttl_timer_begin(timer, NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_timer_end(timer, second, NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_timer_end(timer, second, &milliseconds),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_tensor_copy(device_value, host_source, first));
    EXPECT_OK(ttl_stream_wait(second, first));
    EXPECT_OK(ttl_timer_begin(timer, second));
    EXPECT_STATUS(
        ttl_timer_begin(timer, second), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_tensor_copy(host_result, device_value, second));
    EXPECT_OK(ttl_timer_end(timer, second, &milliseconds));
    CHECK(isfinite(milliseconds));
    CHECK(milliseconds >= 0.0);
    EXPECT_OK(ttl_stream_synchronize(second));
    EXPECT_OK(ttl_tensor_read(host_result, result, sizeof(result)));
    CHECK(memcmp(source, result, sizeof(source)) == 0);

    EXPECT_OK(ttl_tensor_write(device_direct, source, sizeof(source)));
    memset(result, 0, sizeof(result));
    EXPECT_OK(ttl_tensor_read(device_direct, result, sizeof(result)));
    CHECK(memcmp(source, result, sizeof(source)) == 0);

    ttl_timer_destroy(timer);
    ttl_tensor_destroy(wrong_dtype);
    ttl_tensor_destroy(wrong_shape);
    ttl_tensor_destroy(host_result);
    ttl_tensor_destroy(device_direct);
    ttl_tensor_destroy(device_value);
    ttl_tensor_destroy(host_source);
}

static void check_device_storage_dtypes(
    ttl_device_t *device, ttl_stream_t *stream) {
    static const ttl_dtype_t dtypes[] = {
        TTL_DTYPE_BOOL,
        TTL_DTYPE_INT8,
        TTL_DTYPE_INT32,
        TTL_DTYPE_FLOAT16,
        TTL_DTYPE_FLOAT32,
        TTL_DTYPE_FLOAT4_E2M1,
        TTL_DTYPE_FLOAT8_E4M3,
        TTL_DTYPE_FLOAT8_E8M0,
    };
    const int64_t shape[] = {3};
    unsigned char source[16] = {0};
    unsigned char result[16] = {0};
    for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); ++i) {
        ttl_tensor_t *host_source = NULL;
        ttl_tensor_t *device_value = NULL;
        ttl_tensor_t *host_result = NULL;
        size_t bytes = 0;
        EXPECT_OK(ttl_dtype_storage_bytes(dtypes[i], 3, &bytes));
        EXPECT_OK(ttl_tensor_empty_host(
            dtypes[i], 1, shape, &host_source));
        EXPECT_OK(ttl_tensor_empty(device, dtypes[i], 1, shape, &device_value));
        EXPECT_OK(ttl_tensor_empty_host(
            dtypes[i], 1, shape, &host_result));
        EXPECT_OK(ttl_tensor_write(host_source, source, bytes));
        EXPECT_OK(ttl_tensor_copy(device_value, host_source, stream));
        EXPECT_OK(ttl_tensor_copy(host_result, device_value, stream));
        EXPECT_OK(ttl_stream_synchronize(stream));
        memset(result, 0xff, sizeof(result));
        EXPECT_OK(ttl_tensor_read(host_result, result, bytes));
        CHECK(memcmp(source, result, bytes) == 0);
        ttl_tensor_destroy(host_result);
        ttl_tensor_destroy(device_value);
        ttl_tensor_destroy(host_source);
    }
}

static void check_binding_kernel(
    const char *module_path, ttl_device_t *device, ttl_stream_t *stream) {
    const int64_t shape[] = {ELEMENTS};
    const int64_t wrong_shape[] = {1};
    float x_values[ELEMENTS];
    float y_values[ELEMENTS];
    float expected_values[ELEMENTS];
    for (size_t i = 0; i < ELEMENTS; ++i) {
        x_values[i] = (float)i;
        y_values[i] = 100.0f - (float)i;
        expected_values[i] = 103.0f + (float)i;
    }

    ttl_module_t *module = NULL;
    ttl_kernel_t *kernel = NULL;
    ttl_tensor_t *output = NULL;
    ttl_tensor_t *x = NULL;
    ttl_tensor_t *y = NULL;
    ttl_tensor_t *host_x = NULL;
    ttl_tensor_t *wrong_extent = NULL;
    ttl_tensor_t *wrong_dtype = NULL;
    ttl_tensor_t *reference = NULL;
    ttl_capability_set_t required = UINT64_MAX;

    EXPECT_STATUS(ttl_module_load(NULL, &module), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_module_load(module_path, NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_module_load("tests/no-such-module", &module),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_module_load(module_path, &module));
    EXPECT_OK(ttl_module_required_capabilities(module, &required));
    CHECK(required == 0);
    EXPECT_STATUS(
        ttl_module_required_capabilities(module, NULL),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_kernel_prepare(NULL, device, &kernel), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_kernel_prepare(module, NULL, &kernel), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_kernel_prepare(module, device, NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_kernel_prepare(module, device, &kernel));
    ttl_module_destroy(module);
    module = NULL;

    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_FLOAT32, 1, shape, &output));
    EXPECT_OK(ttl_tensor_empty(device, TTL_DTYPE_FLOAT32, 1, shape, &x));
    EXPECT_OK(ttl_tensor_empty(device, TTL_DTYPE_FLOAT32, 1, shape, &y));
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, shape, &host_x));
    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_FLOAT32, 1, wrong_shape, &wrong_extent));
    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_INT32, 1, shape, &wrong_dtype));
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, shape, &reference));
    EXPECT_OK(ttl_tensor_write(x, x_values, sizeof(x_values)));
    EXPECT_OK(ttl_tensor_write(y, y_values, sizeof(y_values)));
    EXPECT_OK(ttl_tensor_write(host_x, x_values, sizeof(x_values)));
    EXPECT_OK(ttl_tensor_write(
        reference, expected_values, sizeof(expected_values)));

    ttl_tensor_t *tensors[] = {output, x, y};
    ttl_scalar_t scalars[] = {
        {.dtype = TTL_DTYPE_FLOAT32, .value.float32 = 2.0f},
        {.dtype = TTL_DTYPE_FLOAT32, .value.float32 = 3.0f},
    };
    EXPECT_STATUS(
        ttl_kernel_launch(NULL, tensors, 3, scalars, 2, stream),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_kernel_launch(kernel, tensors, 2, scalars, 2, stream),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_kernel_launch(kernel, NULL, 3, scalars, 2, stream),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_kernel_launch(kernel, tensors, 3, NULL, 2, stream),
        TTL_STATUS_INVALID_ARGUMENT);
    ttl_tensor_t *host_tensors[] = {output, host_x, y};
    EXPECT_STATUS(
        ttl_kernel_launch(kernel, host_tensors, 3, scalars, 2, stream),
        TTL_STATUS_INVALID_ARGUMENT);
    ttl_tensor_t *extent_tensors[] = {output, wrong_extent, y};
    EXPECT_STATUS(
        ttl_kernel_launch(kernel, extent_tensors, 3, scalars, 2, stream),
        TTL_STATUS_INVALID_ARGUMENT);
    ttl_tensor_t *dtype_tensors[] = {output, wrong_dtype, y};
    EXPECT_STATUS(
        ttl_kernel_launch(kernel, dtype_tensors, 3, scalars, 2, stream),
        TTL_STATUS_INVALID_ARGUMENT);
    scalars[0].dtype = TTL_DTYPE_INT32;
    EXPECT_STATUS(
        ttl_kernel_launch(kernel, tensors, 3, scalars, 2, stream),
        TTL_STATUS_INVALID_ARGUMENT);
    scalars[0].dtype = TTL_DTYPE_FLOAT32;

    EXPECT_OK(ttl_kernel_launch(kernel, tensors, 3, scalars, 2, stream));
    EXPECT_OK(ttl_stream_synchronize(stream));
    ttl_close_result_t close;
    EXPECT_OK(ttl_close(output, reference, 0.0, 0.0, &close));
    CHECK(close.is_close);

    ttl_tensor_destroy(reference);
    ttl_tensor_destroy(wrong_dtype);
    ttl_tensor_destroy(wrong_extent);
    ttl_tensor_destroy(host_x);
    ttl_tensor_destroy(y);
    ttl_tensor_destroy(x);
    ttl_tensor_destroy(output);
    ttl_kernel_destroy(kernel);
}

static void check_dynamic_module(
    const char *module_path, ttl_device_t *device, ttl_stream_t *stream) {
    (void)stream;
    ttl_module_t *module = NULL;
    ttl_kernel_t *kernel = NULL;
    ttl_capability_set_t required = 0;
    EXPECT_OK(ttl_module_load(module_path, &module));
    EXPECT_OK(ttl_module_required_capabilities(module, &required));
    CHECK(required == TTL_CAPABILITY_DYNAMIC_WORKGROUP_MEMORY);
#if defined(TTL_TEST_CUDA)
    EXPECT_OK(ttl_kernel_prepare(module, device, &kernel));
    EXPECT_OK(ttl_kernel_launch(kernel, NULL, 0, NULL, 0, stream));
    EXPECT_OK(ttl_stream_synchronize(stream));
#else
    EXPECT_STATUS(
        ttl_kernel_prepare(module, device, &kernel), TTL_STATUS_UNSUPPORTED);
    CHECK(kernel == NULL);
#endif
    ttl_kernel_destroy(kernel);
    ttl_module_destroy(module);
}

static void check_program(
    const char *program_path, const char *malformed_root,
    ttl_device_t *device, ttl_stream_t *stream) {
    const int64_t shape[] = {ELEMENTS};
    const int64_t wrong_shape[] = {1};
    float values[ELEMENTS];
    ttl_program_t *program = NULL;
    ttl_program_execution_t *execution = NULL;
    ttl_tensor_t *input = NULL;
    ttl_tensor_t *output = NULL;
    ttl_tensor_t *host_output = NULL;
    ttl_tensor_t *reference = NULL;
    ttl_tensor_t *wrong_extent = NULL;
    ttl_tensor_t *wrong_dtype = NULL;

    EXPECT_STATUS(ttl_program_load(NULL, &program), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_load(program_path, NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_load("tests/no-such-program", &program),
        TTL_STATUS_INVALID_ARGUMENT);
    check_malformed_programs(malformed_root);
    EXPECT_OK(ttl_program_load(program_path, &program));

    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_FLOAT32, 1, shape, &input));
    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_FLOAT32, 1, shape, &output));
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, shape, &host_output));
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, shape, &reference));
    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_FLOAT32, 1, wrong_shape, &wrong_extent));
    EXPECT_OK(ttl_tensor_empty(
        device, TTL_DTYPE_INT32, 1, shape, &wrong_dtype));

    ttl_program_binding_t bindings[] = {{"x", input}, {"y", output}};
    ttl_program_binding_t duplicate[] = {{"x", input}, {"x", output}};
    ttl_program_binding_t unknown[] = {{"x", input}, {"z", output}};
    ttl_program_binding_t host[] = {{"x", input}, {"y", host_output}};
    ttl_program_binding_t bad_extent[] = {{"x", input}, {"y", wrong_extent}};
    ttl_program_binding_t bad_dtype[] = {{"x", input}, {"y", wrong_dtype}};
    EXPECT_STATUS(
        ttl_program_prepare(NULL, device, bindings, 2, &execution),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_prepare(program, NULL, bindings, 2, &execution),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_prepare(program, device, bindings, 2, NULL),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_prepare(program, device, bindings, 1, &execution),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_prepare(program, device, NULL, 2, &execution),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_prepare(program, device, duplicate, 2, &execution),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_prepare(program, device, unknown, 2, &execution),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_prepare(program, device, host, 2, &execution),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_prepare(program, device, bad_extent, 2, &execution),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_prepare(program, device, bad_dtype, 2, &execution),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_program_prepare(
        program, device, bindings, 2, &execution));
    ttl_program_destroy(program);
    program = NULL;

    EXPECT_STATUS(
        ttl_program_launch(NULL, stream), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_program_launch(execution, NULL), TTL_STATUS_INVALID_ARGUMENT);
    for (int iteration = 0; iteration < 2; ++iteration) {
        for (size_t i = 0; i < ELEMENTS; ++i) {
            values[i] = (float)(iteration * 100) + (float)i * 0.5f;
        }
        EXPECT_OK(ttl_tensor_write(input, values, sizeof(values)));
        EXPECT_OK(ttl_tensor_write(reference, values, sizeof(values)));
        EXPECT_OK(ttl_program_launch(execution, stream));
        EXPECT_OK(ttl_stream_synchronize(stream));
        ttl_close_result_t close;
        EXPECT_OK(ttl_close(output, reference, 0.0, 0.0, &close));
        CHECK(close.is_close);
    }

    ttl_program_execution_destroy(execution);
    ttl_tensor_destroy(wrong_dtype);
    ttl_tensor_destroy(wrong_extent);
    ttl_tensor_destroy(reference);
    ttl_tensor_destroy(host_output);
    ttl_tensor_destroy(output);
    ttl_tensor_destroy(input);
}

#if defined(TTL_TEST_CUDA)
static void check_scalar_device_stress(
    const char *module_path, ttl_device_t *device,
    ttl_stream_t *first, ttl_stream_t *second) {
    const int64_t shape[] = {1};
    ttl_module_t *module = NULL;
    ttl_kernel_t *kernel = NULL;
    ttl_tensor_t *outputs[STRESS_LAUNCHES] = {NULL};
    EXPECT_OK(ttl_module_load(module_path, &module));
    EXPECT_OK(ttl_kernel_prepare(module, device, &kernel));
    ttl_module_destroy(module);

    for (size_t i = 0; i < STRESS_LAUNCHES; ++i) {
        EXPECT_OK(ttl_tensor_empty(
            device, TTL_DTYPE_FLOAT32, 1, shape, &outputs[i]));
        ttl_tensor_t *arguments[] = {outputs[i]};
        const ttl_scalar_t scalar = {
            .dtype = TTL_DTYPE_FLOAT32,
            .value.float32 = (float)i + 0.25f,
        };
        EXPECT_OK(ttl_kernel_launch(
            kernel, arguments, 1, &scalar, 1,
            i % 2 == 0 ? first : second));
    }
    EXPECT_OK(ttl_stream_synchronize(first));
    EXPECT_OK(ttl_stream_synchronize(second));
    for (size_t i = 0; i < STRESS_LAUNCHES; ++i) {
        float result = -1.0f;
        EXPECT_OK(ttl_tensor_read(outputs[i], &result, sizeof(result)));
        CHECK(result == (float)i + 0.25f);
        ttl_tensor_destroy(outputs[i]);
    }
    ttl_kernel_destroy(kernel);
}
#endif

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s MALFORMED_MODULES MALFORMED_PROGRAMS "
                "BINDING_MODULE DYNAMIC_MODULE PROGRAM SCALAR_STRESS\n",
                argv[0]);
        return 2;
    }

    size_t device_count = 0;
    ttl_device_t *device = NULL;
    ttl_device_t *invalid_device = NULL;
    ttl_stream_t *first = NULL;
    ttl_stream_t *second = NULL;
    ttl_capability_set_t capabilities = 0;
    EXPECT_STATUS(ttl_device_count(NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_device_count(&device_count));
    if (device_count == 0) {
        fputs("TTL DEVICE API TESTS REQUIRE AN AVAILABLE ACCELERATOR\n", stderr);
        return 1;
    }
    EXPECT_STATUS(
        ttl_device_open(device_count, &invalid_device), TTL_STATUS_OUT_OF_RANGE);
    CHECK(invalid_device == NULL);
    EXPECT_STATUS(ttl_device_open(0, NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_device_open(0, &device));
    EXPECT_OK(ttl_device_capabilities(device, &capabilities));
    check_device_contract(device, capabilities);
    EXPECT_STATUS(ttl_stream_create(NULL, &first), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(ttl_stream_create(device, NULL), TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_OK(ttl_stream_create(device, &first));
    EXPECT_OK(ttl_stream_create(device, &second));

    check_malformed_modules(argv[1]);
    check_tensor_stream_timer(device, first, second);
    check_device_storage_dtypes(device, first);
    check_binding_kernel(argv[3], device, first);
    check_dynamic_module(argv[4], device, first);
    check_program(argv[5], argv[2], device, first);
#if defined(TTL_TEST_CUDA)
    check_scalar_device_stress(argv[6], device, first, second);
#endif

    EXPECT_OK(ttl_stream_synchronize(first));
    EXPECT_OK(ttl_stream_synchronize(second));
    ttl_stream_destroy(second);
    ttl_stream_destroy(first);
    ttl_device_destroy(device);

    if (failures != 0) {
        fprintf(stderr, "TTL DEVICE API TESTS FAILED: %d failure(s)\n", failures);
        return 1;
    }
    puts("TTL DEVICE API TESTS PASSED");
    return 0;
}
