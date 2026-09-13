#include "libttl.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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

static void check_dtypes(void) {
    static const struct {
        ttl_dtype_t dtype;
        const char *name;
        size_t bits;
    } cases[] = {
        {TTL_DTYPE_BOOL, "bool", 8},
        {TTL_DTYPE_INT8, "int8", 8},
        {TTL_DTYPE_INT32, "int32", 32},
        {TTL_DTYPE_FLOAT16, "float16", 16},
        {TTL_DTYPE_FLOAT32, "float32", 32},
        {TTL_DTYPE_FLOAT4_E2M1, "float4_e2m1", 4},
        {TTL_DTYPE_FLOAT8_E4M3, "float8_e4m3", 8},
        {TTL_DTYPE_FLOAT8_E8M0, "float8_e8m0", 8},
    };

    CHECK(ttl_api_version() == TTL_API_VERSION);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        size_t bytes = SIZE_MAX;
        CHECK(strcmp(ttl_dtype_name(cases[i].dtype), cases[i].name) == 0);
        CHECK(ttl_dtype_bit_width(cases[i].dtype) == cases[i].bits);
        EXPECT_OK(ttl_dtype_storage_bytes(cases[i].dtype, 3, &bytes));
        CHECK(bytes == (cases[i].bits * 3 + 7) / 8);
    }

    const ttl_dtype_t invalid = (ttl_dtype_t)99;
    CHECK(strcmp(ttl_dtype_name(invalid), "invalid") == 0);
    CHECK(ttl_dtype_bit_width(invalid) == 0);
    size_t bytes = 0;
    EXPECT_STATUS(
        ttl_dtype_storage_bytes(invalid, 1, &bytes),
        TTL_STATUS_INVALID_ARGUMENT);
    CHECK(ttl_last_error()[0] != '\0');
    EXPECT_STATUS(
        ttl_dtype_storage_bytes(TTL_DTYPE_FLOAT32, 1, NULL),
        TTL_STATUS_INVALID_ARGUMENT);
}

static void check_host_tensors(void) {
    const int64_t shape[] = {2, 3};
    const float source[] = {1.0f, -2.0f, 3.5f, 4.0f, 5.0f, -6.0f};
    float destination[6] = {0};
    ttl_tensor_t *tensor = NULL;
    EXPECT_OK(ttl_tensor_empty_host(TTL_DTYPE_FLOAT32, 2, shape, &tensor));

    size_t rank = 0;
    int64_t extent = 0;
    int64_t stride = 0;
    ttl_dtype_t dtype = TTL_DTYPE_BOOL;
    size_t bytes = 0;
    bool value = false;
    EXPECT_OK(ttl_tensor_rank(tensor, &rank));
    CHECK(rank == 2);
    EXPECT_OK(ttl_tensor_extent(tensor, 0, &extent));
    CHECK(extent == 2);
    EXPECT_OK(ttl_tensor_extent(tensor, 1, &extent));
    CHECK(extent == 3);
    EXPECT_OK(ttl_tensor_stride(tensor, 0, &stride));
    CHECK(stride == 3);
    EXPECT_OK(ttl_tensor_stride(tensor, 1, &stride));
    CHECK(stride == 1);
    EXPECT_OK(ttl_tensor_dtype(tensor, &dtype));
    CHECK(dtype == TTL_DTYPE_FLOAT32);
    EXPECT_OK(ttl_tensor_byte_size(tensor, &bytes));
    CHECK(bytes == sizeof(source));
    EXPECT_OK(ttl_tensor_is_host(tensor, &value));
    CHECK(value);
    EXPECT_OK(ttl_tensor_is_contiguous(tensor, &value));
    CHECK(value);

    EXPECT_OK(ttl_tensor_write(tensor, source, sizeof(source)));
    EXPECT_OK(ttl_tensor_read(tensor, destination, sizeof(destination)));
    CHECK(memcmp(source, destination, sizeof(source)) == 0);
    EXPECT_STATUS(
        ttl_tensor_extent(tensor, 2, &extent), TTL_STATUS_OUT_OF_RANGE);
    CHECK(ttl_last_error()[0] != '\0');
    EXPECT_OK(ttl_tensor_rank(tensor, &rank));
    CHECK(ttl_last_error()[0] == '\0');
    EXPECT_STATUS(
        ttl_tensor_write(tensor, source, sizeof(source) - 1),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_read(tensor, destination, sizeof(destination) - 1),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_write(tensor, NULL, sizeof(source)),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_read(tensor, NULL, sizeof(destination)),
        TTL_STATUS_INVALID_ARGUMENT);
    ttl_tensor_destroy(tensor);

    tensor = NULL;
    EXPECT_OK(ttl_tensor_empty_host(TTL_DTYPE_INT32, 0, NULL, &tensor));
    EXPECT_OK(ttl_tensor_rank(tensor, &rank));
    CHECK(rank == 0);
    EXPECT_OK(ttl_tensor_byte_size(tensor, &bytes));
    CHECK(bytes == sizeof(int32_t));
    ttl_tensor_destroy(tensor);

    const int64_t empty_shape[] = {2, 0, 3};
    tensor = NULL;
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT16, 3, empty_shape, &tensor));
    EXPECT_OK(ttl_tensor_byte_size(tensor, &bytes));
    CHECK(bytes == 0);
    ttl_tensor_destroy(tensor);

    const int64_t negative_shape[] = {-1};
    EXPECT_STATUS(
        ttl_tensor_empty_host(
            TTL_DTYPE_FLOAT32, 1, negative_shape, &tensor),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_empty_host(TTL_DTYPE_FLOAT32, 1, NULL, &tensor),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_empty_host((ttl_dtype_t)99, 0, NULL, &tensor),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_tensor_empty_host(TTL_DTYPE_FLOAT32, 0, NULL, NULL),
        TTL_STATUS_INVALID_ARGUMENT);

    const int64_t fp4_shape[] = {3};
    const unsigned char valid_fp4[] = {0x21, 0x03};
    const unsigned char invalid_fp4[] = {0x21, 0xf3};
    unsigned char fp4_result[2] = {0};
    tensor = NULL;
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT4_E2M1, 1, fp4_shape, &tensor));
    EXPECT_OK(ttl_tensor_write(tensor, valid_fp4, sizeof(valid_fp4)));
    EXPECT_OK(ttl_tensor_read(tensor, fp4_result, sizeof(fp4_result)));
    CHECK(memcmp(valid_fp4, fp4_result, sizeof(valid_fp4)) == 0);
    EXPECT_STATUS(
        ttl_tensor_write(tensor, invalid_fp4, sizeof(invalid_fp4)),
        TTL_STATUS_INVALID_ARGUMENT);
    ttl_tensor_destroy(tensor);
}

static void check_dtype_comparisons(void) {
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
    unsigned char zeros[16] = {0};
    for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); ++i) {
        ttl_tensor_t *tensor = NULL;
        size_t bytes = 0;
        ttl_close_result_t result;
        EXPECT_OK(ttl_tensor_empty_host(dtypes[i], 1, shape, &tensor));
        EXPECT_OK(ttl_tensor_byte_size(tensor, &bytes));
        EXPECT_OK(ttl_tensor_write(tensor, zeros, bytes));
        EXPECT_OK(ttl_close(tensor, tensor, 0.0, 0.0, &result));
        CHECK(result.is_close);
        CHECK(result.mismatch_count == 0);
        CHECK(result.worst_linear_index == SIZE_MAX);
        ttl_tensor_destroy(tensor);
    }
}

static void check_close(void) {
    const int64_t shape[] = {3};
    const int64_t other_shape[] = {2};
    const float actual_values[] = {1.0f, 2.5f, 5.0f};
    const float reference_values[] = {1.0f, 2.0f, 3.0f};
    const int32_t integer_values[] = {1, 2, 3};
    ttl_tensor_t *actual = NULL;
    ttl_tensor_t *reference = NULL;
    ttl_tensor_t *integer = NULL;
    ttl_tensor_t *other = NULL;
    ttl_close_result_t result;

    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, shape, &actual));
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, shape, &reference));
    EXPECT_OK(ttl_tensor_empty_host(TTL_DTYPE_INT32, 1, shape, &integer));
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, other_shape, &other));
    EXPECT_OK(ttl_tensor_write(actual, actual_values, sizeof(actual_values)));
    EXPECT_OK(ttl_tensor_write(
        reference, reference_values, sizeof(reference_values)));
    EXPECT_OK(ttl_tensor_write(integer, integer_values, sizeof(integer_values)));

    EXPECT_OK(ttl_close(actual, reference, 0.0, 0.0, &result));
    CHECK(!result.is_close);
    CHECK(result.mismatch_count == 2);
    CHECK(result.worst_linear_index == 2);
    CHECK(fabs(result.max_absolute_error - 2.0) < 1e-12);
    EXPECT_OK(ttl_close(actual, reference, 0.0, 2.0, &result));
    CHECK(result.is_close);
    EXPECT_OK(ttl_close(integer, reference, 0.0, 0.0, &result));
    CHECK(result.is_close);
    EXPECT_STATUS(
        ttl_close(actual, other, 0.0, 0.0, &result),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_close(actual, reference, -1.0, 0.0, &result),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_close(NULL, reference, 0.0, 0.0, &result),
        TTL_STATUS_INVALID_ARGUMENT);
    EXPECT_STATUS(
        ttl_close(actual, reference, 0.0, 0.0, NULL),
        TTL_STATUS_INVALID_ARGUMENT);

    const int64_t special_shape[] = {2};
    const float special_values[] = {NAN, INFINITY};
    ttl_tensor_t *special = NULL;
    EXPECT_OK(ttl_tensor_empty_host(
        TTL_DTYPE_FLOAT32, 1, special_shape, &special));
    EXPECT_OK(ttl_tensor_write(
        special, special_values, sizeof(special_values)));
    EXPECT_OK(ttl_close(special, special, 0.0, 0.0, &result));
    CHECK(!result.is_close);
    CHECK(result.mismatch_count == 1);
    CHECK(result.worst_linear_index == 0);

    ttl_tensor_destroy(special);
    ttl_tensor_destroy(other);
    ttl_tensor_destroy(integer);
    ttl_tensor_destroy(reference);
    ttl_tensor_destroy(actual);
}

static void check_null_destruction(void) {
    ttl_device_destroy(NULL);
    ttl_tensor_destroy(NULL);
    ttl_stream_destroy(NULL);
    ttl_timer_destroy(NULL);
    ttl_module_destroy(NULL);
    ttl_kernel_destroy(NULL);
    ttl_program_destroy(NULL);
    ttl_program_execution_destroy(NULL);
}

int main(void) {
    check_dtypes();
    check_host_tensors();
    check_dtype_comparisons();
    check_close();
    check_null_destruction();
    if (failures != 0) {
        fprintf(stderr, "TTL HOST API TESTS FAILED: %d failure(s)\n", failures);
        return 1;
    }
    puts("TTL HOST API TESTS PASSED");
    return 0;
}
