#include <libttl.h>

#include <stdio.h>
#include <stdlib.h>

static void require(ttl_status_t status, const char *operation) {
    if (status == TTL_STATUS_OK) return;
    fprintf(stderr, "%s: %s\n", operation, ttl_last_error());
    exit(EXIT_FAILURE);
}

int main(void) {
    const int64_t shape[] = {4};
    const float actual_values[] = {1.0f, 2.0f, 3.0f, 4.00001f};
    const float reference_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ttl_tensor_t *actual = NULL;
    ttl_tensor_t *reference = NULL;
    ttl_close_result_t result = {0};

    require(ttl_tensor_empty_host(
                TTL_DTYPE_FLOAT32, 1, shape, &actual),
            "create actual tensor");
    require(ttl_tensor_empty_host(
                TTL_DTYPE_FLOAT32, 1, shape, &reference),
            "create reference tensor");
    require(ttl_tensor_write(
                actual, actual_values, sizeof(actual_values)),
            "write actual tensor");
    require(ttl_tensor_write(
                reference, reference_values, sizeof(reference_values)),
            "write reference tensor");
    require(ttl_close(actual, reference, 1e-4, 1e-5, &result),
            "compare tensors");

    printf("close=%s mismatches=%zu max_abs_error=%.8g\n",
           result.is_close ? "true" : "false",
           result.mismatch_count,
           result.max_absolute_error);

    ttl_tensor_destroy(reference);
    ttl_tensor_destroy(actual);
    return result.is_close ? EXIT_SUCCESS : EXIT_FAILURE;
}
