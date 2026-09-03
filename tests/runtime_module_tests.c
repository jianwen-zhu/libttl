#include "libttl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int expect_invalid(const char *root, const char *fixture) {
    char path[4096];
    ttl_module_t *module = NULL;
    const int written = snprintf(path, sizeof(path), "%s/%s", root, fixture);
    if (written < 0 || (size_t)written >= sizeof(path)) {
        fprintf(stderr, "fixture path is too long\n");
        return 1;
    }
    const ttl_status_t status = ttl_module_load(path, &module);
    if (status == TTL_STATUS_OK || module != NULL) {
        fprintf(stderr, "%s was unexpectedly accepted\n", fixture);
        ttl_module_destroy(module);
        return 1;
    }
    printf("[runtime-test] rejected %s: %s\n", fixture, ttl_last_error());
    return 0;
}

static int check_capabilities(const char *module_path) {
    int failed = 1;
    ttl_device_t *device = NULL;
    ttl_module_t *module = NULL;
    ttl_kernel_t *kernel = NULL;
    ttl_capability_set_t available = 0;
    ttl_capability_set_t required = 0;

    if (ttl_device_open(0, &device) != TTL_STATUS_OK ||
        ttl_device_capabilities(device, &available) != TTL_STATUS_OK) {
        fprintf(stderr, "capability device query failed: %s\n", ttl_last_error());
        goto cleanup;
    }
#if defined(TTL_TEST_CUDA)
    const ttl_capability_set_t expected =
        TTL_CAPABILITY_DYNAMIC_WORKGROUP_MEMORY |
        TTL_CAPABILITY_INDEPENDENT_STREAMS |
        TTL_CAPABILITY_NATIVE_GRAPH_EXECUTION;
#else
    const ttl_capability_set_t expected = 0;
#endif
    if (available != expected) {
        fprintf(stderr, "device capabilities: expected 0x%llx, found 0x%llx\n",
                (unsigned long long)expected, (unsigned long long)available);
        goto cleanup;
    }
    if (ttl_module_load(module_path, &module) != TTL_STATUS_OK ||
        ttl_module_required_capabilities(module, &required) != TTL_STATUS_OK) {
        fprintf(stderr, "module capability query failed: %s\n", ttl_last_error());
        goto cleanup;
    }
    if (required != TTL_CAPABILITY_DYNAMIC_WORKGROUP_MEMORY) {
        fprintf(stderr, "dynamic module reported requirements 0x%llx\n",
                (unsigned long long)required);
        goto cleanup;
    }
#if defined(TTL_TEST_MPS)
    if (ttl_kernel_prepare(module, device, &kernel) != TTL_STATUS_UNSUPPORTED ||
        kernel != NULL) {
        fprintf(stderr, "MPS unexpectedly prepared a dynamic-memory module\n");
        goto cleanup;
    }
#endif
    printf("[runtime-test] capabilities available=0x%llx required=0x%llx\n",
           (unsigned long long)available, (unsigned long long)required);
    failed = 0;

cleanup:
    ttl_kernel_destroy(kernel);
    ttl_module_destroy(module);
    ttl_device_destroy(device);
    return failed;
}

static int check_timer(void) {
    int failed = 1;
    ttl_device_t *device = NULL;
    ttl_stream_t *stream = NULL;
    ttl_timer_t *timer = NULL;
    double milliseconds = -1.0;
    if (ttl_device_open(0, &device) != TTL_STATUS_OK ||
        ttl_stream_create(device, &stream) != TTL_STATUS_OK ||
        ttl_timer_create(device, &timer) != TTL_STATUS_OK) {
        fprintf(stderr, "timer setup failed: %s\n", ttl_last_error());
        goto cleanup;
    }
    if (ttl_timer_end(timer, stream, &milliseconds) !=
            TTL_STATUS_INVALID_ARGUMENT ||
        ttl_timer_begin(timer, stream) != TTL_STATUS_OK ||
        ttl_timer_begin(timer, stream) != TTL_STATUS_INVALID_ARGUMENT ||
        ttl_timer_end(timer, stream, &milliseconds) != TTL_STATUS_OK ||
        !isfinite(milliseconds) || milliseconds < 0.0) {
        fprintf(stderr, "timer contract failed: %s\n", ttl_last_error());
        goto cleanup;
    }
    printf("[runtime-test] empty stream interval %.6f ms\n", milliseconds);
    failed = 0;

cleanup:
    ttl_timer_destroy(timer);
    ttl_stream_destroy(stream);
    ttl_device_destroy(device);
    return failed;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s FIXTURE_ROOT CAPABILITY_MODULE\n", argv[0]);
        return 2;
    }
    int failures = 0;
    failures += expect_invalid(argv[1], "bad_schema_version");
    failures += expect_invalid(argv[1], "duplicate_tensor");
    failures += expect_invalid(argv[1], "unknown_binding");
    failures += expect_invalid(argv[1], "artifact_traversal");
    failures += check_capabilities(argv[2]);
    failures += check_timer();
    if (failures != 0) return 1;
    puts("TTL RUNTIME MODULE TESTS PASSED");
    return 0;
}
