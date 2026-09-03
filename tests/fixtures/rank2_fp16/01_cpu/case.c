#include "harness.h"
#include "transpose_reference.h"

#include <inttypes.h>
#include <stdio.h>

static uint16_t input_values[TRANSPOSE_ELEMENTS];
static uint16_t reference_values[TRANSPOSE_ELEMENTS];
static const int64_t output_shape[] = {
    TRANSPOSE_COLUMNS, TRANSPOSE_ROWS,
};
static const int64_t input_shape[] = {
    TRANSPOSE_ROWS, TRANSPOSE_COLUMNS,
};

static int run_reference(void) {
    transpose_fill_input(TRANSPOSE_ELEMENTS, input_values);
    transpose_reference(
        TRANSPOSE_ROWS, TRANSPOSE_COLUMNS, input_values, reference_values);
    return 0;
}

static void report_reference(void) {
    uint64_t checksum = 0;
    for (size_t i = 0; i < TRANSPOSE_ELEMENTS; ++i) {
        checksum += reference_values[i];
    }
    printf("reference: first=0x%04x last=0x%04x checksum=%" PRIu64 "\n",
           reference_values[0], reference_values[TRANSPOSE_ELEMENTS - 1],
           checksum);
}

static lab_tensor_t tensors[] = {
    {"output", TTL_DTYPE_FLOAT16, 2, output_shape,
     NULL, reference_values, sizeof(reference_values)},
    {"input", TTL_DTYPE_FLOAT16, 2, input_shape,
     input_values, NULL, sizeof(input_values)},
};

static const lab_case_t test_case = {
    "fp16 matrix transpose",
    tensors,
    sizeof(tensors) / sizeof(tensors[0]),
    NULL,
    0,
    0.0,
    0.0,
    run_reference,
    report_reference,
};

const lab_case_t *lab_case(void) { return &test_case; }
