#include "transpose_reference.h"

void transpose_fill_input(size_t count, uint16_t *input) {
    for (size_t i = 0; i < count; ++i) {
        /* Finite positive IEEE binary16 values in [0.5, 1.0). */
        input[i] = (uint16_t)(UINT16_C(0x3800) + i % UINT16_C(0x0400));
    }
}

void transpose_reference(
    size_t rows,
    size_t columns,
    const uint16_t *input,
    uint16_t *output) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            output[column * rows + row] = input[row * columns + column];
        }
    }
}
