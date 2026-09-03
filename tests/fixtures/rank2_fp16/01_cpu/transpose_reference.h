#ifndef TTL_TRANSPOSE_REFERENCE_H
#define TTL_TRANSPOSE_REFERENCE_H

#include <stddef.h>
#include <stdint.h>

#define TRANSPOSE_ROWS 32u
#define TRANSPOSE_COLUMNS 48u
#define TRANSPOSE_ELEMENTS (TRANSPOSE_ROWS * TRANSPOSE_COLUMNS)

void transpose_fill_input(size_t count, uint16_t *input);
void transpose_reference(
    size_t rows,
    size_t columns,
    const uint16_t *input,
    uint16_t *output);

#endif
