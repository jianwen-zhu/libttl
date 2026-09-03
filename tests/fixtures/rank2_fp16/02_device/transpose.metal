#include <metal_stdlib>
using namespace metal;

// ttl-launch: {"grid":[6],"block":[256]}
kernel void transpose_kernel(
    device half *output [[buffer(0)]],
    device const half *input [[buffer(1)]],
    constant int &rows [[buffer(2)]],
    constant int &columns [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const int count = rows * columns;
    if (index < uint(count)) {
        const int row = int(index) / columns;
        const int column = int(index) % columns;
        output[column * rows + row] = input[index];
    }
}
