#include <metal_stdlib>
using namespace metal;

// ttl-launch: {"grid":[1],"block":[64]}
kernel void identity(
    device float *output [[buffer(0)]],
    device const float *input [[buffer(1)]],
    constant int &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < uint(count)) output[index] = input[index];
}
