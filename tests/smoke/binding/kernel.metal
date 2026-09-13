#include <metal_stdlib>
using namespace metal;

// ttl-launch: {"grid":[1],"block":[64]}
kernel void binding_probe(
    device float *output [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *y [[buffer(2)]],
    constant float &alpha [[buffer(3)]],
    device const float *bias [[buffer(4)]],
    constant int &count [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index < uint(count)) {
        output[index] = alpha * x[index] + y[index] + *bias;
    }
}
