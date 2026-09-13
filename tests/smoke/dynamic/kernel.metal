#include <metal_stdlib>
using namespace metal;

// ttl-launch: {"grid":[1],"block":[1],"shared_bytes":64}
kernel void dynamic_noop(uint index [[thread_position_in_grid]]) {
    (void)index;
}
