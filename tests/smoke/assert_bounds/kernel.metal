// Captured TileLang 0.1.13 assertion shape.
// ttl-launch: {"grid":[1],"block":[1],"shared_bytes":0}
#include <metal_stdlib>
using namespace metal;

[[kernel, max_total_threads_per_threadgroup(1)]] void assert_bounds_kernel(
    const device float* table [[buffer(0)]],
    const device int* index [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint3 blockIdx [[threadgroup_position_in_grid]]) {
  int selected = index[0];
  if (!(((0 <= selected) && (selected < 4)))) {
    const char* __tvm_assert_parts[1] = {"gather index out of bounds"};
    TVMFFIErrorSetRaisedFromCStrParts(
        "RuntimeError", __tvm_assert_parts, 1);
    return -1;
  }
  output[0] = table[selected];
}
