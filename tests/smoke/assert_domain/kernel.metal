// Captured TileLang 0.1.13 assertion shape.
// ttl-launch: {"grid":[1],"block":[1],"shared_bytes":0}
#include <metal_stdlib>
using namespace metal;

[[kernel, max_total_threads_per_threadgroup(1)]] void assert_domain_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint3 blockIdx [[threadgroup_position_in_grid]]) {
  if (!((input[0] != 0.0f))) {
    const char* __tvm_assert_parts[1] = {"division by zero"};
    TVMFFIErrorSetRaisedFromCStrParts(
        "RuntimeError", __tvm_assert_parts, 1);
    return -1;
  }
  output[0] = 1.0f / input[0];
}
