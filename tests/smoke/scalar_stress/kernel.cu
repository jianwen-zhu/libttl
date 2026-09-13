// ttl-launch: {"grid":[1],"block":[1]}
extern "C" __global__ void scalar_device_stress(
    float *output, const float *value) {
    const unsigned long long start = clock64();
    while (clock64() - start < 1000000ULL) {}
    output[0] = *value;
}
