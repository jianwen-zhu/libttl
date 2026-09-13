// ttl-launch: {"grid":[1],"block":[1],"shared_bytes":64}
extern "C" __global__ void dynamic_noop() {
    extern __shared__ unsigned char storage[];
    storage[0] = 0;
}
