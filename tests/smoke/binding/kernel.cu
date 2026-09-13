// ttl-launch: {"grid":[1],"block":[64]}
extern "C" __global__ void binding_probe(
    float *output,
    const float *x,
    const float *y,
    float alpha,
    const float *bias,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = alpha * x[index] + y[index] + *bias;
    }
}
