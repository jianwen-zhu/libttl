// ttl-launch: {"grid":[1],"block":[64]}
extern "C" __global__ void identity(
    float *output, const float *input, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index];
}
