// ttl-launch: {"grid":[6],"block":[256]}
#include <cuda_fp16.h>

extern "C" __global__ void transpose_kernel(
    __half *output,
    const __half *input,
    int rows,
    int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * columns;
    if (index < count) {
        const int row = index / columns;
        const int column = index % columns;
        output[column * rows + row] = input[index];
    }
}
