import tilelang
import tilelang.language as T


TTL_ENTRYPOINT = "shared_gemm"
TTL_SPECIALIZATIONS = {}
TTL_LAUNCH = {"grid": [1], "block": [32], "shared_bytes": 0}


@tilelang.jit
def shared_gemm(left, right):
    left: T.Tensor((8, 8), T.float32)
    right: T.Tensor((8, 8), T.float32)
    output = T.empty((8, 8), T.float32)

    with T.Kernel(1, threads=32):
        left_shared = T.alloc_shared((8, 8), T.float32)
        right_shared = T.alloc_shared((8, 8), T.float32)
        output_shared = T.alloc_shared((8, 8), T.float32)
        accumulator = T.alloc_fragment((8, 8), T.float32)
        for row, column in T.Parallel(8, 8):
            left_shared[row, column] = left[row, column]
            right_shared[row, column] = right[row, column]
        T.sync_threads()
        T.gemm(
            left_shared, right_shared, accumulator,
            transpose_B=False, clear_accum=True)
        T.copy(accumulator, output_shared)
        T.sync_threads()
        for row, column in T.Parallel(8, 8):
            output[row, column] = output_shared[row, column]
    return output
