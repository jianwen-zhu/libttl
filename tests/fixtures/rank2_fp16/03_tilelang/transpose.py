import tilelang
import tilelang.language as T

TTL_ENTRYPOINT = "transpose"
TTL_SPECIALIZATIONS = {"rows": 32, "columns": 48}
TTL_LAUNCH = {"grid": [6], "block": [256]}


@tilelang.jit
def transpose(input):
    rows = T.const("rows")
    columns = T.const("columns")
    input: T.Tensor((rows, columns), T.float16)
    output = T.empty((columns, rows), T.float16)

    with T.Kernel(T.ceildiv(rows * columns, 256), threads=256) as block:
        for local in T.Parallel(256):
            index = block * 256 + local
            if index < rows * columns:
                row = index // columns
                column = index % columns
                output[column, row] = input[row, column]
    return output
