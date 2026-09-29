import gpusim as gs

TILE = 32
ROWS_PER_PASS = 8   # blockDim.y


@gs.kernel
def transpose_kernel(t, a, b, rows, cols):
    # a: (rows, cols)  b: (cols, rows)
    # 提示：tile = t.shared("tile", (TILE, TILE + 1))；屏障写成 yield t.syncthreads()
    x = t.blockIdx.x * TILE + t.threadIdx.x
    y = t.blockIdx.y * TILE + t.threadIdx.y
    for j in range(0, TILE, ROWS_PER_PASS):
        if y + j < rows and x < cols:
            b[x, y + j] = a[y + j, x]      # 能得到正确结果，但写不是合并的
