import gpusim as gs

TILE = 32
ROWS_PER_PASS = 8   # blockDim.y


@gs.kernel
def transpose_kernel(t, a, b, rows, cols):
    tile = t.shared("tile", (TILE, TILE + 1))       # 多一列，错开 bank
    tx, ty = t.threadIdx.x, t.threadIdx.y
    x = t.blockIdx.x * TILE + tx
    y = t.blockIdx.y * TILE + ty
    for j in range(0, TILE, ROWS_PER_PASS):         # 合并地读：同一 warp 的 tx 连续
        if y + j < rows and x < cols:
            tile[ty + j, tx] = a[y + j, x]
    yield t.syncthreads()
    x = t.blockIdx.y * TILE + tx                    # 交换块坐标
    y = t.blockIdx.x * TILE + ty
    for j in range(0, TILE, ROWS_PER_PASS):         # 合并地写
        if y + j < cols and x < rows:
            b[y + j, x] = tile[tx, ty + j]
