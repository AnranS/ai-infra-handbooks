import numpy as np

import gpusim as gs

BLOCK = 256


@gs.kernel
def reduce_kernel(t, x, partial, n):
    s = t.shared("s", BLOCK)
    tid = t.threadIdx.x
    i = t.blockIdx.x * BLOCK + tid                 # 每个 block 只处理 256 个元素
    s[tid] = x[i] if i < n else 0.0
    yield t.syncthreads()
    stride = 1
    while stride < BLOCK:                          # 交错寻址：有 bank conflict
        idx = 2 * stride * tid
        if idx < BLOCK:
            s[idx] += s[idx + stride]
        yield t.syncthreads()
        stride *= 2
    if tid == 0:
        partial[t.blockIdx.x] = s[0]


def reduce_sum(x):
    n = len(x)
    cur = gs.to_device(x.astype(np.float32), "x")
    while n > 1:
        grid = gs.cdiv(n, BLOCK)
        partial = gs.empty(grid, name="partial")
        reduce_kernel[grid, BLOCK](cur, partial, n)
        cur, n = partial, grid
    return float(cur.copy_to_host()[0]) if n else 0.0
