import numpy as np

import gpusim as gs

BLOCK = 256


@gs.kernel
def reduce_kernel(t, x, partial, n):
    s = t.shared("s", BLOCK)
    tid = t.threadIdx.x
    i = t.blockIdx.x * 2 * BLOCK + tid
    v = x[i] if i < n else 0.0
    if i + BLOCK < n:
        v += x[i + BLOCK]
    s[tid] = v
    yield t.syncthreads()
    stride = BLOCK // 2
    while stride > 0:
        if tid < stride:
            s[tid] += s[tid + stride]
        yield t.syncthreads()
        stride //= 2
    if tid == 0:
        partial[t.blockIdx.x] = s[0]


def reduce_sum(x):
    n = len(x)
    if n == 0:
        return 0.0
    cur = gs.to_device(np.asarray(x, dtype=np.float32), "x")
    while n > 1:
        grid = gs.cdiv(n, 2 * BLOCK)
        partial = gs.empty(grid, name="partial")
        reduce_kernel[grid, BLOCK](cur, partial, n)
        cur, n = partial, grid
    return float(cur.copy_to_host()[0])
