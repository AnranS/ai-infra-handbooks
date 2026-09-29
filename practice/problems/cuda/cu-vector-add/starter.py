import numpy as np

import gpusim as gs


@gs.kernel
def add_kernel(t, a, b, c, n):
    i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
    c[i] = a[i] + b[i]            # 缺少边界检查


def vector_add(a, b, block=256):
    n = len(a)
    da, db = gs.to_device(a), gs.to_device(b)
    dc = gs.empty(n, dtype=a.dtype)
    add_kernel[n // block, block](da, db, dc, n)      # grid 大小不对
    return dc.copy_to_host()
