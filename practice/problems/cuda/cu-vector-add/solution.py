import numpy as np

import gpusim as gs


@gs.kernel
def add_kernel(t, a, b, c, n):
    i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
    if i < n:
        c[i] = a[i] + b[i]


def vector_add(a, b, block=256):
    n = len(a)
    if n == 0:
        return np.empty(0, dtype=a.dtype)
    da, db = gs.to_device(a, "a"), gs.to_device(b, "b")
    dc = gs.empty(n, dtype=a.dtype, name="c")
    add_kernel[gs.cdiv(n, block), block](da, db, dc, n)
    return dc.copy_to_host()
