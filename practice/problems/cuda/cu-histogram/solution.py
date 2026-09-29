import numpy as np

import gpusim as gs


@gs.kernel
def histogram(t, data, hist, n, bins):
    local = t.shared("local", 256, dtype=np.int32)
    for b in range(t.threadIdx.x, bins, t.blockDim.x):
        local[b] = 0
    yield t.syncthreads()
    i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
    while i < n:
        t.atomic_add(local, data[i], 1)
        i += t.gridDim.x * t.blockDim.x
    yield t.syncthreads()
    for b in range(t.threadIdx.x, bins, t.blockDim.x):
        c = local[b]
        if c:
            t.atomic_add(hist, b, c)
