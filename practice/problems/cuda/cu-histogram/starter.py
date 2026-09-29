import numpy as np

import gpusim as gs


@gs.kernel
def histogram(t, data, hist, n, bins):
    i = t.blockIdx.x * t.blockDim.x + t.threadIdx.x
    while i < n:
        t.atomic_add(hist, data[i], 1)       # 每个元素一次全局原子操作
        i += t.gridDim.x * t.blockDim.x
