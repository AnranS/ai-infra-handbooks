import numpy as np

import gpusim as gs

BLOCK = 256


def warp_sum(t, v):
    # TODO：用 yield t.shfl_down(v, offset) 做 5 轮归约
    return v
    yield  # 让它成为生成器


@gs.kernel
def block_sum(t, x, out, n):
    acc = 0.0
    i = t.blockIdx.x * BLOCK + t.threadIdx.x
    while i < n:
        acc += x[i]
        i += t.gridDim.x * BLOCK
    t.atomic_add(out, 0, acc)          # 每个线程一次原子操作：正确但很慢
    yield t.syncthreads()
