import numpy as np

import gpusim as gs

BLOCK = 256


def warp_sum(t, v):
    offset = 16
    while offset > 0:
        v += yield t.shfl_down(v, offset)
        offset //= 2
    return v


@gs.kernel
def block_sum(t, x, out, n):
    partial = t.shared("partial", 32)
    tid = t.threadIdx.x
    acc = 0.0
    i = t.blockIdx.x * BLOCK + tid
    while i < n:
        acc += x[i]
        i += t.gridDim.x * BLOCK
    s = yield from warp_sum(t, acc)
    if t.lane == 0:
        partial[t.warp_id] = s
    yield t.syncthreads()
    if t.warp_id == 0:
        v = partial[t.lane] if t.lane < BLOCK // 32 else 0.0
        v = yield from warp_sum(t, v)
        if t.lane == 0:
            t.atomic_add(out, 0, v)
