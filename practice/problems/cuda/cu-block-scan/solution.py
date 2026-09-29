import numpy as np

import gpusim as gs

BLOCK = 64


@gs.kernel
def scan_blocks(t, x, out, block_sums, n):
    s = t.shared("s", BLOCK, dtype=np.int32)
    tid = t.threadIdx.x
    i = t.blockIdx.x * BLOCK + tid
    s[tid] = x[i] if i < n else 0
    yield t.syncthreads()
    offset = 1
    while offset < BLOCK:
        v = s[tid - offset] if tid >= offset else 0
        yield t.syncthreads()
        s[tid] += v
        yield t.syncthreads()
        offset *= 2
    if i < n:
        out[i] = s[tid]
    if tid == BLOCK - 1:
        block_sums[t.blockIdx.x] = s[tid]


@gs.kernel
def add_offsets(t, out, scanned_sums, n):
    i = t.blockIdx.x * BLOCK + t.threadIdx.x
    if t.blockIdx.x > 0 and i < n:
        out[i] += scanned_sums[t.blockIdx.x - 1]


def _scan_device(dx, n):
    grid = gs.cdiv(n, BLOCK)
    out = gs.empty(n, dtype=np.int32, name="out")
    sums = gs.empty(grid, dtype=np.int32, name="block_sums")
    scan_blocks[grid, BLOCK](dx, out, sums, n)
    if grid > 1:
        scanned = _scan_device(sums, grid)
        add_offsets[grid, BLOCK](out, scanned, n)
    return out


def inclusive_scan(x):
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    return _scan_device(gs.to_device(np.asarray(x, dtype=np.int32), "x"), n).copy_to_host()
