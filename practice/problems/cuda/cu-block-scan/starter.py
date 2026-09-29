import numpy as np

import gpusim as gs

BLOCK = 64


@gs.kernel
def scan_blocks(t, x, out, block_sums, n):
    # 只由线程 0 串行扫描：正确但完全没有并行
    if t.threadIdx.x == 0:
        acc = 0
        base = t.blockIdx.x * BLOCK
        for j in range(BLOCK):
            if base + j < n:
                acc += x[base + j]
                out[base + j] = acc
        block_sums[t.blockIdx.x] = acc


@gs.kernel
def add_offsets(t, out, scanned_sums, n):
    pass


def inclusive_scan(x):
    n = len(x)
    grid = gs.cdiv(n, BLOCK)
    dx, out, sums = gs.to_device(x, "x"), gs.empty(n, dtype=np.int32, name="out"), gs.empty(grid, dtype=np.int32)
    scan_blocks[grid, BLOCK](dx, out, sums, n)
    return out.copy_to_host()                  # 没有把各 block 的前缀加上去
