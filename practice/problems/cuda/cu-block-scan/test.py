import numpy as np

import gpusim as gs
from checker import check


def data(n, seed=0):
    return np.random.default_rng(seed).integers(-50, 50, n).astype(np.int32)


def test_example():
    from solution import inclusive_scan

    x = np.arange(1, 11, dtype=np.int32)
    check(inclusive_scan(x), np.cumsum(x).astype(np.int32), "1..10 的前缀和")


def test_multi_block():
    from solution import inclusive_scan

    for n in [63, 64, 65, 1000, 4096]:
        x = data(n, n)
        check(inclusive_scan(x), np.cumsum(x).astype(np.int32), f"n={n}")


def test_recursive_levels():
    """5000 个元素：block 和有 79 个，超过一个 block，还要再扫一层"""
    from solution import inclusive_scan

    x = data(5000, 7)
    check(inclusive_scan(x), np.cumsum(x).astype(np.int32), "n=5000")


def test_block_kernel_is_parallel():
    from solution import scan_blocks

    x = data(64, 3)
    out, sums = gs.empty(64, dtype=np.int32), gs.empty(1, dtype=np.int32)
    st = scan_blocks[1, 64](gs.to_device(x), out, sums, 64)
    check(out.copy_to_host(), np.cumsum(x).astype(np.int32), "单个 block")
    assert st.syncthreads >= 6, f"block 内应该用 log2(64)=6 轮并行扫描（带屏障），实际屏障 {st.syncthreads} 次"
