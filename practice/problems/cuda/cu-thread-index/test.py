import numpy as np

import gpusim as gs
from checker import check
from solution import whoami


def expected(grid, block):
    gx, gy = grid
    bx, by = block
    rows = []
    for byi in range(gy):
        for bxi in range(gx):
            for ty in range(by):
                for tx in range(bx):
                    bid = byi * gx + bxi
                    tid = ty * bx + tx
                    rows.append((bid * bx * by + tid, bid, tid // 32, tid % 32))
    return np.array(sorted(rows), dtype=np.int32)


def run(grid, block):
    n = grid[0] * grid[1] * block[0] * block[1]
    out = gs.empty((n, 4), dtype=np.int32, name="out")
    whoami[grid, block](out)
    return out.copy_to_host()


def test_example():
    check(run((2, 1), (64, 1)), expected((2, 1), (64, 1)), "grid=(2,1), block=(64,1)")


def test_2d_blocks():
    check(run((3, 2), (16, 8)), expected((3, 2), (16, 8)), "grid=(3,2), block=(16,8)")


def test_warp_spans_rows():
    """block=(4, 24)：一个 warp 跨 8 行"""
    got = run((1, 1), (4, 24))
    check(got, expected((1, 1), (4, 24)), "block=(4,24)")
    check(int(got[40, 2]), 1, "第 40 个线程属于第 1 个 warp")
