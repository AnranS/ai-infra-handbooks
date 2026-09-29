import numpy as np

import gpusim as gs
from checker import check, check_close
from solution import block_sum, warp_sum


def run(n, grid, seed=0):
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    dx, out = gs.to_device(x, "x"), gs.zeros(1, name="out")
    st = block_sum[grid, 256](dx, out, n)
    return float(out.copy_to_host()[0]), float(x.astype(np.float64).sum()), st


@gs.kernel
def _warp_test(t, x, out):
    s = yield from warp_sum(t, x[t.threadIdx.x])
    if t.lane == 0:
        out[t.warp_id] = s


def test_example_warp_sum():
    x = np.arange(64, dtype=np.float32)
    out = gs.empty(2, name="out")
    _warp_test[1, 64](gs.to_device(x), out)
    check(out.copy_to_host().tolist(), [float(sum(range(32))), float(sum(range(32, 64)))], "两个 warp 各自的和")


def test_example_block_sum():
    got, want, _ = run(1000, 2)
    check_close(got, want, rtol=1e-5, atol=1e-4, what="n=1000 的和")


def test_many_sizes():
    for n, grid in [(1, 1), (255, 1), (256, 1), (5000, 3), (777, 4)]:
        got, want, _ = run(n, grid, seed=n)
        check_close(got, want, rtol=1e-5, atol=1e-4, what=f"n={n}, grid={grid}")


def test_resources():
    _, _, st = run(4096, 4, seed=9)
    check(st.syncthreads, 4, "__syncthreads 次数（每个 block 1 次）")
    check(st.atomics, 4, "原子操作次数（每个 block 1 次）")
    assert st.warp_ops >= 4 * 9 * 5 // 5, f"应该用 warp shuffle 归约：{st}"
    assert st.shared_store_requests <= 4 * 8, f"共享内存只用来交换每个 warp 的部分和：{st}"
