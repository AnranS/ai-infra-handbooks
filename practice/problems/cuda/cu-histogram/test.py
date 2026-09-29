import numpy as np

import gpusim as gs
from checker import check
from solution import histogram


def run(n, bins, grid, seed=0, skew=False):
    rng = np.random.default_rng(seed)
    data = (rng.integers(0, 4, n) if skew else rng.integers(0, bins, n)).astype(np.int32)
    dd, hist = gs.to_device(data, "data"), gs.zeros(bins, dtype=np.int32, name="hist")
    st = histogram[grid, 128](dd, hist, n, bins)
    return hist.copy_to_host(), np.bincount(data, minlength=bins).astype(np.int32), st


def test_example():
    got, want, _ = run(1000, 16, 2)
    check(got, want, "n=1000, bins=16")


def test_sizes():
    for n, bins, grid in [(1, 1, 1), (129, 7, 1), (3000, 256, 3), (500, 100, 5)]:
        got, want, _ = run(n, bins, grid, seed=n)
        check(got, want, f"n={n}, bins={bins}, grid={grid}")


def test_global_atomics_bounded():
    got, want, st = run(4000, 64, 4, seed=7, skew=True)
    check(got, want, "偏斜分布")
    assert st.global_atomics <= 4 * 64, f"全局原子操作 {st.global_atomics} 次，应该不超过 grid × bins = {4 * 64}：{st}"
