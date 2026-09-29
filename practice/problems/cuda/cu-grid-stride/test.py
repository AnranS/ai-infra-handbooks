import numpy as np

import gpusim as gs
from checker import check_close
from solution import saxpy


def run(n, grid, block, a=2.5, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)
    dx, dy = gs.to_device(x, "x"), gs.to_device(y, "y")
    st = saxpy[grid, block](np.float32(a), dx, dy, n)
    return dy.copy_to_host(), a * x + y, st


def test_example():
    got, want, _ = run(5000, 4, 64)
    check_close(got, want, rtol=1e-6, what="n=5000, grid=4, block=64")


def test_many_configs():
    for n, grid, block in [(1, 1, 32), (100, 10, 128), (256, 2, 128), (1000, 1, 32), (777, 3, 96)]:
        got, want, _ = run(n, grid, block, seed=n)
        check_close(got, want, rtol=1e-6, what=f"n={n}, grid={grid}, block={block}")


def test_coalesced():
    _, _, st = run(4096, 4, 128)
    assert st.load_efficiency == 1.0 and st.store_efficiency == 1.0, f"访存应该完全合并：{st}"
