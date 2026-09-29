import numpy as np

import gpusim as gs
from checker import check, check_close


def data(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_example():
    from solution import reduce_sum

    x = data(1000)
    check_close(reduce_sum(x), float(x.astype(np.float64).sum()), rtol=1e-5, atol=1e-4, what="n=1000")


def test_sizes():
    from solution import reduce_sum

    check(reduce_sum(np.zeros(0, np.float32)), 0.0, "n=0")
    for n in [1, 511, 512, 513, 3000, 20001]:          # 20001 需要两轮：先得到 40 个部分和，再归约成 1 个
        x = data(n, n)
        check_close(reduce_sum(x), float(x.astype(np.float64).sum()), rtol=1e-4, atol=1e-3, what=f"n={n}")


def test_no_bank_conflicts_and_first_add():
    import solution

    launches = []
    real = gs.launch

    def spy(fn, grid, block, *args, **kw):
        st = real(fn, grid, block, *args, **kw)
        launches.append((grid, st))
        return st

    gs.launch = spy
    try:
        x = data(5000, 1)
        solution.reduce_sum(x)
    finally:
        gs.launch = real
    check(launches[0][0], 10, "第一轮的 block 数（每个 block 处理 512 个元素）")
    for grid, st in launches:
        check(st.bank_conflicts, 0, f"grid={grid} 那一轮的 bank conflict")
