import numpy as np

import gpusim as gs
from checker import check


def rand(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_example():
    from solution import vector_add

    a, b = rand(1000), rand(1000, 1)
    check(vector_add(a, b), a + b, "n=1000")


def test_sizes():
    from solution import vector_add

    for n, block in [(1, 256), (255, 256), (256, 256), (257, 256), (1000, 32), (777, 128)]:
        a, b = rand(n, n), rand(n, n + 1)
        check(vector_add(a, b, block), a + b, f"n={n}, block={block}")


def test_empty():
    from solution import vector_add

    check(vector_add(np.zeros(0, np.float32), np.zeros(0, np.float32)).shape, (0,), "n=0")


def test_grid_exactly_enough():
    import solution

    calls = []
    real = gs.launch

    def spy(fn, grid, block, *args, **kw):
        calls.append((grid, block))
        return real(fn, grid, block, *args, **kw)

    gs.launch = spy
    try:
        solution.vector_add(rand(1000), rand(1000), 128)
    finally:
        gs.launch = real
    check(len(calls), 1, "启动 kernel 的次数")
    check((calls[0][0], calls[0][1]), (8, 128), "grid、block（n=1000, block=128）")


def test_kernel_bounds_directly():
    from solution import add_kernel

    a, b = gs.to_device(rand(100)), gs.to_device(rand(100, 2))
    c = gs.empty(100)
    add_kernel[4, 32](a, b, c, 100)          # 128 个线程处理 100 个元素
    check(c.copy_to_host(), a.copy_to_host() + b.copy_to_host(), "多出来的线程不能越界")
