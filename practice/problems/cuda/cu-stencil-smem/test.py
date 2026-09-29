import numpy as np

import gpusim as gs
from checker import check, check_close
from solution import conv1d


def ref(x, w, R):
    n = len(x)
    xp = np.concatenate([np.zeros(R), x.astype(np.float64), np.zeros(R)])
    return np.array([np.dot(w, xp[i:i + 2 * R + 1]) for i in range(n)])


def run(n, R, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    w = rng.standard_normal(2 * R + 1).astype(np.float32)
    dx, dw, dout = gs.to_device(x, "inp"), gs.to_device(w, "w"), gs.empty(n, name="out")
    st = conv1d[gs.cdiv(n, 128), 128](dx, dw, dout, n, R)
    return dout.copy_to_host(), ref(x, w, R), st


def test_example():
    got, want, _ = run(300, 3)
    check_close(got, want, rtol=1e-4, atol=1e-4, what="n=300, R=3")


def test_edges_and_sizes():
    for n, R in [(1, 1), (127, 5), (128, 0), (129, 32), (500, 16)]:
        got, want, _ = run(n, R, seed=n)
        check_close(got, want, rtol=1e-4, atol=1e-4, what=f"n={n}, R={R}")


def test_global_traffic():
    """n=2048, R=8：全局读扇区数不超过朴素写法的 1/3"""
    got, want, st = run(2048, 8, seed=3)
    check_close(got, want, rtol=1e-4, atol=1e-4, what="结果")
    naive_sectors = 2048 * 17 * 4 // 32
    assert st.global_load_sectors <= naive_sectors // 3, (
        f"全局读了 {st.global_load_sectors} 个扇区，朴素写法约 {naive_sectors} 个，要求不超过 {naive_sectors // 3}：{st}")
    assert st.shared_load_requests > 0, "没有用共享内存"
