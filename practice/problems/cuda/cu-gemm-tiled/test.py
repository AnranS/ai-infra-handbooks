import numpy as np

import gpusim as gs
from checker import check_close
from solution import gemm_tiled


def run(M, N, K, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    dA, dB, dC = gs.to_device(A, "A"), gs.to_device(B, "B"), gs.empty((M, N), name="C")
    st = gemm_tiled[(gs.cdiv(N, 8), gs.cdiv(M, 8)), (8, 8)](dA, dB, dC, M, N, K)
    return dC.copy_to_host(), A.astype(np.float64) @ B.astype(np.float64), st


def test_example():
    got, want, _ = run(16, 16, 16)
    check_close(got, want, rtol=1e-4, atol=1e-4, what="16×16×16")


def test_odd_shapes():
    for M, N, K in [(1, 1, 1), (9, 7, 13), (20, 3, 17), (5, 24, 8)]:
        got, want, _ = run(M, N, K, seed=M * 100 + N)
        check_close(got, want, rtol=1e-4, atol=1e-4, what=f"M={M}, N={N}, K={K}")


def test_data_reuse():
    """32×32×32：全局读扇区数不超过朴素写法的 1/3"""
    got, want, st = run(32, 32, 32, seed=5)
    check_close(got, want, rtol=1e-4, atol=1e-4, what="结果")
    naive = 2 * 32 * 32 * 32 * 4 // 32 // 8 * 8
    assert st.global_load_sectors <= naive // 3, (
        f"全局读了 {st.global_load_sectors} 个扇区，朴素写法约 {naive} 个，要求不超过 {naive // 3}：{st}")
