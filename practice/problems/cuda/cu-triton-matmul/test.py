import numpy as np

import tritonkit
from checker import check, check_close
from solution import matmul


def run(M, N, K, seed=0, **blocks):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    c = tritonkit.to_host(matmul(tritonkit.to_dev(a), tritonkit.to_dev(b), **blocks))
    return c, a.astype(np.float64) @ b.astype(np.float64)


def test_example():
    got, want = run(64, 64, 64)
    check_close(got, want, rtol=1e-3, atol=1e-3, what="64×64×64")


def test_ragged_shapes():
    for M, N, K in [(1, 1, 1), (33, 17, 50), (100, 40, 7), (16, 16, 100)]:
        got, want = run(M, N, K, seed=M + N + K, BM=16, BN=16, BK=16)
        check_close(got, want, rtol=1e-3, atol=1e-3, what=f"M={M}, N={N}, K={K}")


def test_blocking_structure():
    run(64, 96, 80, seed=5, BM=32, BN=32, BK=16)
    st = tritonkit.last_stats()
    if st is None:
        return
    programs = 2 * 3
    check(st.programs, programs, "program 数")
    check(st.loads, programs * 2 * 5, "每个 program 对 A、B 各 load cdiv(K, BK) = 5 次")
    check(st.dots, programs * 5, "tl.dot 的次数")
