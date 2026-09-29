import numpy as np

import gpusim as gs
from checker import check_close


def make(N, K, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 16, (N, K)).astype(np.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    scales = rng.uniform(0.01, 0.1, N).astype(np.float32)
    x = rng.standard_normal(K).astype(np.float32)
    W = (q.astype(np.float64) - 8) * scales[:, None]
    return packed, scales, x, W @ x.astype(np.float64)


def run(N, K, seed=0):
    from solution import gemv_w4

    packed, scales, x, want = make(N, K, seed)
    y = gs.empty(N, name="y")
    st = gemv_w4[gs.cdiv(N, 4), (32, 4)](gs.to_device(packed, "packed"), gs.to_device(scales, "scales"),
                                         gs.to_device(x, "x"), y, N, K)
    return y.copy_to_host(), want, st


def test_example():
    got, want, _ = run(8, 128)
    check_close(got, want, rtol=1e-4, atol=1e-4, what="N=8, K=128")


def test_shapes():
    for N, K in [(1, 2), (5, 70), (13, 256), (6, 1000)]:
        got, want, _ = run(N, K, seed=N)
        check_close(got, want, rtol=1e-4, atol=1e-4, what=f"N={N}, K={K}")


def test_coalesced():
    got, want, st = run(16, 512, seed=3)
    check_close(got, want, rtol=1e-4, atol=1e-4, what="结果")
    eff = st.array("packed")["load_efficiency"]
    assert eff >= 0.9, f"读权重 packed 的效率只有 {eff:.0%}，要求 ≥ 90%（相邻 lane 应该读相邻的字节）"
