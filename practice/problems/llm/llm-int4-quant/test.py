import numpy as np

import solution
from checker import check, check_close
from solution import dequantize, gemv_w4, pack, quantize, unpack


def setup(K=256, N=48, seed=0):
    return np.random.default_rng(seed).standard_normal((K, N)).astype(np.float32)


def test_example():
    W = setup()
    q, s, z = quantize(W, 64)
    check((q.shape, q.dtype, s.shape, s.dtype, z.shape, z.dtype),
          ((256, 48), np.dtype(np.uint8), (4, 48), np.dtype(np.float32), (4, 48), np.dtype(np.uint8)), "形状与类型")
    Wh = dequantize(q, s, z, 64)
    err = np.abs(W - Wh).reshape(4, 64, 48)
    assert (err <= s[:, None, :] / 2 + 1e-6).all(), f"量化误差应该不超过 scale/2，最大超出 {(err - s[:, None, :] / 2).max()}"


def test_matches_reference_formula():
    W = setup(K=64, N=8, seed=1)
    q, s, z = quantize(W, 32)
    Wg = W.reshape(2, 32, 8)
    lo = np.minimum(Wg.min(1), 0)
    hi = np.maximum(Wg.max(1), 0)
    sc = (hi - lo) / 15
    zr = np.round(-lo / sc)
    qq = np.clip(np.round(Wg / sc[:, None]) + zr[:, None], 0, 15).reshape(64, 8)
    check_close(s, sc, rtol=1e-6, what="scales")
    check(z.astype(int), zr.astype(int), "zeros")
    check(q.astype(int), qq.astype(int), "q")


def test_zero_exact_and_constant_groups():
    W = np.zeros((32, 4), dtype=np.float32)
    W[:16, 1] = 5.0                     # 第 0 组第 1 列是常数 5
    W[16:, 2] = -3.0
    q, s, z = quantize(W, 16)
    Wh = dequantize(q, s, z, 16)
    check_close(Wh, W, atol=1e-6, what="0 和常数组都能精确表示")


def test_pack_roundtrip():
    q = np.random.default_rng(2).integers(0, 16, (64, 10)).astype(np.uint8)
    p = pack(q)
    check((p.shape, p.dtype), ((32, 10), np.dtype(np.uint8)), "打包后的形状")
    check(int(p[0, 0]), int(q[0, 0]) | (int(q[1, 0]) << 4), "第 0 个字节：低 4 位是第 0 行，高 4 位是第 1 行")
    check(unpack(p), q, "解包还原")


def test_gemv_grouped():
    W = setup(K=512, N=64, seed=3)
    q, s, z = quantize(W, 128)
    p = pack(q)
    x = np.random.default_rng(4).standard_normal(512).astype(np.float32)
    calls = []
    orig = solution.unpack

    def watched(a):
        calls.append(a.shape[0])
        return orig(a)

    solution.unpack = watched
    try:
        y = gemv_w4(x, p, s, z, 128)
    finally:
        solution.unpack = orig
    check_close(y, x @ dequantize(q, s, z, 128), rtol=1e-4, atol=1e-3, what="gemv_w4")
    assert calls and max(calls) <= 64, f"每次解包最多 group_size/2 = 64 个字节行，实际一次解包了 {max(calls) if calls else 0} 行"
