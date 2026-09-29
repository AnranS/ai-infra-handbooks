import numpy as np

from checker import check, check_close
from solution import block_gemm, e4m3, quant_act, quant_weight


def grid():
    """全部非负的 E4M3 值，按从小到大排列：下标就是它的编码（0 是 0x00）"""
    vals = [m * 2.0**-9 for m in range(8)]
    vals += [(1 + m / 8) * 2.0**e for e in range(-6, 9) for m in range(8)]
    return np.array([v for v in vals if v <= 448.0])


def test_example():
    got = e4m3(np.array([1.0, 1.0625, 0.3, 500.0, 2**-10, 3 * 2**-10]))
    check(got.dtype, np.dtype(np.float32), "返回 float32")
    check(got, np.array([1.0, 1.0, 0.3125, 448.0, 0.0, 2**-8], dtype=np.float32), "几个典型值")


def test_e4m3_grid():
    g = grid()
    check(len(g), 127, "非负的 E4M3 值一共 127 个（0x7F 是 NaN）")
    check(e4m3(g), g.astype(np.float32), "E4M3 值本身舍入后不变")
    check(e4m3(-g), (-g).astype(np.float32), "负数对称")
    mids = (g[:-1] + g[1:]) / 2
    want = np.where(np.arange(len(mids)) % 2 == 0, g[:-1], g[1:])       # 正中间：取编码为偶数（尾数最低位为 0）的那个
    check(e4m3(mids), want.astype(np.float32), "正好在两个值中间时向偶数舍入")
    check(e4m3(mids * (1 + 1e-6)), g[1:].astype(np.float32), "略大于中点时向上")
    check(e4m3(mids * (1 - 1e-6)), g[:-1].astype(np.float32), "略小于中点时向下")
    check(e4m3(np.array([449.0, 470.0, 1e6, -1e9])), np.array([448, 448, 448, -448], dtype=np.float32), "超出范围饱和到 ±448")


def test_quant_act():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 512)).astype(np.float32) * np.exp2(rng.integers(-8, 8, (6, 1)))
    a[2, 128:256] = 0.0                                                   # 一整组都是 0
    q, s = quant_act(a)
    check(q.shape, (6, 512), "q 的形状")
    check(s.shape, (6, 4), "s 的形状：每行每 128 个一个")
    amax = np.abs(a.reshape(6, 4, 128)).max(-1)
    check_close(s[amax > 0], amax[amax > 0] / 448, rtol=1e-6, what="缩放 = amax / 448")
    check(float(s[2, 1]), 1.0, "全 0 的组缩放为 1")
    check(e4m3(q), q, "q 里都是 E4M3 值")
    err = np.abs(q.reshape(6, 4, 128) * s[..., None] - a.reshape(6, 4, 128))
    bound = np.maximum(np.abs(a.reshape(6, 4, 128)) * 2.0**-4, s[..., None] * 2.0**-10) * (1 + 1e-5)
    check(bool((err <= bound).all()), True, "反量化误差不超过半个间隔")
    q2, s2 = quant_act(a, group=64, pow2=True)
    check(s2.shape, (6, 8), "group=64")
    check(bool(np.all(np.exp2(np.round(np.log2(s2))) == s2)), True, "pow2=True 时缩放是 2 的幂")
    amax2 = np.abs(a.reshape(6, 8, 64)).max(-1)
    nz = amax2 > 0
    check(bool(np.all(s2[nz] * 448 >= amax2[nz]) and np.all(s2[nz] * 224 < amax2[nz])), True,
          "2 的幂的缩放不小于 amax / 448，也不会大出一倍以上")
    check(bool(np.abs(q2).max() <= 448), True, "2 的幂缩放后不溢出")


def test_quant_weight_and_gemm():
    rng = np.random.default_rng(1)
    M, K, N = 5, 384, 256
    a = rng.standard_normal((M, K)).astype(np.float32)
    w = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
    w[:128, 128:] *= 50                                                   # 一个块的数值大得多
    wq, sw = quant_weight(w)
    check(sw.shape, (3, 2), "权重缩放：每 128×128 一个")
    check_close(sw[0, 1], np.abs(w[:128, 128:]).max() / 448, rtol=1e-6, what="块 (0, 1) 的缩放")
    aq, sa = quant_act(a)
    out = block_gemm(aq, sa, wq, sw)
    check(out.shape, (M, N), "输出形状")
    deq = (aq.reshape(M, 3, 128) * sa[..., None]).reshape(M, K) @ (wq.reshape(3, 128, 2, 128) * sw[:, None, :, None]).reshape(K, N)
    check_close(out, deq, rtol=1e-4, atol=1e-4, what="分块累加 = 先反量化再相乘")
    ref = a @ w
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    check(bool(rel < 0.06), True, f"和全精度结果的相对误差应在 E4M3 的精度范围内（实际 {rel:.4f}）")


def test_outliers():
    rng = np.random.default_rng(2)
    M, K, N = 16, 1024, 128
    w = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
    a = rng.standard_normal((M, K)).astype(np.float32)
    out_ch = [5, 77, 300, 301]                                           # 离群通道落在第 0、2 组
    a[:, out_ch] *= 1e5
    clean = np.ones(K, dtype=bool)                                       # 不含离群值的组里的通道
    clean[0:128] = clean[256:384] = False
    ref = a[:, clean].astype(np.float64) @ w[clean]
    errs = {}
    for name, group in (("逐 token", K), ("每 128 个一组", 128)):
        aq, sa = quant_act(a, group=group)
        a_dq = (aq.reshape(M, K // group, group) * sa[..., None]).reshape(M, K)
        got = a_dq[:, clean] @ w[clean]
        errs[name] = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    check(bool(errs["逐 token"] > 0.15), True, f"逐 token 缩放时正常通道被压进非规格化数，误差应该很大（实际 {errs['逐 token']:.3f}）")
    check(bool(errs["每 128 个一组"] < 0.05), True, f"分块缩放时，不含离群值的组不受影响（实际 {errs['每 128 个一组']:.3f}）")
