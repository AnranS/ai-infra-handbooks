import numpy as np

FP8_MAX = 448.0


def e4m3(x):
    """舍入到最近的 E4M3 值（向偶数舍入），超过 448 的饱和"""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.clip(e, -6, 8)                       # 低于 2^-6 的是非规格化数，步长和 e = -6 的区间一样（2^-9）
    step = np.exp2(e - 3)                       # 3 位尾数：一个区间里 8 个值
    q = np.minimum(np.rint(a / step) * step, FP8_MAX)
    return (np.sign(x) * q).astype(np.float32)


def _scale(amax, pow2):
    s = np.where(amax > 0, amax / FP8_MAX, 1.0)
    if pow2:
        s = np.exp2(np.ceil(np.log2(s)))
    return s


def quant_act(a, group=128, pow2=False):
    a = np.asarray(a, dtype=np.float32)
    M, K = a.shape
    g = a.reshape(M, K // group, group)
    s = _scale(np.abs(g).max(-1), pow2)
    q = e4m3(g / s[..., None]).reshape(M, K)
    return q, s.astype(np.float32)


def quant_weight(w, block=128, pow2=False):
    w = np.asarray(w, dtype=np.float32)
    K, N = w.shape
    b = w.reshape(K // block, block, N // block, block)
    s = _scale(np.abs(b).max(axis=(1, 3)), pow2)
    q = e4m3(b / s[:, None, :, None]).reshape(K, N)
    return q, s.astype(np.float32)


def block_gemm(aq, sa, wq, sw):
    M, K = aq.shape
    N = wq.shape[1]
    g, bn = K // sa.shape[1], N // sw.shape[1]
    out = np.zeros((M, N), dtype=np.float32)
    for b in range(K // g):
        part = aq[:, b * g:(b + 1) * g] @ wq[b * g:(b + 1) * g]          # Tensor Core 上的 FP8 矩阵乘（这里用 fp32 模拟）
        out += part * sa[:, b:b + 1] * np.repeat(sw[b], bn)[None, :]     # 在 fp32 里乘上两个缩放、累加
    return out
