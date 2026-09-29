import math

import numpy as np

MASK = (1 << 64) - 1


def splitmix64(x):
    """把一个 64 位整数打散成另一个 64 位整数（相邻的输入得到毫不相关的输出）"""
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def heuristic_batch_decode(requests):
    """按 batch 大小决定切几段：请求越少切得越多（结果随 batch 变化，仅供对比）"""
    splits = max(1, 16 // len(requests))
    return [decode_attention(q, K, V, max(1, math.ceil(len(K) / splits))) for q, K, V in requests]


def decode_attention(q, K, V, split_len):
    q, K, V = (np.asarray(a, dtype=np.float32) for a in (q, K, V))
    scale = np.float32(1 / math.sqrt(len(q)))
    out, lse = None, None
    for s in range(0, len(K), split_len):                  # 段长固定：段数只由这个请求自己的长度决定
        sc = (K[s:s + split_len] @ q) * scale
        m = sc.max()
        p = np.exp(sc - m)
        l_c = m + np.log(p.sum())
        o_c = (p / p.sum()) @ V[s:s + split_len]
        if out is None:
            out, lse = o_c, l_c
        else:                                              # 从左到右依次合并（online softmax 的公式）
            new = np.logaddexp(lse, l_c)
            out = out * np.exp(lse - new) + o_c * np.exp(l_c - new)
            lse = new
    return out.astype(np.float32)


def batch_decode(requests, split_len=256):
    return [decode_attention(q, K, V, split_len) for q, K, V in requests]


def sample(probs, seed, position):
    z = splitmix64((splitmix64(seed & MASK) + position) & MASK)          # 计数器式随机数：只由 (seed, position) 决定
    u = (z >> 11) * 2.0**-53                                             # 取高 53 位，得到 [0, 1) 里的 double
    cdf = np.cumsum(np.asarray(probs, dtype=np.float64))
    return int(min(np.searchsorted(cdf, u * cdf[-1], side="right"), len(cdf) - 1))
