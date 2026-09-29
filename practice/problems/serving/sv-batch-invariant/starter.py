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
    outs = []
    for s in range(0, len(K), split_len):
        sc = K[s:s + split_len] @ q / np.float32(math.sqrt(len(q)))
        p = np.exp(sc - sc.max())
        outs.append(p / p.sum() @ V[s:s + split_len])
    return np.mean(outs, axis=0)                           # 直接平均各段：错误，没有按各段的 lse 加权


def batch_decode(requests, split_len=256):
    return heuristic_batch_decode(requests)


_rng = np.random.default_rng(0)


def sample(probs, seed, position):
    return int(_rng.choice(len(probs), p=probs))          # 全局随机数流：结果取决于调用顺序
