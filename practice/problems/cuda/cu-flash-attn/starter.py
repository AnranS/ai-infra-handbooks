import numpy as np


def flash_attn_fwd(q, k, v, causal=False, Br=64, Bc=64):
    T, S, d = q.shape[0], k.shape[0], q.shape[1]
    s = q @ k.T / np.sqrt(d)                    # 完整的 T×S 分数矩阵：内存 O(TS)
    if causal:
        s = np.where(np.arange(S)[None] <= (S - T + np.arange(T))[:, None], s, -np.inf)
    m = s.max(1, keepdims=True)
    p = np.exp(s - m)
    l = p.sum(1, keepdims=True)
    return p @ v / l, (m + np.log(l))[:, 0]
