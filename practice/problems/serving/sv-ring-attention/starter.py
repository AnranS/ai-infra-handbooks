import numpy as np


def partial_attention(q, k, v, q_pos, k_pos, causal):
    pass


def merge(parts):
    o = sum(p[0] for p in parts) / len(parts)      # 简单平均：不对
    return o, parts[0][1]


def context_parallel_attention(q, k, v, n, causal):
    S, T = len(k), len(q)
    q_pos = np.arange(S - T, S)
    parts = []
    for idx in np.array_split(np.arange(S), n):
        parts.append(partial_attention(q, k[idx], v[idx], q_pos, idx, causal))
    return merge(parts)
