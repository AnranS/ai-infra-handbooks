import numpy as np


def repeat_kv(kv, n_rep):
    return np.repeat(kv, n_rep, axis=1)


def gqa_attention(q, k, v):
    T, Hq, dh = q.shape
    S, Hkv, _ = k.shape
    k, v = repeat_kv(k, Hq // Hkv), repeat_kv(v, Hq // Hkv)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(dh)          # (H, T, S)
    allowed = np.arange(S)[None, :] <= (S - T + np.arange(T))[:, None]
    scores = np.where(allowed[None], scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    return np.einsum("hts,shd->thd", p, v)


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, n_tokens, bytes_per_elem=2):
    return 2 * n_layers * n_kv_heads * head_dim * n_tokens * bytes_per_elem
