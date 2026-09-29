import numpy as np


def repeat_kv(kv, n_rep):
    return np.tile(kv, (1, n_rep, 1))     # 顺序不对


def gqa_attention(q, k, v):
    pass


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, n_tokens, bytes_per_elem=2):
    pass
