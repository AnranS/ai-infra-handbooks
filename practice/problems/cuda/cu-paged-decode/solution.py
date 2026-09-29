import numpy as np


def paged_decode(q, k_cache, v_cache, block_tables, seq_lens):
    B, Hq, d = q.shape
    bs, Hkv = k_cache.shape[1], k_cache.shape[2]
    rep = Hq // Hkv
    out = np.empty((B, Hq, d), dtype=np.float64)
    for b in range(B):
        L = int(seq_lens[b])
        blocks = block_tables[b, :(L + bs - 1) // bs]
        k = k_cache[blocks].reshape(-1, Hkv, d)[:L]          # (L, Hkv, d)
        v = v_cache[blocks].reshape(-1, Hkv, d)[:L]
        k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        s = np.einsum("hd,lhd->hl", q[b], k) / np.sqrt(d)
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        out[b] = np.einsum("hl,lhd->hd", p, v)
    return out
