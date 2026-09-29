import numpy as np


def prepare_metadata(reqs, padded_reqs, page_table):
    ext = [d - c for _, c, d in padded_reqs]
    cu = np.concatenate([[0], np.cumsum(ext)]).astype(np.int32)
    seqlens = np.array([d for _, _, d in padded_reqs], dtype=np.int32)
    max_k = int(seqlens.max()) if len(seqlens) else 0
    table = np.stack([page_table[row, :max_k] for row, _, _ in padded_reqs]) if padded_reqs else \
        np.zeros((0, 0), dtype=page_table.dtype)
    return {"cu_seqlens_q": cu, "cache_seqlens": seqlens, "max_seqlen_k": max_k, "page_table": table,
            "last_indices": (cu[1:len(reqs) + 1] - 1).astype(np.int64)}


def forward(q, k_pool, v_pool, meta):
    N, H, d = q.shape
    Hkv = k_pool.shape[1]
    out = np.zeros((N, H, d))
    cu, lens, table = meta["cu_seqlens_q"], meta["cache_seqlens"], meta["page_table"]
    for i in range(len(lens)):
        s0, s1, L = int(cu[i]), int(cu[i + 1]), int(lens[i])
        n = s1 - s0
        if n == 0:
            continue
        loc = table[i, :L]
        k = np.repeat(k_pool[loc], H // Hkv, axis=1)
        v = np.repeat(v_pool[loc], H // Hkv, axis=1)
        sc = np.einsum("thd,shd->hts", q[s0:s1], k) / np.sqrt(d)
        allowed = np.arange(L)[None, :] <= (L - n + np.arange(n))[:, None]
        sc = np.where(allowed[None], sc, -np.inf)
        sc -= sc.max(-1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(-1, keepdims=True)
        out[s0:s1] = np.einsum("hts,shd->thd", p, v)
    return out
