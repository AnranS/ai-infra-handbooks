import math

import numpy as np


def flashinfer_metadata(reqs, page_table, page_size):
    qo, kv, idx, last = [0], [0], [], []
    for row, extend_len, seq_len in reqs:
        n = math.ceil(seq_len / page_size)
        idx += [int(page_table[row, j * page_size]) // page_size for j in range(n)]
        qo.append(qo[-1] + extend_len)
        kv.append(kv[-1] + n)
        last.append(seq_len - (n - 1) * page_size)
    as32 = lambda x: np.array(x, dtype=np.int32)  # noqa: E731
    return {"qo_indptr": as32(qo), "kv_indptr": as32(kv), "kv_indices": as32(idx), "kv_last_page_len": as32(last)}


def paged_attention_from_meta(q, k_cache, v_cache, meta, page_size):
    N, H, d = q.shape
    out = np.zeros((N, H, d))
    qo, kvp, idx, last = meta["qo_indptr"], meta["kv_indptr"], meta["kv_indices"], meta["kv_last_page_len"]
    for i in range(len(qo) - 1):
        pages = idx[kvp[i]:kvp[i + 1]]
        L = (len(pages) - 1) * page_size + int(last[i])
        k = k_cache[pages].reshape(-1, H, d)[:L]
        v = v_cache[pages].reshape(-1, H, d)[:L]
        s0, s1 = qo[i], qo[i + 1]
        n = s1 - s0
        sc = np.einsum("thd,shd->hts", q[s0:s1], k) / np.sqrt(d)
        allowed = np.arange(L)[None, :] <= (L - n + np.arange(n))[:, None]
        sc = np.where(allowed[None], sc, -np.inf)
        sc -= sc.max(-1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(-1, keepdims=True)
        out[s0:s1] = np.einsum("hts,shd->thd", p, v)
    return out
