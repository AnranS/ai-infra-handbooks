import numpy as np


def build_batch(items, block_size):
    input_ids, positions, slots, qsl, seq_lens, tables, logits_idx = [], [], [], [0], [], [], []
    for token_ids, num_computed, table, need_logits in items:
        pos = list(range(num_computed, num_computed + len(token_ids)))
        input_ids += list(token_ids)
        positions += pos
        slots += [table[p // block_size] * block_size + p % block_size for p in pos]
        qsl.append(qsl[-1] + len(token_ids))
        seq_lens.append(num_computed + len(token_ids))
        tables.append(list(table))
        if need_logits:
            logits_idx.append(qsl[-1] - 1)
    width = max((len(t) for t in tables), default=0)
    bt = np.full((len(tables), width), -1, dtype=np.int32)
    for i, t in enumerate(tables):
        bt[i, :len(t)] = t
    return {
        "input_ids": np.array(input_ids, dtype=np.int64), "positions": np.array(positions, dtype=np.int64),
        "slot_mapping": np.array(slots, dtype=np.int64), "query_start_loc": qsl, "seq_lens": seq_lens,
        "logits_indices": np.array(logits_idx, dtype=np.int64),
        "max_seqlen_q": max((qsl[i + 1] - qsl[i] for i in range(len(items))), default=0),
        "max_seqlen_k": max(seq_lens, default=0), "block_tables": bt,
    }


def varlen_paged_attention(q, k_cache, v_cache, batch, scale):
    N, H, d = q.shape
    bs, Hkv = k_cache.shape[1], k_cache.shape[2]
    out = np.zeros((N, H, d))
    qsl = batch["query_start_loc"]
    for i, L in enumerate(batch["seq_lens"]):
        s0, s1 = qsl[i], qsl[i + 1]
        n = s1 - s0
        blocks = batch["block_tables"][i, :(L + bs - 1) // bs]
        k = np.repeat(k_cache[blocks].reshape(-1, Hkv, d)[:L], H // Hkv, axis=1)
        v = np.repeat(v_cache[blocks].reshape(-1, Hkv, d)[:L], H // Hkv, axis=1)
        sc = np.einsum("thd,shd->hts", q[s0:s1], k) * scale
        allowed = np.arange(L)[None, :] <= (L - n + np.arange(n))[:, None]
        sc = np.where(allowed[None], sc, -np.inf)
        sc -= sc.max(-1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(-1, keepdims=True)
        out[s0:s1] = np.einsum("hts,shd->thd", p, v)
    return out
