import math

import numpy as np

from checker import check, check_close
from solution import build_batch, varlen_paged_attention

BS = 4


def example_items():
    # A：完整 prefill 7 个；B：decode（已有 5 个）；C：分块 prefill 第二段（已有 10 个，本步 12 个）
    A = (list(range(100, 107)), 0, [3, 9], True)
    B = ([205], 5, [1, 7], True)
    C = (list(range(310, 322)), 10, [0, 2, 5, 6, 8, 11], False)
    return [A, B, C]


def test_example():
    b = build_batch(example_items(), BS)
    check(b["query_start_loc"], [0, 7, 8, 20], "query_start_loc")
    check(b["seq_lens"], [7, 6, 22], "seq_lens")
    check(b["positions"].tolist(), list(range(7)) + [5] + list(range(10, 22)), "positions")
    check(b["logits_indices"].tolist(), [6, 7], "logits_indices（C 不需要采样）")
    check((b["max_seqlen_q"], b["max_seqlen_k"]), (12, 22), "max_seqlen_q / max_seqlen_k")
    check(b["block_tables"].tolist(), [[3, 9, -1, -1, -1, -1], [1, 7, -1, -1, -1, -1], [0, 2, 5, 6, 8, 11]], "block_tables")
    check(b["slot_mapping"][:8].tolist(), [12, 13, 14, 15, 36, 37, 38, 29], "slot_mapping 前 8 个")
    check(b["block_tables"].dtype, np.dtype(np.int32), "block_tables 的类型")


def ref_attention(q, kc, vc, items, scale):
    outs = []
    Hkv, d = kc.shape[2], kc.shape[3]
    H = q.shape[1]
    start = 0
    for toks, nc, table, _ in items:
        n, L = len(toks), nc + len(toks)
        ks = np.stack([kc[table[p // BS], p % BS] for p in range(L)])
        vs = np.stack([vc[table[p // BS], p % BS] for p in range(L)])
        for j in range(n):
            pos = nc + j
            o = np.zeros((H, d))
            for h in range(H):
                kh = h // (H // Hkv)
                s = np.array([q[start + j, h] @ ks[t, kh] * scale for t in range(pos + 1)])
                w = np.exp(s - s.max())
                o[h] = (w[:, None] * vs[:pos + 1, kh]).sum(0) / w.sum()
            outs.append(o)
        start += n
    return np.stack(outs)


def test_attention_mixed_batch():
    rng = np.random.default_rng(0)
    items = example_items()
    b = build_batch(items, BS)
    kc, vc = rng.standard_normal((12, BS, 2, 8)), rng.standard_normal((12, BS, 2, 8))
    q = rng.standard_normal((20, 4, 8))
    got = varlen_paged_attention(q, kc, vc, b, 1 / math.sqrt(8))
    check_close(got, ref_attention(q, kc, vc, items, 1 / math.sqrt(8)), rtol=1e-9, atol=1e-10, what="混合批次的注意力")


def test_only_decodes():
    rng = np.random.default_rng(1)
    items = [([7], 9, [4, 1, 2], True), ([8], 0, [5], True), ([9], 3, [0], True)]
    b = build_batch(items, BS)
    check(b["query_start_loc"], [0, 1, 2, 3], "纯 decode 的 query_start_loc")
    kc, vc = rng.standard_normal((6, BS, 1, 4)), rng.standard_normal((6, BS, 1, 4))
    q = rng.standard_normal((3, 2, 4))
    check_close(varlen_paged_attention(q, kc, vc, b, 0.5), ref_attention(q, kc, vc, items, 0.5), rtol=1e-9, atol=1e-10,
                what="纯 decode（第二个请求是第一个 token）")
