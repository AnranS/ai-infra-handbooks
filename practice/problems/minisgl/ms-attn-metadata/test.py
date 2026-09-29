import math

import numpy as np

from checker import check, check_close
from solution import forward, prepare_metadata


def setup(seed=0):
    rng = np.random.default_rng(seed)
    page_table = rng.permutation(64)[:4 * 16].reshape(4, 16).astype(np.int32)
    page_table[3] = 63                                          # dummy 行：全部指向 dummy 位置
    return page_table


def test_example():
    pt = setup()
    reqs = [(0, 0, 6), (1, 4, 7), (2, 8, 9)]                    # 书中的例子：A 从头 6 个，B 命中 4 个算 3 个，C decode 1 个
    meta = prepare_metadata(reqs, reqs, pt)
    check(meta["cu_seqlens_q"].tolist(), [0, 6, 9, 10], "cu_seqlens_q")
    check(meta["cache_seqlens"].tolist(), [6, 7, 9], "cache_seqlens")
    check(meta["max_seqlen_k"], 9, "max_seqlen_k")
    check(meta["page_table"].shape, (3, 9), "page_table 截取到 max_seqlen_k")
    check(meta["last_indices"].tolist(), [5, 8, 9], "last_indices")


def test_padded_dummy():
    pt = setup(1)
    reqs = [(0, 5, 6), (1, 2, 3), (2, 7, 8)]
    padded = reqs + [(3, 0, 1)]
    meta = prepare_metadata(reqs, padded, pt)
    check(meta["cu_seqlens_q"].tolist(), [0, 1, 2, 3, 4], "补齐后的 cu_seqlens_q")
    check(meta["last_indices"].tolist(), [0, 1, 2], "last_indices 只包含真实请求")
    check(meta["page_table"][3].tolist(), [63] * 8, "dummy 请求的 page table 行")


def ref(q, kp, vp, reqs, pt):
    H, d, Hkv = q.shape[1], q.shape[2], kp.shape[1]
    out, s = [], 0
    for row, c, L in reqs:
        for j in range(L - c):
            pos = c + j
            o = np.zeros((H, d))
            for h in range(H):
                kh = h // (H // Hkv)
                sc = np.array([q[s + j, h] @ kp[pt[row, t], kh] / math.sqrt(d) for t in range(pos + 1)])
                w = np.exp(sc - sc.max())
                o[h] = (w[:, None] * vp[pt[row, :pos + 1], kh]).sum(0) / w.sum()
            out.append(o)
        s += L - c
    return np.stack(out)


def test_forward_mixed_batch():
    rng = np.random.default_rng(2)
    pt = setup(2)
    reqs = [(0, 0, 6), (1, 4, 7), (2, 8, 9)]
    meta = prepare_metadata(reqs, reqs, pt)
    kp, vp = rng.standard_normal((64, 2, 4)), rng.standard_normal((64, 2, 4))
    q = rng.standard_normal((10, 4, 4))
    check_close(forward(q, kp, vp, meta), ref(q, kp, vp, reqs, pt), rtol=1e-9, atol=1e-10, what="混合 batch 的注意力")
