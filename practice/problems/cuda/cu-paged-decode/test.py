import math

import numpy as np

from checker import check_close
from solution import paged_decode


def build(seq_lens, Hq=4, Hkv=2, d=8, bs=4, num_blocks=40, seed=0):
    rng = np.random.default_rng(seed)
    kc = rng.standard_normal((num_blocks, bs, Hkv, d))
    vc = rng.standard_normal((num_blocks, bs, Hkv, d))
    perm = rng.permutation(num_blocks)
    max_blocks = max(math.ceil(L / bs) for L in seq_lens) + 2
    bt = np.full((len(seq_lens), max_blocks), -1, dtype=np.int64)
    used = 0
    for b, L in enumerate(seq_lens):
        n = math.ceil(L / bs)
        bt[b, :n] = perm[used:used + n]
        used += n
    q = rng.standard_normal((len(seq_lens), Hq, d))
    return q, kc, vc, bt, np.array(seq_lens)


def ref(q, kc, vc, bt, lens):
    B, Hq, d = q.shape
    bs, Hkv = kc.shape[1], kc.shape[2]
    out = np.zeros_like(q)
    for b in range(B):
        for h in range(Hq):
            kh = h // (Hq // Hkv)
            ks = [kc[bt[b, p // bs], p % bs, kh] for p in range(lens[b])]
            vs = [vc[bt[b, p // bs], p % bs, kh] for p in range(lens[b])]
            s = np.array([q[b, h] @ kk / math.sqrt(d) for kk in ks])
            w = np.exp(s - s.max())
            out[b, h] = (w[:, None] * np.array(vs)).sum(0) / w.sum()
    return out


def test_example():
    args = build([5, 12, 1])
    check_close(paged_decode(*args), ref(*args), rtol=1e-10, atol=1e-12, what="3 个序列")


def test_block_boundaries():
    for lens in [[4, 8, 3], [16], [1, 2, 3, 4, 5, 6, 7]]:
        args = build(lens, seed=len(lens))
        check_close(paged_decode(*args), ref(*args), rtol=1e-10, atol=1e-12, what=f"seq_lens={lens}")


def test_mqa_and_block_size():
    args = build([9, 30, 17], Hq=8, Hkv=1, d=16, bs=16, num_blocks=10, seed=5)
    check_close(paged_decode(*args), ref(*args), rtol=1e-10, atol=1e-12, what="MQA、block_size=16")


def test_invalid_entries_not_touched():
    """block table 里无效项是 -1：访问它会取到最后一个块，结果就错了"""
    q, kc, vc, bt, lens = build([6, 3], seed=9)
    kc2 = kc.copy()
    kc2[-1] = 1e6                                # 如果误用了 -1 这个下标，结果会被污染
    check_close(paged_decode(q, kc2, vc, bt, lens), ref(q, kc2, vc, bt, lens), rtol=1e-10, atol=1e-12,
                what="不访问无效的 block table 项")
