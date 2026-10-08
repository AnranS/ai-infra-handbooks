import random

import numpy as np

from checker import check
from solution import cu_seqlens, doc_mask, pack, positions


def test_example():
    check(pack([[1, 2, 3], [4, 5], [6, 7, 8, 9]], seq_len=4, eos=0), [[1, 2, 3, 0, 4], [4, 5, 0, 6, 7]], "打包")
    check(positions([1, 2, 3, 0, 4], 0), [0, 1, 2, 3, 0], "位置编号")
    check(cu_seqlens([1, 2, 3, 0, 4], 0), [0, 4, 5], "段边界")


def test_pack_random():
    rng = random.Random(0)
    for trial in range(50):
        docs = [[rng.randint(1, 9) for _ in range(rng.randint(1, 12))] for _ in range(rng.randint(1, 8))]
        seq_len = rng.randint(1, 10)
        stream = [t for d in docs for t in d + [0]]
        samples = pack(docs, seq_len, 0)
        check(len(samples), (len(stream) - 1) // seq_len, f"第 {trial} 组：样本数")
        for i, s in enumerate(samples):
            check(s, stream[i * seq_len:i * seq_len + seq_len + 1], f"第 {trial} 组第 {i} 个样本")


def test_segments():
    toks = [5, 0, 0, 7, 8, 0]
    check(positions(toks, 0), [0, 1, 0, 0, 1, 2], "连续两个 eos：第二个 eos 自成一段")
    check(cu_seqlens(toks, 0), [0, 2, 3, 6], "最后一个 token 是 eos 时不重复结尾")
    check(cu_seqlens([3, 4, 5], 0), [0, 3], "没有 eos：一整段")
    m = doc_mask([1, 2, 0, 3, 4], 0)
    want = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1]], dtype=bool)
    check(np.asarray(m), want, "块对角的因果掩码")


def test_consistency():
    rng = random.Random(1)
    for trial in range(40):
        toks = [rng.choice([0, 1, 2, 3]) for _ in range(rng.randint(1, 20))]
        pos, cu, m = positions(toks, 0), cu_seqlens(toks, 0), np.asarray(doc_mask(toks, 0))
        check(cu[0] == 0 and cu[-1] == len(toks) and all(a < b for a, b in zip(cu, cu[1:])), True, f"第 {trial} 组：边界递增")
        for a, b in zip(cu, cu[1:]):
            check(pos[a:b], list(range(b - a)), f"第 {trial} 组：段 [{a}, {b}) 里的位置从 0 开始")
        check(m.sum(1).tolist(), [p + 1 for p in pos], f"第 {trial} 组：每个 token 看到的个数 = 它的位置 + 1")
