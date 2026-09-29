import numpy as np

from checker import check, check_close
from solution import ring_attention_zigzag, zigzag_order


def causal_attention(q, k, v):
    s = q @ k.T / np.sqrt(q.shape[1])
    s = np.where(np.tril(np.ones(s.shape, dtype=bool)), s, -np.inf)
    p = np.exp(s - s.max(1, keepdims=True))
    return p @ v / p.sum(1, keepdims=True)


def data(S, d=8, dv=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((S, d)), rng.standard_normal((S, d)), rng.standard_normal((S, dv))


def test_example():
    check(zigzag_order(4), [[0, 7], [1, 6], [2, 5], [3, 4]], "P=4 的之字形切分")
    q, k, v = data(24)
    out, work = ring_attention_zigzag(q, k, v, P=3)
    check_close(out, causal_attention(q, k, v), rtol=1e-9, atol=1e-10, what="与单卡的因果注意力一致")


def test_work_balanced():
    for P, S in ((2, 16), (3, 24), (4, 64)):
        q, k, v = data(S, seed=P)
        out, work = ring_attention_zigzag(q, k, v, P)
        c = S // (2 * P)
        check(len(work), P, f"P={P}：一共 {P} 步")
        check(work[0], [2 * c * c + c] * P, f"P={P}：第 0 步每个 rank 算自己的两块（两个对角块 + 一个整块）")
        for i in range(1, P):
            check(work[i], [2 * c * c] * P, f"P={P}：第 {i} 步每个 rank 都是 2c²")
        check(sum(map(sum, work)), S * (S + 1) // 2, f"P={P}：总工作量等于因果注意力的可见对数")
        check_close(out, causal_attention(q, k, v), rtol=1e-9, atol=1e-10, what=f"P={P}：输出")


def test_single_rank_and_large_scores():
    q, k, v = data(10, seed=5)
    out, work = ring_attention_zigzag(q, k, v, P=1)
    check(work, [[5 * 5 + 5 * 6]], "P=1：两块 5 个 token，一个整块 + 两个对角块")
    check_close(out, causal_attention(q, k, v), rtol=1e-9, atol=1e-10, what="P=1 的输出")
    q2, k2, v2 = data(32, seed=6)
    q2 *= 30                                                  # 分数很大：合并时要防止溢出
    out2, _ = ring_attention_zigzag(q2, k2, v2, P=4)
    check(bool(np.isfinite(out2).all()), True, "分数很大时不能出现 inf / nan")
    check_close(out2, causal_attention(q2, k2, v2), rtol=1e-7, atol=1e-9, what="分数很大时的输出")
