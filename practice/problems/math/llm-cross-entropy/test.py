import math

import numpy as np

from checker import check, check_close
from solution import cross_entropy, kl_divergence, perplexity


def ref_ce(logits, targets, ignore=-100):
    tot, n = 0.0, 0
    for row, t in zip(logits, targets):
        if t == ignore:
            continue
        m = max(row)
        lse = m + math.log(sum(math.exp(v - m) for v in row))
        tot += lse - row[t]
        n += 1
    return tot / n if n else 0.0


def test_example():
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    targets = np.array([0, 1])
    check_close(cross_entropy(logits, targets), ref_ce(logits.tolist(), targets.tolist()), what="交叉熵")
    check_close(perplexity(logits, targets), math.exp(ref_ce(logits.tolist(), targets.tolist())), what="困惑度")


def test_uniform():
    V = 50
    logits = np.zeros((7, V))
    check_close(cross_entropy(logits, np.arange(7)), math.log(V), what="均匀分布的交叉熵 = log V")
    check_close(perplexity(logits, np.arange(7)), V, what="均匀分布的困惑度 = V")


def test_ignore_index():
    logits = np.random.default_rng(0).standard_normal((5, 4))
    targets = np.array([1, -100, 3, -100, 0])
    check_close(cross_entropy(logits, targets), ref_ce(logits.tolist(), targets.tolist()), what="忽略两个位置")
    check(cross_entropy(logits, np.full(5, -100)), 0.0, "全部忽略时返回 0.0")
    check_close(cross_entropy(logits, np.array([1, 7, 3, 7, 0]), ignore_index=7),
                ref_ce(logits.tolist(), [1, 7, 3, 7, 0], 7), what="自定义 ignore_index（越界的值不能拿去索引）")


def test_stability():
    logits = np.array([[1000.0, 0.0, -1000.0], [-5000.0, 5000.0, 0.0]])
    ce = cross_entropy(logits, np.array([0, 1]))
    assert math.isfinite(ce), f"大 logits 时交叉熵应该是有限值，实际 {ce}"
    check_close(ce, 0.0, atol=1e-6, what="几乎确定正确的预测")


def test_kl():
    rng = np.random.default_rng(1)
    p, q = rng.standard_normal((6, 10)) * 3, rng.standard_normal((6, 10)) * 3
    kl = kl_divergence(p, q)
    check(kl.shape, (6,), "KL 的形状")
    P = np.exp(p) / np.exp(p).sum(-1, keepdims=True)
    Q = np.exp(q) / np.exp(q).sum(-1, keepdims=True)
    check_close(kl, (P * np.log(P / Q)).sum(-1), rtol=1e-7, what="KL(P||Q)")
    check_close(kl_divergence(p, p), np.zeros(6), atol=1e-12, what="KL(P||P) = 0")
    assert (kl >= -1e-12).all(), "KL 散度非负"
    big = np.array([[800.0, 0.0], [0.0, 800.0]])
    assert np.isfinite(kl_divergence(big, big[::-1])).all(), "大 logits 时 KL 也要是有限值"
