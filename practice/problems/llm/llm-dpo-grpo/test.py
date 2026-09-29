import math

import numpy as np

from checker import check, check_close
from solution import dpo_loss, grpo_advantages, sequence_logprob


def ref_seq(logits, tokens, mask):
    B, T, V = logits.shape
    out = np.zeros(B)
    for b in range(B):
        for t in range(T - 1):
            if mask[b, t + 1]:
                row = logits[b, t]
                m = row.max()
                out[b] += row[tokens[b, t + 1]] - m - math.log(np.exp(row - m).sum())
    return out


def test_example():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 5, 7))
    tokens = rng.integers(0, 7, (2, 5))
    mask = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]], dtype=bool)
    check_close(sequence_logprob(logits, tokens, mask), ref_seq(logits, tokens, mask), rtol=1e-10,
                what="序列对数概率（注意错位一格）")
    loss, acc = dpo_loss(np.array([-10.0, -12.0]), np.array([-11.0, -9.0]), np.array([-10.5, -11.0]),
                         np.array([-10.5, -10.0]))
    m = np.array([(-10 + 10.5) - (-11 + 10.5), (-12 + 11) - (-9 + 10)])
    check_close(loss, float(np.mean([math.log(1 + math.exp(-0.1 * x)) for x in m])), rtol=1e-10, what="DPO loss")
    check(acc, 0.5, "reward_acc")


def test_dpo_extremes():
    big = np.array([1e5, -1e5])
    loss, acc = dpo_loss(big, np.zeros(2), np.zeros(2), np.zeros(2), beta=1.0)
    assert math.isfinite(loss), f"margin 很大时损失应该是有限值，实际 {loss}"
    check_close(loss, 1e5 / 2, rtol=1e-9, what="一个样本几乎 0 损失、一个样本损失约 1e5")
    check(acc, 0.5, "reward_acc")


def test_dpo_zero_margin():
    z = np.zeros(4)
    check_close(dpo_loss(z, z, z, z)[0], math.log(2), rtol=1e-12, what="策略与参考模型相同时损失为 log 2")


def test_grpo():
    r = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    adv = grpo_advantages(r, 4)
    g1 = np.array([1.0, 0.0, 1.0, 1.0])
    check_close(adv[:4], (g1 - g1.mean()) / (g1.std() + 1e-6), rtol=1e-10, what="第一组的优势")
    check_close(adv[4:], np.zeros(4), atol=1e-12, what="全错的一组优势全为 0")
    check(adv.shape, (8,), "形状")


def test_seq_logprob_prompt_masking():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((1, 6, 5))
    tokens = rng.integers(0, 5, (1, 6))
    full = sequence_logprob(logits, tokens, np.ones((1, 6), dtype=bool))
    ans = sequence_logprob(logits, tokens, np.array([[0, 0, 0, 1, 1, 1]], dtype=bool))
    prompt = sequence_logprob(logits, tokens, np.array([[1, 1, 1, 0, 0, 0]], dtype=bool))
    check_close(full, ans + prompt, rtol=1e-10, what="提示词部分 + 回答部分 = 全部")
