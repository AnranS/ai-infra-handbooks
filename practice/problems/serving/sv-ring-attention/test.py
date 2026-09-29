import numpy as np

from checker import check, check_close
from solution import context_parallel_attention, merge, partial_attention


def full(q, k, v, causal):
    S, T, d = len(k), len(q), q.shape[1]
    s = q @ k.T / np.sqrt(d)
    if causal:
        s = np.where(np.arange(S)[None] <= np.arange(S - T, S)[:, None], s, -np.inf)
    m = s.max(1, keepdims=True)
    e = np.exp(s - m)
    return e @ v / e.sum(1, keepdims=True), (m[:, 0] + np.log(e.sum(1)))


def rand(T, S, d, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, d)), rng.standard_normal((S, d)), rng.standard_normal((S, d))


def test_example():
    q, k, v = rand(6, 40, 8)
    o, lse = context_parallel_attention(q, k, v, 4, causal=False)
    fo, fl = full(q, k, v, False)
    check_close(o, fo, rtol=1e-10, atol=1e-12, what="合并后的输出")
    check_close(lse, fl, rtol=1e-10, atol=1e-12, what="合并后的 lse")


def test_causal_prefill_many_splits():
    q, k, v = rand(30, 30, 4, seed=1)
    fo, fl = full(q, k, v, True)
    for n in [1, 2, 3, 7, 30]:
        o, lse = context_parallel_attention(q, k, v, n, causal=True)
        assert not np.isnan(o).any(), f"n={n} 时出现了 nan：有的分段对某些 query 完全不可见"
        check_close(o, fo, rtol=1e-9, atol=1e-11, what=f"因果、切成 {n} 段")
        check_close(lse, fl, rtol=1e-9, atol=1e-11, what=f"因果、切成 {n} 段的 lse")


def test_partial_all_masked():
    q, k, v = rand(2, 3, 4, seed=2)
    o, lse = partial_attention(q, k, v, np.array([0, 1]), np.array([5, 6, 7]), causal=True)
    check_close(o, np.zeros((2, 4)), what="看不到任何 key 时输出 0")
    check(bool(np.all(np.isneginf(lse))), True, "看不到任何 key 时 lse = -inf")


def test_merge_properties():
    q, k, v = rand(3, 20, 4, seed=3)
    a = partial_attention(q, k[:12], v[:12], np.arange(17, 20), np.arange(12), False)
    b = partial_attention(q, k[12:], v[12:], np.arange(17, 20), np.arange(12, 20), False)
    o1, l1 = merge([a, b])
    o2, l2 = merge([b, a])
    check_close(o1, o2, rtol=1e-12, atol=1e-14, what="合并与顺序无关")
    big = (a[0], a[1] + 800.0)
    o3, l3 = merge([big, b])
    assert np.isfinite(o3).all() and np.isfinite(l3).all(), "lse 很大时也不能溢出"
    check_close(o3, a[0], rtol=1e-9, atol=1e-12, what="一段的 lse 远大于其他段时，结果就是这一段")
