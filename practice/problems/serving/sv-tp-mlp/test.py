import numpy as np

import solution
from checker import check, check_close, raises
from solution import shard_column, shard_row, tp_swiglu, vocab_parallel_embedding


def silu(z):
    return z / (1 + np.exp(-z))


def weights(d=16, dff=48, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((5, d)), rng.standard_normal((d, dff)) * 0.3, rng.standard_normal((d, dff)) * 0.3,
            rng.standard_normal((dff, d)) * 0.3)


def test_example():
    x, g, u, dn = weights()
    solution.COMM["all_reduce"] = 0
    got = tp_swiglu(x, g, u, dn, 4)
    check_close(got, (silu(x @ g) * (x @ u)) @ dn, rtol=1e-9, atol=1e-10, what="TP=4 的 SwiGLU 输出")
    check(solution.COMM["all_reduce"], 1, "all-reduce 次数")


def test_shards():
    W = np.arange(24).reshape(4, 6)
    check(shard_column(W, 1, 3), W[:, 2:4], "列并行的第 1 份")
    check(shard_row(W, 1, 2), W[2:4, :], "行并行的第 1 份")
    shards = [shard_column(W, r, 2) for r in range(2)]
    check(np.concatenate(shards, axis=1), W, "列分片拼回原矩阵")
    with raises(ValueError, "6 列分给 4 份"):
        shard_column(W, 0, 4)


def test_tp_sizes():
    x, g, u, dn = weights(dff=48, seed=1)
    want = (silu(x @ g) * (x @ u)) @ dn
    for n in [1, 2, 3, 6, 8]:
        check_close(tp_swiglu(x, g, u, dn, n), want, rtol=1e-9, atol=1e-10, what=f"TP={n}")


def test_vocab_parallel_embedding():
    rng = np.random.default_rng(2)
    E = rng.standard_normal((40, 8))
    ids = rng.integers(0, 40, (3, 7))
    solution.COMM["all_reduce"] = 0
    check_close(vocab_parallel_embedding(ids, E, 4), E[ids], what="词表并行嵌入")
    check(solution.COMM["all_reduce"], 1, "all-reduce 次数")
    check_close(vocab_parallel_embedding(np.array([0, 39, 20]), E, 5), E[[0, 39, 20]], what="边界 token")
