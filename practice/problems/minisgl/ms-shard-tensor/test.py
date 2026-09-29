import numpy as np

from checker import check, raises
from solution import div_even, shard_tensor


def test_example():
    q = np.arange(16 * 2).reshape(16, 2)
    check(shard_tensor("model.layers.0.self_attn.q_proj.weight", q, 1, 4, 2), q[4:8], "q_proj 的第 1 份")
    o = np.arange(2 * 16).reshape(2, 16)
    check(shard_tensor("model.layers.0.self_attn.o_proj.weight", o, 3, 4, 2), o[:, 12:16], "o_proj 的第 3 份")


def test_kv_replication():
    k = np.arange(8 * 3).reshape(8, 3)            # 2 个 KV 头，head_dim = 4
    got = [shard_tensor("x.self_attn.k_proj.weight", k, r, 4, 2).tolist() for r in range(4)]
    check(got, [k[0:4].tolist(), k[0:4].tolist(), k[4:8].tolist(), k[4:8].tolist()], "TP=4、2 个 KV 头：每两个 rank 共用一个头")
    got = shard_tensor("x.self_attn.v_proj.weight", k, 1, 2, 2)
    check(got, k[4:8], "KV 头够分时正常切分")


def test_vocab_parallel_uneven():
    E = np.arange(10 * 2).reshape(10, 2)
    parts = [shard_tensor("model.embed_tokens.weight", E, r, 3, 1) for r in range(3)]
    check([p.shape[0] for p in parts], [4, 4, 2], "10 行分 3 份：4、4、2")
    check(np.concatenate(parts), E, "拼回原矩阵")


def test_replicated_and_copies():
    n = np.ones(8)
    check(shard_tensor("model.norm.weight", n, 2, 4, 1), n, "norm 每个 rank 一份完整的")
    q = np.zeros((8, 2))
    s = shard_tensor("a.q_proj.weight", q, 0, 2, 1)
    assert not np.shares_memory(s, q), "分片要是副本"
    check(shard_tensor("a.q_proj.weight", q, 0, 1, 1) is q, True, "size=1 时原样返回")


def test_div_even():
    check((div_even(16, 4), div_even(8, 16, allow_replicate=True)), (4, 1), "div_even")
    with raises(ValueError, "div_even(10, 4)"):
        div_even(10, 4)
    with raises(ValueError, "8 个头分给 12 个 rank"):
        div_even(8, 12, allow_replicate=True)
