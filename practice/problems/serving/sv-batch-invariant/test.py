import math

import numpy as np

from checker import check, check_close
from solution import batch_decode, decode_attention, heuristic_batch_decode, sample


def make_requests(seed, lengths, d=64, dv=64):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(d).astype(np.float32), (rng.standard_normal((n, d)) * 2).astype(np.float32),
             rng.standard_normal((n, dv)).astype(np.float32)) for n in lengths]


def reference(q, K, V):
    s = K.astype(np.float64) @ q.astype(np.float64) / math.sqrt(len(q))
    p = np.exp(s - s.max())
    return p / p.sum() @ V.astype(np.float64)


def test_example():
    (q, K, V), = make_requests(0, [1000])
    out = decode_attention(q, K, V, 256)
    check(out.dtype, np.dtype(np.float32), "输出是 float32")
    check_close(out, reference(q, K, V), rtol=1e-4, atol=1e-5, what="4 段合并的结果等于完整的 softmax 注意力")


def test_split_lengths():
    (q, K, V), = make_requests(1, [777])
    ref = reference(q, K, V)
    for split in (1, 7, 64, 256, 777, 5000):
        check_close(decode_attention(q, K, V, split), ref, rtol=1e-4, atol=1e-5, what=f"split_len={split}")
    q2, K2, V2 = q, K * 30, V                                  # 分数很大：合并时要防止 exp 溢出
    out = decode_attention(q2, K2, V2, 100)
    check(bool(np.isfinite(out).all()), True, "分数很大时不能溢出")
    check_close(out, reference(q2, K2, V2), rtol=1e-3, atol=1e-4, what="分数很大时的结果")


def test_batch_invariance():
    mine = make_requests(2, [3000])[0]
    others = make_requests(3, [500, 4000, 64, 2500, 1200, 3100, 90, 700], d=64)
    alone = batch_decode([mine])[0]
    for k in (1, 3, 8):
        batch = others[:k] + [mine]
        got = batch_decode(batch)[-1]
        check(got.tobytes() == alone.tobytes(), True, f"和 {k} 个别的请求一起时，结果与单独运行逐位相同")
    shuffled = batch_decode([others[0], mine, others[1]])[1]
    check(shuffled.tobytes() == alone.tobytes(), True, "在 batch 中的位置不同，结果也逐位相同")
    h1 = heuristic_batch_decode([mine])[0]
    h8 = heuristic_batch_decode(others + [mine])[-1]
    check(h1.tobytes() != h8.tobytes(), True, "（对照）按 batch 大小决定段数时，结果会变——测试数据确实能暴露这个问题")


def test_sample_deterministic():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    a = [sample(probs, 42, pos) for pos in range(50)]
    b = [sample(probs, 42, pos) for pos in reversed(range(50))][::-1]
    check(a, b, "同一个 (seed, position) 无论什么顺序调用，结果都相同")
    check(sample(probs, 7, 3) == sample(probs, 7, 3), True, "重复调用结果相同")
    c = [sample(probs, 43, pos) for pos in range(50)]
    check(a != c, True, "换一个种子，采样序列应该不同")
    check(len(set(a)) > 1, True, "不同的位置应该采到不同的 token")


def test_sample_distribution():
    probs = np.array([0.5, 0.25, 0.125, 0.0, 0.125])
    counts = np.zeros(5)
    for seed in range(4000):
        for pos in range(5):
            counts[sample(probs, seed, pos)] += 1
    freq = counts / counts.sum()
    check_close(freq, probs, atol=0.015, rtol=0, what="采样频率接近给定的分布")
    check(counts[3], 0.0, "概率为 0 的 token 永远不会被采到")
    check(sample(np.array([0.0, 1.0]), 123, 0), 1, "只有一个可能的 token")
