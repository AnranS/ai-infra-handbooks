import tracemalloc

import numpy as np

from checker import check, check_close
from solution import flash_attn_fwd


def ref(q, k, v, causal):
    T, S, d = q.shape[0], k.shape[0], q.shape[1]
    s = q @ k.T / np.sqrt(d)
    if causal:
        s = np.where(np.arange(S)[None] <= (S - T + np.arange(T))[:, None], s, -np.inf)
    m = s.max(1, keepdims=True)
    p = np.exp(s - m)
    l = p.sum(1, keepdims=True)
    return p @ v / l, (m + np.log(l))[:, 0]


def rand(T, S, d, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, d)), rng.standard_normal((S, d)), rng.standard_normal((S, d))


def test_example():
    q, k, v = rand(100, 100, 32)
    o, lse = flash_attn_fwd(q, k, v, Br=16, Bc=16)
    ro, rl = ref(q, k, v, False)
    check_close(o, ro, rtol=1e-9, atol=1e-10, what="输出")
    check_close(lse, rl, rtol=1e-9, atol=1e-10, what="lse")


def test_causal_and_odd_blocks():
    for T, S, Br, Bc in [(37, 37, 8, 16), (10, 50, 4, 7), (1, 30, 64, 8), (64, 64, 64, 64)]:
        q, k, v = rand(T, S, 16, seed=T + S)
        for causal in (False, True):
            o, lse = flash_attn_fwd(q, k, v, causal=causal, Br=Br, Bc=Bc)
            ro, rl = ref(q, k, v, causal)
            check_close(o, ro, rtol=1e-9, atol=1e-10, what=f"T={T}, S={S}, Br={Br}, Bc={Bc}, causal={causal} 的输出")
            check_close(lse, rl, rtol=1e-9, atol=1e-10, what=f"T={T}, S={S}, causal={causal} 的 lse")


def test_large_scores_stable():
    q, k, v = rand(20, 20, 8, seed=3)
    o, lse = flash_attn_fwd(q * 50, k * 50, v, causal=True, Br=8, Bc=8)
    assert np.isfinite(o).all() and np.isfinite(lse).all(), "分数很大时也不能出现 inf/nan"
    check_close(o, ref(q * 50, k * 50, v, True)[0], rtol=1e-7, atol=1e-9, what="大分数下的输出")


def test_memory_is_tiled():
    q, k, v = rand(1024, 1024, 64, seed=4)
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    o, _ = flash_attn_fwd(q, k, v, causal=True, Br=64, Bc=64)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    check(o.shape, (1024, 64), "输出形状")
    if peak == 0:
        return                         # 当前环境不支持追踪 numpy 的内存分配
    assert peak < 4 * 1024 * 1024, f"峰值额外内存 {peak / 2**20:.1f} MB：看起来构造了完整的 1024×1024 分数矩阵（8 MB）"
