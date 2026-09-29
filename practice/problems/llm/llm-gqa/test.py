import math

import numpy as np

from checker import check, check_close
from solution import gqa_attention, kv_cache_bytes, repeat_kv


def ref(q, k, v):
    T, Hq, dh = q.shape
    S, Hkv, _ = k.shape
    rep = Hq // Hkv
    out = np.zeros_like(q)
    for h in range(Hq):
        kh = h // rep
        for i in range(T):
            js = range(S - T + i + 1)
            s = [q[i, h] @ k[j, kh] / math.sqrt(dh) for j in js]
            m = max(s)
            w = [math.exp(x - m) for x in s]
            out[i, h] = sum(wi * v[j, kh] for wi, j in zip(w, js)) / sum(w)
    return out


def rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def test_example():
    check(kv_cache_bytes(28, 8, 128, 32768), 3_758_096_384, "Qwen3-0.6B 32K token 的 KV Cache")
    kv = np.arange(2 * 2 * 1).reshape(2, 2, 1)
    check(repeat_kv(kv, 3)[0, :, 0].tolist(), [0, 0, 0, 1, 1, 1], "repeat_kv 的头顺序")


def test_prefill():
    q, k, v = rand(6, 8, 4, seed=1), rand(6, 2, 4, seed=2), rand(6, 2, 4, seed=3)
    check_close(gqa_attention(q, k, v), ref(q, k, v), rtol=1e-9, atol=1e-9, what="prefill（S = T）")


def test_decode_with_cache():
    q, k, v = rand(1, 4, 8, seed=4), rand(10, 1, 8, seed=5), rand(10, 1, 8, seed=6)
    check_close(gqa_attention(q, k, v), ref(q, k, v), rtol=1e-9, atol=1e-9, what="decode（T=1，MQA）")


def test_chunk_with_prefix():
    q, k, v = rand(3, 6, 4, seed=7), rand(9, 3, 4, seed=8), rand(9, 3, 4, seed=9)
    check_close(gqa_attention(q, k, v), ref(q, k, v), rtol=1e-9, atol=1e-9, what="带前缀缓存的一段（T=3, S=9）")


def test_mha_special_case():
    q, k, v = rand(5, 4, 4, seed=10), rand(5, 4, 4, seed=11), rand(5, 4, 4, seed=12)
    check_close(gqa_attention(q, k, v), ref(q, k, v), rtol=1e-9, atol=1e-9, what="H_q = H_kv（普通多头注意力）")


def test_kv_bytes_variants():
    check(kv_cache_bytes(32, 32, 128, 4096), 2 * 32 * 32 * 128 * 4096 * 2, "LLaMA-2-7B（MHA）4K token")
    check(kv_cache_bytes(80, 8, 128, 1, bytes_per_elem=1), 2 * 80 * 8 * 128, "70B 模型每个 token，FP8 KV")
