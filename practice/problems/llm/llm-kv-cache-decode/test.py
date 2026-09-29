import math

import numpy as np

import solution
from checker import check, check_close
from solution import IncrementalDecoder, forward

CFG = {"vocab": 40, "d": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "d_ff": 48, "rope_base": 10000.0, "eps": 1e-6}


def make_weights(cfg, seed=0):
    rng = np.random.default_rng(seed)
    d, dh = cfg["d"], cfg["d"] // cfg["n_heads"]
    r = lambda *s: rng.standard_normal(s) / math.sqrt(s[0])  # noqa: E731
    layers = [{"attn_norm": 1 + 0.1 * rng.standard_normal(d), "mlp_norm": 1 + 0.1 * rng.standard_normal(d),
               "wq": r(d, cfg["n_heads"] * dh), "wk": r(d, cfg["n_kv_heads"] * dh), "wv": r(d, cfg["n_kv_heads"] * dh),
               "wo": r(cfg["n_heads"] * dh, d), "w_gate": r(d, cfg["d_ff"]), "w_up": r(d, cfg["d_ff"]),
               "w_down": r(cfg["d_ff"], d)} for _ in range(cfg["n_layers"])]
    return {"embed": rng.standard_normal((cfg["vocab"], d)), "layers": layers,
            "final_norm": 1 + 0.1 * rng.standard_normal(d)}


def no_forward(*a, **k):
    raise AssertionError("step() 里调用了完整的 forward()：应该只算新 token")


def test_example():
    w = make_weights(CFG)
    prompt = [1, 5, 9, 2]
    dec = IncrementalDecoder(w, CFG)
    pre = dec.prefill(prompt)
    check_close(pre, forward(np.array(prompt), w, CFG), rtol=1e-9, atol=1e-9, what="prefill 的 logits")
    check(dec.length, 4, "prefill 之后的缓存长度")
    orig = solution.forward
    solution.forward = no_forward
    try:
        out = dec.step(7)
    finally:
        solution.forward = orig
    check_close(out, forward(np.array(prompt + [7]), w, CFG)[-1], rtol=1e-8, atol=1e-8, what="第一步 decode 的 logits")
    check(dec.length, 5, "decode 一步之后的缓存长度")


def test_greedy_generation_matches():
    w = make_weights(CFG, seed=1)
    toks = [3, 3, 8]
    dec = IncrementalDecoder(w, CFG)
    logits = dec.prefill(toks)[-1]
    orig = solution.forward
    solution.forward = no_forward
    gen = []
    try:
        for _ in range(12):
            t = int(np.argmax(logits))
            gen.append(t)
            logits = dec.step(t)
    finally:
        solution.forward = orig
    full = list(toks)
    want = []
    for _ in range(12):
        t = int(np.argmax(forward(np.array(full), w, CFG)[-1]))
        want.append(t)
        full.append(t)
    check(gen, want, "用 KV Cache 贪心生成 12 个 token")


def test_chunked_prefill():
    """prefill 可以分几段做（第二段的 query 要看到第一段的 KV）"""
    w = make_weights(CFG, seed=2)
    toks = [4, 8, 15, 16, 23, 42 % 40, 7]
    dec = IncrementalDecoder(w, CFG)
    a = dec.prefill(toks[:3])
    b = dec.prefill(toks[3:])
    check_close(np.concatenate([a, b]), forward(np.array(toks), w, CFG), rtol=1e-8, atol=1e-8, what="分两段 prefill")
    check(dec.length, 7, "缓存长度")


def test_mqa_config():
    cfg = dict(CFG, n_kv_heads=1, n_layers=3)
    w = make_weights(cfg, seed=3)
    dec = IncrementalDecoder(w, cfg)
    dec.prefill([0, 1])
    outs = [dec.step(t) for t in [2, 3, 4]]
    full = forward(np.array([0, 1, 2, 3, 4]), w, cfg)
    check_close(np.stack(outs), full[2:], rtol=1e-8, atol=1e-8, what="MQA、3 层")
