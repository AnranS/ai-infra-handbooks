import math

import numpy as np

from checker import check, check_close
from solution import forward, greedy_generate

CFG = {"vocab": 50, "d": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "d_ff": 64, "rope_base": 10000.0,
       "eps": 1e-6}


def make_weights(cfg, seed=0):
    rng = np.random.default_rng(seed)
    d, dh = cfg["d"], cfg["d"] // cfg["n_heads"]
    r = lambda *s: rng.standard_normal(s) / math.sqrt(s[0])  # noqa: E731
    layers = [{
        "attn_norm": 1 + 0.1 * rng.standard_normal(d), "mlp_norm": 1 + 0.1 * rng.standard_normal(d),
        "wq": r(d, cfg["n_heads"] * dh), "wk": r(d, cfg["n_kv_heads"] * dh), "wv": r(d, cfg["n_kv_heads"] * dh),
        "wo": r(cfg["n_heads"] * dh, d), "w_gate": r(d, cfg["d_ff"]), "w_up": r(d, cfg["d_ff"]),
        "w_down": r(cfg["d_ff"], d)} for _ in range(cfg["n_layers"])]
    return {"embed": rng.standard_normal((cfg["vocab"], d)), "layers": layers, "final_norm": 1 + 0.1 * rng.standard_normal(d)}


def ref_forward(tokens, w, cfg):
    """逐 token、逐头的朴素实现"""
    T, d, H, Hkv = len(tokens), cfg["d"], cfg["n_heads"], cfg["n_kv_heads"]
    dh = d // H
    norm = lambda v, g: v / math.sqrt(float((v * v).mean()) + cfg["eps"]) * g  # noqa: E731

    def rot(vec, pos):
        out = vec.copy()
        for i in range(dh // 2):
            th = pos * cfg["rope_base"] ** (-2 * i / dh)
            a, b = vec[i], vec[i + dh // 2]
            out[i], out[i + dh // 2] = a * math.cos(th) - b * math.sin(th), a * math.sin(th) + b * math.cos(th)
        return out

    x = [w["embed"][t].copy() for t in tokens]
    for lw in w["layers"]:
        hs = [norm(v, lw["attn_norm"]) for v in x]
        q = [(h @ lw["wq"]).reshape(H, dh) for h in hs]
        k = [(h @ lw["wk"]).reshape(Hkv, dh) for h in hs]
        vv = [(h @ lw["wv"]).reshape(Hkv, dh) for h in hs]
        q = [np.stack([rot(q[t][h], t) for h in range(H)]) for t in range(T)]
        k = [np.stack([rot(k[t][h], t) for h in range(Hkv)]) for t in range(T)]
        new = []
        for t in range(T):
            heads = []
            for h in range(H):
                kh = h // (H // Hkv)
                s = [q[t][h] @ k[j][kh] / math.sqrt(dh) for j in range(t + 1)]
                m = max(s)
                e = [math.exp(z - m) for z in s]
                heads.append(sum(ei * vv[j][kh] for ei, j in zip(e, range(t + 1))) / sum(e))
            new.append(x[t] + np.concatenate(heads) @ lw["wo"])
        x = new
        out = []
        for v in x:
            h = norm(v, lw["mlp_norm"])
            g = h @ lw["w_gate"]
            out.append(v + (g / (1 + np.exp(-g)) * (h @ lw["w_up"])) @ lw["w_down"])
        x = out
    return np.stack([norm(v, w["final_norm"]) @ w["embed"].T for v in x])


def test_example():
    w = make_weights(CFG)
    tokens = np.array([3, 17, 42, 5, 9])
    got = forward(tokens, w, CFG)
    check(got.shape, (5, 50), "logits 形状")
    check_close(got, ref_forward(tokens, w, CFG), rtol=1e-8, atol=1e-8, what="logits")


def test_causality():
    """后面的 token 不影响前面位置的 logits"""
    w = make_weights(CFG, seed=1)
    a = forward(np.array([1, 2, 3, 4]), w, CFG)
    b = forward(np.array([1, 2, 3, 49]), w, CFG)
    check_close(a[:3], b[:3], rtol=1e-10, atol=1e-10, what="前 3 个位置的 logits")


def test_other_config():
    cfg = dict(CFG, d=48, n_layers=3, n_heads=6, n_kv_heads=1, d_ff=40, rope_base=1e6, vocab=30)
    w = make_weights(cfg, seed=2)
    tokens = np.array([0, 29, 7, 7, 13, 2, 1])
    check_close(forward(tokens, w, cfg), ref_forward(tokens, w, cfg), rtol=1e-8, atol=1e-8, what="MQA、3 层、base=1e6")


def test_greedy_generate():
    w = make_weights(CFG, seed=3)
    prompt = [5, 6, 7]
    got = greedy_generate(prompt, w, CFG, 6)
    toks = list(prompt)
    want = []
    for _ in range(6):
        nxt = int(np.argmax(ref_forward(toks, w, CFG)[-1]))
        toks.append(nxt)
        want.append(nxt)
    check(got, want, "贪心生成的 6 个 token")
