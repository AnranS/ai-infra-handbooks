import numpy as np


def rms_norm(x, weight, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * weight


def rope(x, positions, base):
    dh = x.shape[-1]
    inv = base ** (-np.arange(0, dh, 2) / dh)
    ang = positions[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., : dh // 2], x[..., dh // 2:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)


def attention(x, lw, cfg, positions):
    T = x.shape[0]
    H, Hkv = cfg["n_heads"], cfg["n_kv_heads"]
    dh = cfg["d"] // H
    q = (x @ lw["wq"]).reshape(T, H, dh)
    k = (x @ lw["wk"]).reshape(T, Hkv, dh)
    v = (x @ lw["wv"]).reshape(T, Hkv, dh)
    q, k = rope(q, positions, cfg["rope_base"]), rope(k, positions, cfg["rope_base"])
    k, v = np.repeat(k, H // Hkv, axis=1), np.repeat(v, H // Hkv, axis=1)
    s = np.einsum("thd,shd->hts", q, k) / np.sqrt(dh)
    s = np.where(np.tril(np.ones((T, T), dtype=bool))[None], s, -np.inf)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", p, v).reshape(T, H * dh) @ lw["wo"]


def swiglu(x, lw):
    g = x @ lw["w_gate"]
    return (g / (1 + np.exp(-g)) * (x @ lw["w_up"])) @ lw["w_down"]


def forward(tokens, w, cfg):
    tokens = np.asarray(tokens)
    positions = np.arange(len(tokens), dtype=np.float64)
    x = w["embed"][tokens]
    for lw in w["layers"]:
        x = x + attention(rms_norm(x, lw["attn_norm"], cfg["eps"]), lw, cfg, positions)
        x = x + swiglu(rms_norm(x, lw["mlp_norm"], cfg["eps"]), lw)
    return rms_norm(x, w["final_norm"], cfg["eps"]) @ w["embed"].T


def greedy_generate(prompt, w, cfg, n):
    tokens = list(prompt)
    out = []
    for _ in range(n):
        nxt = int(np.argmax(forward(np.array(tokens), w, cfg)[-1]))
        tokens.append(nxt)
        out.append(nxt)
    return out
