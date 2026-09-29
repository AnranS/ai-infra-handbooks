import numpy as np


def rms_norm(x, weight, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * weight


def rope(x, positions, base):
    dh = x.shape[-1]
    inv = base ** (-np.arange(0, dh, 2) / dh)
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., : dh // 2], x[..., dh // 2:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)


def swiglu(x, lw):
    g = x @ lw["w_gate"]
    return (g / (1 + np.exp(-g)) * (x @ lw["w_up"])) @ lw["w_down"]


def attend(q, k, v, q_positions):
    """q: (T, H, dh)；k、v: (S, H, dh)（已经按 GQA 展开）；第 i 个 query 只能看到位置 <= q_positions[i] 的 key。"""
    S = k.shape[0]
    s = np.einsum("thd,shd->hts", q, k) / np.sqrt(q.shape[-1])
    allowed = np.arange(S)[None, :] <= np.asarray(q_positions)[:, None]
    s = np.where(allowed[None], s, -np.inf)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", p, v)


def forward(tokens, w, cfg):
    """完整前向：每次都重新算整个序列。"""
    tokens = np.asarray(tokens)
    T = len(tokens)
    H, Hkv = cfg["n_heads"], cfg["n_kv_heads"]
    dh = cfg["d"] // H
    pos = np.arange(T)
    x = w["embed"][tokens]
    for lw in w["layers"]:
        h = rms_norm(x, lw["attn_norm"], cfg["eps"])
        q = rope((h @ lw["wq"]).reshape(T, H, dh), pos, cfg["rope_base"])
        k = rope((h @ lw["wk"]).reshape(T, Hkv, dh), pos, cfg["rope_base"])
        v = (h @ lw["wv"]).reshape(T, Hkv, dh)
        k, v = np.repeat(k, H // Hkv, axis=1), np.repeat(v, H // Hkv, axis=1)
        x = x + attend(q, k, v, pos).reshape(T, H * dh) @ lw["wo"]
        x = x + swiglu(rms_norm(x, lw["mlp_norm"], cfg["eps"]), lw)
    return rms_norm(x, w["final_norm"], cfg["eps"]) @ w["embed"].T


class IncrementalDecoder:
    def __init__(self, w, cfg):
        self.w, self.cfg = w, cfg
        self.k = [None] * len(w["layers"])
        self.v = [None] * len(w["layers"])
        self._len = 0

    @property
    def length(self):
        return self._len

    def _run(self, tokens):
        cfg, w = self.cfg, self.w
        T = len(tokens)
        H, Hkv = cfg["n_heads"], cfg["n_kv_heads"]
        dh = cfg["d"] // H
        pos = self._len + np.arange(T)
        x = w["embed"][np.asarray(tokens)]
        for i, lw in enumerate(w["layers"]):
            h = rms_norm(x, lw["attn_norm"], cfg["eps"])
            q = rope((h @ lw["wq"]).reshape(T, H, dh), pos, cfg["rope_base"])
            k = rope((h @ lw["wk"]).reshape(T, Hkv, dh), pos, cfg["rope_base"])
            v = (h @ lw["wv"]).reshape(T, Hkv, dh)
            self.k[i] = k if self.k[i] is None else np.concatenate([self.k[i], k])
            self.v[i] = v if self.v[i] is None else np.concatenate([self.v[i], v])
            kk = np.repeat(self.k[i], H // Hkv, axis=1)
            vv = np.repeat(self.v[i], H // Hkv, axis=1)
            x = x + attend(q, kk, vv, pos).reshape(T, H * dh) @ lw["wo"]
            x = x + swiglu(rms_norm(x, lw["mlp_norm"], cfg["eps"]), lw)
        self._len += T
        return rms_norm(x, w["final_norm"], cfg["eps"]) @ w["embed"].T

    def prefill(self, tokens):
        return self._run(list(tokens))

    def step(self, token):
        return self._run([int(token)])[0]
