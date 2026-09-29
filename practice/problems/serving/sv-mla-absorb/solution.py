import numpy as np

# 权重的形状（W 是一个字典）：
#   W_q  (D, H * (NOPE + ROPE))  每个头的 query 是 [非位置部分 NOPE 维 | RoPE 部分 ROPE 维]
#   W_dkv (D, DC)                hidden → 潜向量 c（缓存它）
#   W_kr (D, ROPE)               hidden → 所有头共享的 RoPE 键（旋转后缓存）
#   W_uk (H, DC, NOPE)           潜向量 → 每个头 k 的非位置部分
#   W_uv (H, DC, DV)             潜向量 → 每个头的 v
#   W_o  (H * DV, D)             输出投影
# 维度记在 W["dims"] = (D, H, NOPE, ROPE, DV, DC)


def rope(x, pos):
    """按位置旋转最后一维的相邻两维。x: (..., ROPE)，pos: 与 x 去掉最后一维后形状相同（或能广播）的位置"""
    r = x.shape[-1]
    ang = np.asarray(pos, dtype=np.float64)[..., None] / 10000 ** (np.arange(0, r, 2) / r)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * np.cos(ang) - x2 * np.sin(ang)
    out[..., 1::2] = x1 * np.sin(ang) + x2 * np.cos(ang)
    return out


def mla_reference(h, W):
    """展开路径：从潜向量展开出每个头的 K、V，做普通的因果多头注意力。h: (T, D) → (T, D)"""
    D, H, NOPE, ROPE, DV, DC = W["dims"]
    T = h.shape[0]
    pos = np.arange(T)
    c = h @ W["W_dkv"]                                              # (T, DC)
    k_r = rope(h @ W["W_kr"], pos)                                  # (T, ROPE)
    q = (h @ W["W_q"]).reshape(T, H, NOPE + ROPE).transpose(1, 0, 2)
    q_n, q_r = q[..., :NOPE], rope(q[..., NOPE:], pos)
    k = np.concatenate([np.einsum("tc,hcd->htd", c, W["W_uk"]), np.broadcast_to(k_r, (H, T, ROPE))], -1)
    v = np.einsum("tc,hcd->htd", c, W["W_uv"])
    s = np.concatenate([q_n, q_r], -1) @ k.transpose(0, 2, 1) / np.sqrt(NOPE + ROPE)
    s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    out = (p @ v).transpose(1, 0, 2).reshape(T, H * DV)
    return out @ W["W_o"]


def absorb(W):
    """预先算好吸收后的矩阵"""
    D, H, NOPE, ROPE, DV, DC = W["dims"]
    wq = W["W_q"].reshape(D, H, NOPE + ROPE)
    W_q_lat = np.einsum("dhn,hcn->hdc", wq[..., :NOPE], W["W_uk"])      # hidden → 每个头在潜空间里的 query
    W_q_rope = wq[..., NOPE:].transpose(1, 0, 2)                         # (H, D, ROPE)
    W_o_lat = np.einsum("hcv,hvd->hcd", W["W_uv"], W["W_o"].reshape(H, DV, D))   # 潜向量 → 输出，合并了 W_uv 和 W_o
    return {"W_q_lat": W_q_lat, "W_q_rope": W_q_rope, "W_o_lat": W_o_lat, "W_dkv": W["W_dkv"], "W_kr": W["W_kr"],
            "scale": 1 / np.sqrt(NOPE + ROPE)}


def decode_step(x, pos, cache, A):
    """把新 token 的 c、kr 追加进 cache，只用潜向量缓存算出这个位置的输出 (D,)"""
    cache["c"].append(x @ A["W_dkv"])
    cache["kr"].append(rope(x @ A["W_kr"], pos))
    c, kr = np.stack(cache["c"]), np.stack(cache["kr"])                 # (n, DC)、(n, ROPE)
    q_lat = np.einsum("d,hdc->hc", x, A["W_q_lat"])                     # (H, DC)
    q_r = rope(np.einsum("d,hdr->hr", x, A["W_q_rope"]), pos)            # (H, ROPE)
    s = (q_lat @ c.T + q_r @ kr.T) * A["scale"]                          # (H, n)：所有头共用同一份缓存
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ctx = p @ c                                                          # (H, DC)：在潜空间里加权求和
    return np.einsum("hc,hcd->d", ctx, A["W_o_lat"])
