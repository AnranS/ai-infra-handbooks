import math


def _per_layer_token(cfg, dtype_bytes):
    kind = cfg["attn"]
    if kind == "mha":
        return 2 * cfg["n_heads"] * cfg["head_dim"] * dtype_bytes
    if kind == "gqa":
        return 2 * cfg["n_kv_heads"] * cfg["head_dim"] * dtype_bytes
    if kind == "mla":
        return (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * dtype_bytes
    raise ValueError(f"未知的注意力类型：{kind}")


def kv_bytes_per_token(cfg, dtype_bytes=2):
    return _per_layer_token(cfg, dtype_bytes) * cfg["n_layers"]


def kv_bytes(cfg, seq_len, dtype_bytes=2):
    per = _per_layer_token(cfg, dtype_bytes)
    L = cfg["n_layers"]
    if "window" not in cfg:
        return per * L * seq_len
    g = cfg.get("n_global_layers", 0)
    return per * (g * seq_len + (L - g) * min(seq_len, cfg["window"]))


def max_context(cfg, budget_gib, dtype_bytes=2):
    budget = budget_gib * 2 ** 30
    per = _per_layer_token(cfg, dtype_bytes)
    L = cfg["n_layers"]
    if "window" not in cfg or kv_bytes(cfg, cfg["window"], dtype_bytes) > budget:
        return int(budget // (per * L))
    g = cfg.get("n_global_layers", 0)
    if g == 0:
        return math.inf
    return int((budget / per - (L - g) * cfg["window"]) // g)
