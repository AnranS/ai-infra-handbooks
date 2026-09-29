def _expert(cfg):
    return 3 * cfg["d"] * cfg["moe_ffn"]


def moe_params(cfg):
    d, L = cfg["d"], cfg["n_layers"]
    n_dense = cfg.get("n_dense_layers", 0)
    n_moe = L - n_dense
    common = (L * cfg["attn_params"] + n_dense * 3 * d * cfg.get("dense_ffn", 0)
              + n_moe * d * cfg["n_experts"] + 2 * cfg["vocab"] * d)
    shared = cfg.get("n_shared", 0)
    total = common + n_moe * (cfg["n_experts"] + shared) * _expert(cfg)
    active = common + n_moe * (cfg["top_k"] + shared) * _expert(cfg)
    return total, active


def expected_distinct_experts(n_experts, top_k, n_tokens):
    return n_experts * (1 - (1 - top_k / n_experts) ** n_tokens)


def decode_expert_bytes(cfg, batch, bytes_per_param=1):
    n_moe = cfg["n_layers"] - cfg.get("n_dense_layers", 0)
    per_layer = expected_distinct_experts(cfg["n_experts"], cfg["top_k"], batch) + cfg.get("n_shared", 0)
    return n_moe * per_layer * _expert(cfg) * bytes_per_param
