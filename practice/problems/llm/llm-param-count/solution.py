def count_params(c: dict) -> tuple[int, int]:
    V, d, L = c["vocab_size"], c["hidden_size"], c["num_hidden_layers"]
    nh = c["num_attention_heads"]
    nkv = c.get("num_key_value_heads", nh)
    hd = c.get("head_dim") or d // nh
    attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
    layer = attn + 2 * d + (2 * hd if c.get("qk_norm", False) else 0)
    E = c.get("num_experts", 0) or 0
    inactive = 0
    if E:
        k, dmoe = c["num_experts_per_tok"], c["moe_intermediate_size"]
        layer += d * E + E * 3 * d * dmoe
        inactive = L * (E - k) * 3 * d * dmoe
    else:
        layer += 3 * d * c["intermediate_size"]
    emb = V * d * (1 if c.get("tie_word_embeddings", False) else 2)
    total = emb + L * layer + d
    return total, total - inactive
