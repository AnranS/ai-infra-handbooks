def count_params(c: dict) -> tuple[int, int]:
    V, d, L, dff = c["vocab_size"], c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"]
    nh = c["num_attention_heads"]
    nkv = c.get("num_key_value_heads", nh)
    hd = c.get("head_dim") or d // nh
    attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
    layer = attn + 3 * d * dff + 2 * d
    emb = V * d * (1 if c.get("tie_word_embeddings", False) else 2)
    total = emb + L * layer + d
    return total, total        # 没有处理 qk_norm 和 MoE
