def decode_attn_intensity(kind, n_heads, n_kv_heads=None, head_dim=128, kv_lora_rank=512, rope_dim=64,
                          q_len=1, dtype_bytes=2):
    if kind in ("mha", "gqa"):
        n_kv = n_heads if kind == "mha" or n_kv_heads is None else n_kv_heads
        flops = 4 * q_len * n_heads * head_dim
        nbytes = 2 * n_kv * head_dim * dtype_bytes
    elif kind == "mla":
        r, rho = kv_lora_rank, rope_dim
        flops = 2 * q_len * n_heads * (r + rho) + 2 * q_len * n_heads * r
        nbytes = (r + rho) * dtype_bytes
    else:
        raise ValueError(f"未知的注意力类型：{kind}")
    return flops / nbytes          # 都是"每个 KV token"的量，s 已经约掉


def tp_intensity(kind, n_heads, tp, n_kv_heads=None, **kw):
    heads = n_heads / tp
    if kind == "mla":
        return decode_attn_intensity("mla", heads, **kw)
    n_kv = n_heads if kind == "mha" or n_kv_heads is None else n_kv_heads
    return decode_attn_intensity("gqa", heads, n_kv_heads=max(n_kv / tp, 1), **kw)


def bound(intensity, peak_tflops, bw_gbs):
    ridge = peak_tflops * 1e12 / (bw_gbs * 1e9)
    return "compute" if intensity >= ridge else "memory"
