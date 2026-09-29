def decode_attn_intensity(kind, n_heads, n_kv_heads=None, head_dim=128, kv_lora_rank=512, rope_dim=64,
                          q_len=1, dtype_bytes=2):
    pass


def tp_intensity(kind, n_heads, tp, n_kv_heads=None, **kw):
    pass


def bound(intensity, peak_tflops, bw_gbs):
    pass
