def kv_bytes(tokens, layers, kv_heads, head_dim, dtype_bytes):
    return tokens * layers * kv_heads * head_dim * dtype_bytes


def swap_ms(nbytes, gbps, overhead_ms=0.0):
    return nbytes / (gbps * 1e9) * 1e3 + overhead_ms


def recompute_ms(tokens, params, tflops):
    return 2 * params * tokens / (tflops * 1e12) * 1e3


def choose(tokens, model, gbps, tflops, overhead_ms=0.0):
    s = swap_ms(kv_bytes(tokens, model["layers"], model["kv_heads"], model["head_dim"], model["dtype_bytes"]), gbps, overhead_ms)
    r = recompute_ms(tokens, model["params"], tflops)
    return ("swap" if s <= r else "recompute", s, r)
