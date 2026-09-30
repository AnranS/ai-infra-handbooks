def ridge(tflops, gbs):
    return tflops * 1e12 / (gbs * 1e9)


def breakeven_batch(tflops, gbs, weight_bytes=2):
    return ridge(tflops, gbs) * weight_bytes / 2


def attn_intensity(group_size, kv_bytes=2):
    return 2 * group_size / kv_bytes


def decode_floor_ms(weight_gb, gbs):
    return weight_gb * 1e9 / (gbs * 1e9) * 1e3
