QUALITY = ["F16", "Q8_0", "Q6_K", "Q4_K", "Q4_1", "Q4_0"]
BITS = {"F16": 16.0, "Q8_0": 8.5, "Q6_K": 6.5625, "Q4_K": 4.5, "Q4_1": 5.0, "Q4_0": 4.5}


def bits_per_weight(fmt):
    if fmt not in BITS:
        raise ValueError(f"不认识的格式 {fmt}")
    return BITS[fmt]


def kv_bytes(layers, kv_heads, head_dim, context, kv_bits=16):
    return 2 * layers * kv_heads * head_dim * context * kv_bits / 8


def decode_tps(weight_bytes, kv_bytes, bandwidth_gbs, efficiency=0.7):
    return efficiency * bandwidth_gbs * 1e9 / (weight_bytes + kv_bytes)


def pick_format(params, target_tps, bandwidth_gbs, efficiency=0.7):
    for fmt in QUALITY:                                   # 按质量从高到低，第一个够快的就是答案
        if decode_tps(params * BITS[fmt] / 8, 0, bandwidth_gbs, efficiency) >= target_tps:
            return fmt
    return None


def prefill_seconds(params, prompt_tokens, tflops, utilization=0.4):
    return 2 * params * prompt_tokens / (tflops * 1e12 * utilization)
