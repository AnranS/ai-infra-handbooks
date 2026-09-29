QUALITY = ["F16", "Q8_0", "Q6_K", "Q4_K", "Q4_1", "Q4_0"]


def bits_per_weight(fmt):
    return {"F16": 16, "Q8_0": 8, "Q4_0": 4}[fmt]          # 忘了缩放占的比特，也缺了几种格式


def kv_bytes(layers, kv_heads, head_dim, context, kv_bits=16):
    pass


def decode_tps(weight_bytes, kv_bytes, bandwidth_gbs, efficiency=0.7):
    pass


def pick_format(params, target_tps, bandwidth_gbs, efficiency=0.7):
    pass


def prefill_seconds(params, prompt_tokens, tflops, utilization=0.4):
    pass
