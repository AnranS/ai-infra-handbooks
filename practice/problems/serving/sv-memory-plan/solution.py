import math

WEIGHT_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}
KV_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1}
GB = 2 ** 30


def plan(cfg, gpu_mem_gb, weight_dtype, kv_dtype, context_len, util=0.9, activation_gb=2.0):
    if weight_dtype == "int4":
        weight_bytes = cfg["params"] * 0.5 + math.ceil(cfg["params"] / 128) * 2
    elif weight_dtype in WEIGHT_BYTES:
        weight_bytes = cfg["params"] * WEIGHT_BYTES[weight_dtype]
    else:
        raise ValueError(f"未知的权重类型：{weight_dtype}")
    if kv_dtype not in KV_BYTES:
        raise ValueError(f"未知的 KV 类型：{kv_dtype}")
    per_token = 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * KV_BYTES[kv_dtype]
    free = gpu_mem_gb * GB * util - weight_bytes - activation_gb * GB
    kv_tokens = max(0, int(free // per_token))
    max_seqs = kv_tokens // context_len
    return {"weight_gb": weight_bytes / GB, "kv_bytes_per_token": per_token, "kv_tokens": kv_tokens,
            "max_seqs": max_seqs, "fits": max_seqs >= 1}
