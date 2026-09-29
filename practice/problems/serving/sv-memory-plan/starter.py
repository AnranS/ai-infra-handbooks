import math

WEIGHT_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}
KV_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1}


def plan(cfg, gpu_mem_gb, weight_dtype, kv_dtype, context_len, util=0.9, activation_gb=2.0):
    pass
