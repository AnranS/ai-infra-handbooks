import math

import numpy as np

SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
SPLIT_DIM_1 = [".o_proj", ".down_proj"]


def div_even(a, b, allow_replicate=False):
    if allow_replicate and b > a:
        if b % a:
            raise ValueError(f"{b} 不能被 {a} 整除，无法复制")
        return 1
    if a % b:
        raise ValueError(f"{a} 不能被 {b} 整除")
    return a // b


def shard_tensor(key, value, rank, size, num_kv_heads):
    if size == 1:
        return value
    if any(s in key for s in SPLIT_DIM_0):
        is_kv = ".k_proj" in key or ".v_proj" in key
        if is_kv and num_kv_heads < size:
            head_dim = value.shape[0] // num_kv_heads
            head = rank * num_kv_heads // size
            return value[head * head_dim:(head + 1) * head_dim].copy()
        return np.split(value, size, axis=0)[rank].copy()
    if any(s in key for s in SPLIT_DIM_1):
        return np.split(value, size, axis=1)[rank].copy()
    if "lm_head" in key or "embed_tokens" in key:
        per = math.ceil(value.shape[0] / size)
        return value[rank * per:min((rank + 1) * per, value.shape[0])].copy()
    return value
