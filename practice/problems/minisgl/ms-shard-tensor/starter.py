import math

import numpy as np

SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
SPLIT_DIM_1 = [".o_proj", ".down_proj"]


def div_even(a, b, allow_replicate=False):
    return a // b


def shard_tensor(key, value, rank, size, num_kv_heads):
    if any(s in key for s in SPLIT_DIM_0):
        return np.split(value, size, axis=0)[rank]
    if any(s in key for s in SPLIT_DIM_1):
        return np.split(value, size, axis=1)[rank]
    return value
