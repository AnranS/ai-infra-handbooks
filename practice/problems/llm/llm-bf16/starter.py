import numpy as np


def to_bf16_bits(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return (u >> 16).astype(np.uint16)      # 直接截断，不是就近舍入


def from_bf16_bits(bits):
    pass


def bf16_ulp_at(x):
    pass
