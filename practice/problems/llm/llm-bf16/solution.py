import math

import numpy as np


def to_bf16_bits(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    rounded = ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)
    return np.where(np.isnan(x), np.uint16(0x7FC0), rounded).astype(np.uint16)


def from_bf16_bits(bits):
    b = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return b.view(np.float32)


def bf16_ulp_at(x):
    return 2.0 ** (math.floor(math.log2(x)) - 7)
