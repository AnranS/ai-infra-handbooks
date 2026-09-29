import numpy as np


def quantize(W, group_size):
    pass


def dequantize(q, scales, zeros, group_size):
    pass


def pack(q):
    pass


def unpack(packed):
    pass


def gemv_w4(x, packed, scales, zeros, group_size):
    return x @ dequantize(unpack(packed), scales, zeros, group_size)   # 先解出完整矩阵，不符合要求
