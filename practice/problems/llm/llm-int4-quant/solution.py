import numpy as np


def quantize(W, group_size):
    K, N = W.shape
    Wg = W.astype(np.float32).reshape(K // group_size, group_size, N)
    lo = np.minimum(Wg.min(axis=1), 0)
    hi = np.maximum(Wg.max(axis=1), 0)
    scales = (hi - lo) / 15
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    zeros = np.round(-lo / scales).astype(np.float32)
    q = np.clip(np.round(Wg / scales[:, None, :]) + zeros[:, None, :], 0, 15)
    return q.reshape(K, N).astype(np.uint8), scales, zeros.astype(np.uint8)


def dequantize(q, scales, zeros, group_size):
    K, N = q.shape
    qg = q.reshape(K // group_size, group_size, N).astype(np.float32)
    return ((qg - zeros[:, None, :].astype(np.float32)) * scales[:, None, :]).reshape(K, N).astype(np.float32)


def pack(q):
    return (q[0::2] | (q[1::2] << 4)).astype(np.uint8)


def unpack(packed):
    out = np.empty((packed.shape[0] * 2, packed.shape[1]), dtype=np.uint8)
    out[0::2] = packed & 0xF
    out[1::2] = packed >> 4
    return out


def gemv_w4(x, packed, scales, zeros, group_size):
    N = packed.shape[1]
    y = np.zeros(N, dtype=np.float32)
    half = group_size // 2
    for gi in range(scales.shape[0]):
        rows = unpack(packed[gi * half:(gi + 1) * half]).astype(np.float32)          # (group_size, N)
        w = (rows - zeros[gi].astype(np.float32)) * scales[gi]
        y += x[gi * group_size:(gi + 1) * group_size].astype(np.float32) @ w
    return y
