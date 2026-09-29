import numpy as np


def _angles(positions, dh, base):
    inv = base ** (-np.arange(0, dh, 2, dtype=np.float64) / dh)
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]       # (T, 1, dh/2)


def rope_half(x, positions, base=10000.0):
    dh = x.shape[-1]
    cos, sin = _angles(positions, dh, base)
    a, b = x[..., : dh // 2], x[..., dh // 2:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)


def rope_interleaved(x, positions, base=10000.0):
    dh = x.shape[-1]
    cos, sin = _angles(positions, dh, base)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x, dtype=np.result_type(x, np.float64))
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return out


def half_perm(dh):
    return np.concatenate([np.arange(0, dh, 2), np.arange(1, dh, 2)])
