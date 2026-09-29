import numpy as np


def rms_norm(x, weight, eps=1e-6):
    dtype = x.dtype
    x32 = x.astype(np.float32)
    y = x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (y * weight.astype(np.float32)).astype(dtype)


def layer_norm(x, weight, bias, eps=1e-5):
    dtype = x.dtype
    x32 = x.astype(np.float32)
    mu = x32.mean(axis=-1, keepdims=True)
    var = ((x32 - mu) ** 2).mean(axis=-1, keepdims=True)
    y = (x32 - mu) / np.sqrt(var + eps)
    return (y * weight.astype(np.float32) + bias.astype(np.float32)).astype(dtype)


def pre_norm_block(x, sublayer, weight, eps=1e-6):
    return x + sublayer(rms_norm(x, weight, eps))


def fused_add_rms_norm(x, residual, weight, eps=1e-6):
    residual = (x.astype(np.float32) + residual.astype(np.float32)).astype(x.dtype)
    return rms_norm(residual, weight, eps), residual
