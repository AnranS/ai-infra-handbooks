import numpy as np


def logsumexp(x, axis=-1):
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


def softmax(x, axis=-1):
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    s = np.sum(e, axis=axis, keepdims=True)
    return np.divide(e, s, out=np.zeros_like(e), where=s > 0)


def log_softmax(x, axis=-1):
    x = np.asarray(x)
    return x - np.expand_dims(logsumexp(x, axis), axis)
