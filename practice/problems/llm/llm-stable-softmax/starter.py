import numpy as np


def logsumexp(x, axis=-1):
    return np.log(np.sum(np.exp(x), axis=axis))


def softmax(x, axis=-1):
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    return np.log(softmax(x, axis))
