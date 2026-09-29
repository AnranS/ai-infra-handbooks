import numpy as np


def log_softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    return np.log(np.exp(z) / np.exp(z).sum(-1, keepdims=True))      # 不稳定：logits 大时溢出


def kd_loss(student, teacher, T):
    pass


def kd_grad(student, teacher, T):
    pass


def reverse_kl(student, teacher):
    pass


def topk_kd_loss(student, topk_ids, topk_logprobs):
    pass
