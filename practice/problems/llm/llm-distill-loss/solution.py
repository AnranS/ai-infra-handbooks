import numpy as np


def log_softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(-1, keepdims=True)                   # 先减最大值，exp 不会溢出
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def kd_loss(student, teacher, T):
    lp, lq = log_softmax(teacher, T), log_softmax(student, T)
    return float(T * T * (np.exp(lp) * (lp - lq)).sum(-1).mean())


def kd_grad(student, teacher, T):
    p, q = np.exp(log_softmax(teacher, T)), np.exp(log_softmax(student, T))
    return T * (q - p) / np.asarray(student).shape[0]


def reverse_kl(student, teacher):
    lp, lq = log_softmax(teacher), log_softmax(student)
    return float((np.exp(lq) * (lq - lp)).sum(-1).mean())


def topk_kd_loss(student, topk_ids, topk_logprobs):
    lq = np.take_along_axis(log_softmax(student), np.asarray(topk_ids), axis=-1)   # 学生在全词表上的 log 概率
    lp = log_softmax(topk_logprobs)                                             # 老师的 top-k 重新归一化
    return float((np.exp(lp) * (lp - lq)).sum(-1).mean())
