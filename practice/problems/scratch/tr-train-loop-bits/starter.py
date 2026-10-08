import math

import numpy as np


def lr_at(step, lr, min_lr, warmup, max_steps):
    if step < warmup:
        return lr * step / warmup                     # 第 0 步学习率为 0：这一步白算了
    progress = (step - warmup) / (max_steps - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))     # 衰减到 0 而不是 min_lr


def clip_by_global_norm(grads, max_norm):
    pass


def decay_groups(named_shapes):
    pass
