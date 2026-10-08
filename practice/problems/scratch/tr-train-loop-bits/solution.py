import math

import numpy as np


def lr_at(step, lr, min_lr, warmup, max_steps):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def clip_by_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(np.square(g, dtype=np.float64))) for g in grads))
    if total <= max_norm:
        return [g.copy() for g in grads], total
    scale = max_norm / (total + 1e-6)
    return [g * scale for g in grads], total


def decay_groups(named_shapes):
    decay = [n for n, s in named_shapes.items() if len(s) >= 2]
    no_decay = [n for n, s in named_shapes.items() if len(s) < 2]
    return decay, no_decay
