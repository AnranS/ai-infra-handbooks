import numpy as np


def fp16_backward(true_grads, scale):
    """模拟 fp16 反向：损失乘以 scale 后，梯度在 fp16 里是 (真实梯度 × scale) 的舍入值，超出范围变成 inf"""
    with np.errstate(over="ignore"):
        return [(np.asarray(g, dtype=np.float32) * np.float32(scale)).astype(np.float16) for g in true_grads]


def underflow_fraction(true_grads, scale):
    nonzero = lost = 0
    for g, g16 in zip(true_grads, fp16_backward(true_grads, scale)):
        g = np.asarray(g)
        nonzero += int((g != 0).sum())
        lost += int(((g != 0) & (g16 == 0)).sum())
    return lost / nonzero if nonzero else 0.0


class GradScaler:
    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale
        self.growth_factor, self.backoff_factor, self.growth_interval = growth_factor, backoff_factor, growth_interval
        self.good_steps = 0                                   # 连续没有溢出的步数

    def unscale(self, grads16):
        inv = np.float32(1.0 / self.scale)
        grads = [g.astype(np.float32) * inv for g in grads16]
        found_inf = any(not np.isfinite(g).all() for g in grads)
        return grads, found_inf

    def update(self, found_inf):
        if found_inf:
            self.scale *= self.backoff_factor
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps == self.growth_interval:
                self.scale *= self.growth_factor
                self.good_steps = 0

    def step(self, params, grads16, lr):
        grads, found_inf = self.unscale(grads16)
        if not found_inf:                                    # 溢出的一步整个跳过
            for p, g in zip(params, grads):
                p -= lr * g
        self.update(found_inf)
        return not found_inf
