import numpy as np


def fp16_backward(true_grads, scale):
    """模拟 fp16 反向：损失乘以 scale 后，梯度在 fp16 里是 (真实梯度 × scale) 的舍入值，超出范围变成 inf"""
    with np.errstate(over="ignore"):
        return [(np.asarray(g, dtype=np.float32) * np.float32(scale)).astype(np.float16) for g in true_grads]


def underflow_fraction(true_grads, scale):
    pass


class GradScaler:
    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale

    def unscale(self, grads16):
        return [g.astype(np.float32) / self.scale for g in grads16], False     # 没有检查 inf

    def update(self, found_inf):
        pass

    def step(self, params, grads16, lr):
        grads, _ = self.unscale(grads16)
        for p, g in zip(params, grads):
            p -= lr * g
        return True
