import numpy as np


def rms_norm(x, weight, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * weight


def forward(tokens, w, cfg):
    x = w["embed"][tokens]
    # TODO：各层的注意力（RoPE + GQA + 因果）和 SwiGLU，最后的 final_norm 和输出层
    return rms_norm(x, w["final_norm"], cfg["eps"]) @ w["embed"].T


def greedy_generate(prompt, w, cfg, n):
    pass
