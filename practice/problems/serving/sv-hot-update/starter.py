import numpy as np


def quantize(w):
    scale = np.abs(w).max(axis=1, keepdims=True) / 127
    return np.round(w / scale).astype(np.int8), scale


def hot_update(layer, new_w):
    layer.qweight, layer.scale = quantize(new_w)      # 换成了新数组：CUDA Graph 还在读旧的存储


def pause_costs(mode, done, left, prompt, step_ms):
    pass
