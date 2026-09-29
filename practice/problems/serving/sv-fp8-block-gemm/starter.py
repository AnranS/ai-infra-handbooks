import numpy as np

FP8_MAX = 448.0


def e4m3(x):
    """舍入到最近的 E4M3 值（向偶数舍入），超过 448 的饱和"""
    return np.clip(np.asarray(x, dtype=np.float32), -FP8_MAX, FP8_MAX)       # 只做了饱和，没有舍入


def quant_act(a, group=128, pow2=False):
    pass


def quant_weight(w, block=128, pow2=False):
    pass


def block_gemm(aq, sa, wq, sw):
    pass
