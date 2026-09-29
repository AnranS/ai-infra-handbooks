import math

import numpy as np


def capacity(T, k, E, factor):
    return int(factor * T * k / E)                    # 向下取整：少了一份


def dispatch(topk_idx, topk_w, E, cap, policy="order"):
    pass


def combine(expert_out, topk_idx, topk_w, slot):
    pass


def aux_loss(logits, topk_idx):
    pass
