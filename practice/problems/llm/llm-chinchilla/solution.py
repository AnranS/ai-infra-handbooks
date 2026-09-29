import math

import numpy as np


def fit_power_law(x, y):
    b, log_a = np.polyfit(np.log(np.asarray(x, dtype=np.float64)), np.log(np.asarray(y, dtype=np.float64)), 1)
    return float(math.exp(log_a)), float(b)


def chinchilla_optimal(compute_flops, tokens_per_param=20):
    n = math.sqrt(compute_flops / (6 * tokens_per_param))
    return n, tokens_per_param * n


def train_days(n_params, n_tokens, n_gpus, peak_tflops, mfu):
    seconds = 6 * n_params * n_tokens / (n_gpus * peak_tflops * 1e12 * mfu)
    return seconds / 86400
