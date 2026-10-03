import numpy as np


def low_rank(W, r):
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    s = np.sqrt(S[:r])
    return U[:, :r] * s, s[:, None] * Vt[:r]


def rank_for_energy(W, ratio):
    S = np.linalg.svd(W, compute_uv=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    return int(np.argmax(energy >= ratio - 1e-12)) + 1


def lora_params(shapes, r):
    full = sum(m * n for m, n in shapes)
    lora = sum(r * (m + n) for m, n in shapes)
    return full, lora, lora / full
