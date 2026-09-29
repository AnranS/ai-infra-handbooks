import numpy as np

COMM = {"all_reduce": 0}


def silu(z):
    return z * (0.5 * (1 + np.tanh(0.5 * z)))


def shard_column(W, rank, n):
    pass


def shard_row(W, rank, n):
    pass


def all_reduce(parts):
    COMM["all_reduce"] += 1
    total = sum(parts)
    return [total.copy() for _ in parts]


def tp_swiglu(x, w_gate, w_up, w_down, n):
    pass


def vocab_parallel_embedding(ids, E, n):
    pass
