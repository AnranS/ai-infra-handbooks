import numpy as np

COMM = {"all_reduce": 0}


def silu(z):
    return z * (0.5 * (1 + np.tanh(0.5 * z)))


def _split(size, rank, n):
    if size % n:
        raise ValueError(f"维度 {size} 不能被 {n} 整除")
    k = size // n
    return slice(rank * k, (rank + 1) * k)


def shard_column(W, rank, n):
    return W[:, _split(W.shape[1], rank, n)]


def shard_row(W, rank, n):
    return W[_split(W.shape[0], rank, n), :]


def all_reduce(parts):
    COMM["all_reduce"] += 1
    total = sum(parts)
    return [total.copy() for _ in parts]


def tp_swiglu(x, w_gate, w_up, w_down, n):
    partial = []
    for r in range(n):
        h = silu(x @ shard_column(w_gate, r, n)) * (x @ shard_column(w_up, r, n))
        partial.append(h @ shard_row(w_down, r, n))
    return all_reduce(partial)[0]


def vocab_parallel_embedding(ids, E, n):
    V = E.shape[0]
    parts = []
    for r in range(n):
        sl = _split(V, r, n)
        local = shard_row(E, r, n)
        mask = (ids >= sl.start) & (ids < sl.stop)
        out = np.zeros(ids.shape + (E.shape[1],), dtype=E.dtype)
        out[mask] = local[ids[mask] - sl.start]
        parts.append(out)
    return all_reduce(parts)[0]
