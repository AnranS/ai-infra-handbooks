import numpy as np


def ring_allreduce(data):
    n = len(data)
    chunks = [[c.copy() for c in np.array_split(np.asarray(x, dtype=np.float64), n)] for x in data]
    sent = [0] * n
    for s in range(n - 1):                                     # reduce-scatter
        msgs = [(r, (r - s) % n, chunks[r][(r - s) % n].copy()) for r in range(n)]
        for r, c, payload in msgs:
            chunks[(r + 1) % n][c] += payload
            sent[r] += payload.size
    for s in range(n - 1):                                     # all-gather
        msgs = [(r, (r + 1 - s) % n, chunks[r][(r + 1 - s) % n].copy()) for r in range(n)]
        for r, c, payload in msgs:
            chunks[(r + 1) % n][c] = payload
            sent[r] += payload.size
    return [np.concatenate(cs) for cs in chunks], sent
