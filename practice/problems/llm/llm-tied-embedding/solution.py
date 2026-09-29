import numpy as np


class TiedLM:
    def __init__(self, E):
        self.E = np.asarray(E)

    def embed(self, ids):
        ids = np.asarray(ids)
        V = self.E.shape[0]
        if ids.size and (ids.min() < 0 or ids.max() >= V):
            raise IndexError(f"token id 越界：词表大小是 {V}")
        return self.E[ids]

    def logits(self, h):
        return np.asarray(h) @ self.E.T

    def pad_batch(self, seqs, pad_id):
        B, T = len(seqs), max((len(s) for s in seqs), default=0)
        ids = np.full((B, T), pad_id, dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
            mask[i, :len(s)] = True
        return ids, mask

    def last_token_logits(self, seqs, pad_id):
        ids, mask = self.pad_batch(seqs, pad_id)
        last = ids[np.arange(len(seqs)), mask.sum(axis=1) - 1]
        return self.logits(self.embed(last))
