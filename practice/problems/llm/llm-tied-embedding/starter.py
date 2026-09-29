import numpy as np


class TiedLM:
    def __init__(self, E):
        self.E = np.asarray(E)

    def embed(self, ids):
        return self.E[ids]

    def logits(self, h):
        pass

    def pad_batch(self, seqs, pad_id):
        pass

    def last_token_logits(self, seqs, pad_id):
        ids, mask = self.pad_batch(seqs, pad_id)
        return self.logits(self.embed(ids[:, -1]))     # bug：短序列取到的是 pad
