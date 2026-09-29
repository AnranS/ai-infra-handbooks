from collections import OrderedDict


class CacheAwareRouter:
    def __init__(self, num_replicas, block_size, capacity_blocks, load_weight):
        self.n, self.bs, self.cap, self.w = num_replicas, block_size, capacity_blocks, load_weight
        self.load = [0] * num_replicas
        self.index = [OrderedDict() for _ in range(num_replicas)]

    def _hashes(self, token_ids):
        hashes, parent = [], None
        for start in range(0, len(token_ids) - self.bs + 1, self.bs):
            parent = hash((parent, tuple(token_ids[start:start + self.bs])))
            hashes.append(parent)
        return hashes

    def route(self, token_ids):
        hashes = self._hashes(token_ids)
        best = None
        for r in range(self.n):
            m = 0
            for h in hashes:
                if h not in self.index[r]:
                    break
                m += 1
            key = (-(m - self.w * self.load[r]), self.load[r], r)
            if best is None or key < best[0]:
                best = (key, r, m)
        _, r, m = best
        idx = self.index[r]
        for h in hashes:
            if h in idx:
                idx.move_to_end(h)
            else:
                idx[h] = None
                if len(idx) > self.cap:
                    idx.popitem(last=False)
        self.load[r] += 1
        return r, m

    def finish(self, replica):
        self.load[replica] -= 1
