from collections import OrderedDict


class CacheAwareRouter:
    def __init__(self, num_replicas, block_size, capacity_blocks, load_weight):
        self.n, self.bs, self.cap, self.w = num_replicas, block_size, capacity_blocks, load_weight
        self.load = [0] * num_replicas
        self.rr = 0

    def route(self, token_ids):
        r = self.rr % self.n                 # 轮询：完全不考虑缓存
        self.rr += 1
        self.load[r] += 1
        return r, 0

    def finish(self, replica):
        self.load[replica] -= 1
