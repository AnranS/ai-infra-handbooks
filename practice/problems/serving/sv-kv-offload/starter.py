from collections import OrderedDict


class TieredKVCache:
    def __init__(self, gpu_blocks, cpu_blocks):
        self.gpu_cap, self.cpu_cap = gpu_blocks, cpu_blocks
        self.gpu, self.cpu = OrderedDict(), OrderedDict()
        self.stats = {k: 0 for k in ("gpu_hits", "cpu_hits", "misses", "offloads", "loads", "drops")}

    def access(self, h):
        # 只有 GPU 一级：淘汰的块直接丢掉
        if h in self.gpu:
            self.gpu.move_to_end(h)
            self.stats["gpu_hits"] += 1
            return "gpu"
        self.stats["misses"] += 1
        if len(self.gpu) >= self.gpu_cap:
            self.gpu.popitem(last=False)
            self.stats["drops"] += 1
        self.gpu[h] = None
        return "miss"

    def gpu_contents(self):
        return list(self.gpu)

    def cpu_contents(self):
        return list(self.cpu)
