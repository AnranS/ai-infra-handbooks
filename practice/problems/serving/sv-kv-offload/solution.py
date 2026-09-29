from collections import OrderedDict


class TieredKVCache:
    def __init__(self, gpu_blocks, cpu_blocks):
        self.gpu_cap, self.cpu_cap = gpu_blocks, cpu_blocks
        self.gpu, self.cpu = OrderedDict(), OrderedDict()
        self.stats = {k: 0 for k in ("gpu_hits", "cpu_hits", "misses", "offloads", "loads", "drops")}

    def _evict_gpu(self):
        h, _ = self.gpu.popitem(last=False)
        if self.cpu_cap == 0:
            self.stats["drops"] += 1
            return
        if len(self.cpu) >= self.cpu_cap:
            self.cpu.popitem(last=False)
            self.stats["drops"] += 1
        self.cpu[h] = None
        self.stats["offloads"] += 1

    def _put_gpu(self, h):
        if len(self.gpu) >= self.gpu_cap:
            self._evict_gpu()
        self.gpu[h] = None

    def access(self, h):
        if h in self.gpu:
            self.gpu.move_to_end(h)
            self.stats["gpu_hits"] += 1
            return "gpu"
        if h in self.cpu:
            del self.cpu[h]
            self.stats["cpu_hits"] += 1
            self.stats["loads"] += 1
            self._put_gpu(h)
            return "cpu"
        self.stats["misses"] += 1
        self._put_gpu(h)
        return "miss"

    def gpu_contents(self):
        return list(self.gpu)

    def cpu_contents(self):
        return list(self.cpu)
