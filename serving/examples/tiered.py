"""tiered.py —— 两级前缀缓存：GPU 上的块被驱逐时，把内容存到 CPU 内存；之后命中时再拷回来，而不是重算。"""

from collections import OrderedDict

from prefix_cache import PrefixCachingBlockPool


class TieredPrefixCachingBlockPool(PrefixCachingBlockPool):
    def __init__(self, num_blocks: int, block_size: int, host_capacity_blocks: int):
        super().__init__(num_blocks, block_size)
        self.kv = None                                   # 引擎创建好 KV 张量后绑定
        self.host: OrderedDict[int, list] = OrderedDict()  # 块哈希 -> 各层的 (K, V)，按 LRU 排列
        self.host_capacity = host_capacity_blocks
        self.num_offloaded = self.num_loaded = 0

    def allocate(self, n: int):
        if n > len(self.free_queue):
            return None
        for b in list(self.free_queue)[:n]:              # 即将被复用的块：如果带着哈希，先卸载到 CPU
            h = self.block_hash[b]
            if h is not None and h not in self.host:
                self.host[h] = [(self.kv.k[l][b].clone(), self.kv.v[l][b].clone()) for l in range(len(self.kv.k))]
                self.num_offloaded += 1
                if len(self.host) > self.host_capacity:
                    self.host.popitem(last=False)
        return super().allocate(n)

    def lookup(self, token_ids: list[int]) -> list[int]:
        plan = []                                        # 每个命中块：(哈希, GPU 块号或 None 表示在 CPU 上)
        for h in self.block_hashes(token_ids):
            b = self.hash_to_block.get(h)
            if b is None and h not in self.host:
                break
            plan.append((h, b))
        # 先把 GPU 上命中、但处于空闲队列中的块摘出来，免得下面为 CPU 命中分配新块时把它们复用掉
        protected = [b for _, b in plan if b is not None and self.ref_cnt[b] == 0]
        for b in protected:
            del self.free_queue[b]
        need = sum(b is None for _, b in plan)
        new = self.allocate(need) if need else []
        if new is None:                                  # 放不下：只保留 GPU 上连续命中的部分
            new = []
            plan = plan[:next((i for i, (_, b) in enumerate(plan) if b is None), len(plan))]
        loaded = iter(new)
        hit = []
        for h, b in plan:
            if b is None:                                # CPU 命中：拷回刚分配的 GPU 块
                b = next(loaded)
                for l, (k, v) in enumerate(self.host[h]):
                    self.kv.k[l][b].copy_(k)
                    self.kv.v[l][b].copy_(v)
                self.host.move_to_end(h)
                self.block_hash[b], self.hash_to_block[h] = h, b
                self.ref_cnt[b] = 0                      # 以"空闲但已缓存"的状态登记，调度器随后的 touch 会引用它
                self.free_queue[b] = None
                self.num_loaded += 1
            hit.append(b)
        for b in protected:
            self.free_queue[b] = None
        self.num_queries += len(token_ids)
        self.num_hits += len(hit) * self.block_size
        return hit
