from collections import deque


class BlockPool:
    def __init__(self, num_blocks):
        self.free_list = deque(range(num_blocks))   # 队首是最久未使用的
        self.cache = {}                        # 块哈希 -> 块号
        self.hash_of = {}                      # 块号 -> 块哈希
        self.refs = {}                         # 块号 -> 引用计数
        self.calls = self.hits = 0

    def allocate(self, block_hash):
        self.calls += 1
        if block_hash is not None and block_hash in self.cache:
            bid = self.cache[block_hash]
            self.hits += 1
            if self.refs.get(bid, 0) == 0:     # 还在空闲队列里，要摘出来
                self.free_list.remove(bid)
            self.refs[bid] = self.refs.get(bid, 0) + 1
            return bid
        if not self.free_list:
            return -1
        bid = self.free_list.popleft()
        old = self.hash_of.get(bid)
        if old is not None and self.cache.get(old) == bid:
            del self.cache[old]                # 旧内容失效
        self.hash_of[bid] = block_hash
        if block_hash is not None:
            self.cache[block_hash] = bid
        self.refs[bid] = 1
        return bid

    def free(self, block_id):
        if self.refs.get(block_id, 0) == 0:
            return
        self.refs[block_id] -= 1
        if self.refs[block_id] == 0:
            self.free_list.append(block_id)         # 放到队尾：最近使用的最后被复用

    def num_free(self):
        return len(self.free_list)

    def hit_rate(self):
        return self.hits / self.calls if self.calls else 0.0
