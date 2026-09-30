from collections import deque


class BlockPool:
    def __init__(self, num_blocks):
        self.free_list = deque(range(num_blocks))
        self.cache = {}
        self.refs = {}
        self.calls = self.hits = 0

    def allocate(self, block_hash):
        self.calls += 1
        if block_hash in self.cache:           # 没排除 None：所有不可复用的块会互相串用
            bid = self.cache[block_hash]
            self.hits += 1
            self.refs[bid] = self.refs.get(bid, 0) + 1
            return bid                         # 忘了把它从空闲队列里摘掉
        if not self.free_list:
            return -1
        bid = self.free_list.popleft()
        self.cache[block_hash] = bid           # 复用旧块时没有让旧哈希失效
        self.refs[bid] = 1
        return bid

    def free(self, block_id):
        self.refs[block_id] -= 1
        self.free_list.append(block_id)             # 引用计数还没归零就放回队列

    def num_free(self):
        return len(self.free_list)

    def hit_rate(self):
        return self.hits / self.calls if self.calls else 0.0
