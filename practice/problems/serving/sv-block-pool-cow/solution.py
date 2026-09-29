import math
from collections import deque


class BlockPool:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.ref_cnt = [0] * num_blocks
        self.free_queue = deque(range(num_blocks))
        self.copies = []

    def num_free(self) -> int:
        return len(self.free_queue)

    def allocate(self, n):
        if n > len(self.free_queue):
            return None
        blocks = [self.free_queue.popleft() for _ in range(n)]
        for b in blocks:
            self.ref_cnt[b] = 1
        return blocks

    def free(self, blocks):
        for b in blocks:
            self.ref_cnt[b] -= 1
            if self.ref_cnt[b] == 0:
                self.free_queue.append(b)

    def share(self, blocks):
        for b in blocks:
            self.ref_cnt[b] += 1


class Sequence:
    def __init__(self, pool: BlockPool, block_size: int):
        self.pool, self.bs = pool, block_size
        self.num_tokens = 0
        self.block_table = []

    def append(self, n: int) -> None:
        cow = self.num_tokens % self.bs != 0 and self.pool.ref_cnt[self.block_table[-1]] > 1
        need = math.ceil((self.num_tokens + n) / self.bs) - len(self.block_table)
        if self.pool.num_free() < int(cow) + max(need, 0):
            raise MemoryError("KV Cache 空间不够")
        if cow:
            old = self.block_table[-1]
            (new,) = self.pool.allocate(1)
            self.pool.free([old])
            self.pool.copies.append((old, new))
            self.block_table[-1] = new
        if need > 0:
            self.block_table += self.pool.allocate(need)
        self.num_tokens += n

    def fork(self) -> "Sequence":
        child = Sequence(self.pool, self.bs)
        child.num_tokens = self.num_tokens
        child.block_table = list(self.block_table)
        self.pool.share(self.block_table)
        return child

    def slot_mapping(self, positions):
        return [self.block_table[p // self.bs] * self.bs + p % self.bs for p in positions]

    def free(self) -> None:
        self.pool.free(self.block_table)
        self.block_table = []
        self.num_tokens = 0
