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
        pass


class Sequence:
    def __init__(self, pool: BlockPool, block_size: int):
        self.pool, self.bs = pool, block_size
        self.num_tokens = 0
        self.block_table = []

    def append(self, n: int) -> None:
        need = math.ceil((self.num_tokens + n) / self.bs) - len(self.block_table)
        if need > 0:
            self.block_table += self.pool.allocate(need)
        self.num_tokens += n

    def fork(self) -> "Sequence":
        pass

    def slot_mapping(self, positions):
        pass

    def free(self) -> None:
        pass
