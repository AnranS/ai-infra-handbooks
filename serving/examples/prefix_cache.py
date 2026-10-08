"""prefix_cache.py —— vLLM 式的前缀缓存：按块做链式哈希，空闲块兼作 LRU 缓存。

- 每个"装满的"块有一个哈希 = hash(父块哈希, 本块 token)，因此哈希相同 ⇔ 从开头到这个块的所有 token 都相同；
- 引用计数降为 0 的块进入空闲队列，但内容和哈希保留，仍可被命中；
- 分配新块时从空闲队列头部取，如果取到的块带着哈希，就把它从缓存中"驱逐"——队列顺序即 LRU 顺序。
"""

from collections import OrderedDict

from paged import BlockPool


class PrefixCachingBlockPool(BlockPool):
    enable_caching = True

    def __init__(self, num_blocks: int, block_size: int):
        super().__init__(num_blocks)
        self.block_size = block_size
        self.free_queue = OrderedDict((b, None) for b in range(num_blocks))   # 头部最先被复用
        self.hash_to_block: dict[int, int] = {}
        self.block_hash: list[int | None] = [None] * num_blocks
        self.num_queries = self.num_hits = 0                                   # 以 token 计

    def allocate(self, n: int) -> list[int] | None:
        if n > len(self.free_queue):
            return None
        blocks = []
        for _ in range(n):
            b, _ = self.free_queue.popitem(last=False)
            if self.block_hash[b] is not None:                                 # 驱逐：这个块要被新内容覆盖了
                del self.hash_to_block[self.block_hash[b]]
                self.block_hash[b] = None
            self.ref_cnt[b] = 1
            blocks.append(b)
        return blocks

    def free(self, blocks: list[int]) -> None:
        # 倒序释放：序列尾部的块最不可能被别人共享，让它们排在前面、先被驱逐
        for b in reversed(blocks):
            self.ref_cnt[b] -= 1
            if self.ref_cnt[b] == 0:
                self.free_queue[b] = None

    def touch(self, blocks: list[int]) -> None:
        """命中的块被新请求引用：引用计数 +1；如果它在空闲队列里，就移出来。"""
        for b in blocks:
            if self.ref_cnt[b] == 0:
                del self.free_queue[b]
            self.ref_cnt[b] += 1

    def block_hashes(self, token_ids: list[int]) -> list[int]:
        hashes, parent = [], None
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            parent = hash((parent, tuple(token_ids[start:start + self.block_size])))
            hashes.append(parent)
        return hashes

    def lookup(self, token_ids: list[int]) -> list[int]:
        """返回最长的已缓存前缀对应的块（只看装满的块）。"""
        hit = []
        for h in self.block_hashes(token_ids):
            b = self.hash_to_block.get(h)
            if b is None:
                break
            hit.append(b)
        self.num_queries += len(token_ids)
        self.num_hits += len(hit) * self.block_size
        return hit

    def cache_blocks(self, token_ids: list[int], block_ids: list[int], num_computed: int) -> None:
        """前向之后调用：把新装满的块登记进缓存。"""
        for i, h in enumerate(self.block_hashes(token_ids[:num_computed])):
            b = block_ids[i]
            if self.block_hash[b] is None and h not in self.hash_to_block:
                self.block_hash[b] = h
                self.hash_to_block[h] = b
