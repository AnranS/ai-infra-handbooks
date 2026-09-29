from collections import OrderedDict


class PrefixCachingBlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.ref_cnt = [0] * num_blocks
        self.free_queue = OrderedDict((b, None) for b in range(num_blocks))
        self.hash_to_block = {}
        self.block_hash = [None] * num_blocks
        self.num_queries = self.num_hits = 0

    def num_free(self):
        return len(self.free_queue)

    def allocate(self, n):
        if n > len(self.free_queue):
            return None
        blocks = []
        for _ in range(n):
            b, _ = self.free_queue.popitem(last=False)
            if self.block_hash[b] is not None:
                del self.hash_to_block[self.block_hash[b]]
                self.block_hash[b] = None
            self.ref_cnt[b] = 1
            blocks.append(b)
        return blocks

    def free(self, blocks):
        for b in reversed(blocks):
            self.ref_cnt[b] -= 1
            if self.ref_cnt[b] == 0:
                self.free_queue[b] = None

    def touch(self, blocks):
        for b in blocks:
            if self.ref_cnt[b] == 0:
                del self.free_queue[b]
            self.ref_cnt[b] += 1

    def block_hashes(self, token_ids, extra_key=None):
        hashes, parent = [], None
        for i, start in enumerate(range(0, len(token_ids) - self.block_size + 1, self.block_size)):
            parent = hash((extra_key if i == 0 else parent, tuple(token_ids[start:start + self.block_size])))
            hashes.append(parent)
        return hashes

    def lookup(self, token_ids, extra_key=None):
        hit = []
        for h in self.block_hashes(token_ids, extra_key):
            b = self.hash_to_block.get(h)
            if b is None:
                break
            hit.append(b)
        self.num_queries += len(token_ids)
        self.num_hits += len(hit) * self.block_size
        return hit

    def cache_blocks(self, token_ids, block_ids, num_computed, extra_key=None):
        for i, h in enumerate(self.block_hashes(token_ids[:num_computed], extra_key)):
            b = block_ids[i]
            if self.block_hash[b] is None and h not in self.hash_to_block:
                self.block_hash[b] = h
                self.hash_to_block[h] = b

    def hit_rate(self):
        return self.num_hits / self.num_queries if self.num_queries else 0.0
