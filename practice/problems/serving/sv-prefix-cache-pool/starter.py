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
        pass

    def free(self, blocks):
        pass

    def touch(self, blocks):
        pass

    def block_hashes(self, token_ids, extra_key=None):
        pass

    def lookup(self, token_ids, extra_key=None):
        pass

    def cache_blocks(self, token_ids, block_ids, num_computed, extra_key=None):
        pass

    def hit_rate(self):
        pass
