from checker import check, check_close
from solution import PrefixCachingBlockPool


def test_example_from_chapter():
    pool = PrefixCachingBlockPool(num_blocks=4, block_size=2)
    a = [1, 2, 3, 4, 5]
    blocks = pool.allocate(3)
    pool.cache_blocks(a, blocks, num_computed=5)
    check([pool.block_hash[b] is not None for b in blocks], [True, True, False], "只有装满的块进入缓存")
    pool.free(blocks)
    check(list(pool.free_queue), [3, 2, 1, 0], "倒序释放")
    hit = pool.lookup([1, 2, 3, 4, 9])
    check(hit, [0, 1], "命中前两个块")
    pool.touch(hit)
    check(list(pool.free_queue), [3, 2], "命中的块移出空闲队列")
    pool.free(hit)
    check(pool.allocate(3), [3, 2, 1], "复用块 1，它的缓存失效")
    check(pool.lookup([1, 2, 3, 4, 9]), [0], "现在只能命中第一个块")


def test_extra_key_isolation():
    pool = PrefixCachingBlockPool(8, 2)
    toks = [5, 6, 7, 8, 9]
    b1 = pool.allocate(3)
    pool.cache_blocks(toks, b1, 5, extra_key="lora-A")
    check(pool.lookup(toks, extra_key="lora-B"), [], "不同 LoRA 不能命中")
    check(pool.lookup(toks), [], "没有 LoRA 的请求也不能命中")
    check(pool.lookup(toks, extra_key="lora-A"), b1[:2], "同一个 LoRA 可以命中")
    b2 = pool.allocate(2)
    pool.cache_blocks(toks, b2, 4, extra_key="lora-B")
    check(pool.lookup(toks, extra_key="lora-B"), b2, "LoRA-B 自己的块")
    check(pool.lookup(toks, extra_key="lora-A"), b1[:2], "LoRA-A 的块不受影响")


def test_chain_hash_depends_on_prefix():
    pool = PrefixCachingBlockPool(8, 2)
    b = pool.allocate(2)
    pool.cache_blocks([1, 2, 3, 4], b, 4)
    check(pool.lookup([9, 9, 3, 4]), [], "第二块内容相同但前缀不同，不能命中")
    check(pool.block_hashes([1, 2, 3]), pool.block_hashes([1, 2, 3, 4])[:1], "没装满的块不计算哈希")


def test_duplicate_registration_and_stats():
    pool = PrefixCachingBlockPool(8, 2)
    a, b = pool.allocate(2), pool.allocate(2)
    pool.cache_blocks([1, 2, 3, 4], a, 4)
    pool.cache_blocks([1, 2, 3, 4], b, 4)          # 同样的内容又算了一遍：不重复登记
    check([pool.block_hash[x] is None for x in b], [True, True], "重复的块不登记")
    pool.lookup([1, 2, 3, 4, 5, 6])
    pool.lookup([7, 7])
    check((pool.num_queries, pool.num_hits), (8, 4), "统计")
    check_close(pool.hit_rate(), 0.5, what="命中率")


def test_lru_order():
    pool = PrefixCachingBlockPool(4, 1)
    seqs = [[10], [20], [30], [40]]
    owned = []
    for s in seqs:
        blk = pool.allocate(1)
        pool.cache_blocks(s, blk, 1)
        owned.append(blk)
    for blk in owned:
        pool.free(blk)                            # 释放顺序：10、20、30、40
    hit = pool.lookup([10])                       # 访问 10，让它变"新"
    pool.touch(hit)
    pool.free(hit)
    pool.allocate(2)                              # 驱逐最久没用的两个：20、30
    check([bool(pool.lookup(s)) for s in seqs], [True, False, False, True], "LRU 驱逐之后哪些还能命中")
