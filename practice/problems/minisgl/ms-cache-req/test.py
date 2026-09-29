from checker import check
from solution import Handle, ReqState, ToyPrefixCache, cache_req


def run(cache, req, finished):
    freed = []
    cache_req(cache, req, finished, lambda xs: freed.append(list(xs)))
    return freed


def test_example_four_regions():
    cache = ToyPrefixCache(page_size=2)
    empty = Handle(0)
    cache.lock(empty)
    # 别的请求先插入了 [1,2,3,4]（位置 100..103）
    cache.insert_prefix([1, 2, 3, 4], [100, 101, 102, 103])
    # 本请求：当初只命中了 0 个，自己算了 7 个 token，位置 10..16
    req = ReqState([1, 2, 3, 4, 5, 6, 7], 7, list(range(10, 17)), empty)
    freed = run(cache, req, finished=True)
    check(freed, [[10, 11, 12, 13], [16]], "释放：重复的 [0,4) 和不足一页的尾巴 [6,7)")
    check(cache.locks[empty], 0, "旧句柄被解锁")


def test_not_finished_keeps_tail_and_locks_new():
    cache = ToyPrefixCache(page_size=2)
    h0 = Handle(0)
    cache.lock(h0)
    req = ReqState([5, 6, 7, 8, 9], 5, [20, 21, 22, 23, 24], h0)
    freed = run(cache, req, finished=False)
    check(freed, [], "没有重复、尾巴留给请求")
    check(req.handle.cached_len, 4, "新句柄长度（按页对齐）")
    check(cache.locks[req.handle], 1, "新句柄被锁住")
    check(cache.locks[h0], 0, "旧句柄被解锁")


def test_prefix_hit_then_finish():
    cache = ToyPrefixCache(page_size=1)
    cache.insert_prefix([1, 2, 3], [7, 8, 9])
    hit = Handle(3, (1, 2, 3))
    cache.lock(hit)
    # 命中了 3 个（位置 7,8,9），自己又算了 2 个
    req = ReqState([1, 2, 3, 4, 5], 5, [7, 8, 9, 40, 41], hit)
    freed = run(cache, req, finished=True)
    check(freed, [], "命中部分不属于本请求、新算的部分交给缓存：什么都不释放")
    check(cache.locks[hit], 0, "旧句柄解锁")
    check(cache.store[(1, 2, 3, 4, 5)], (7, 8, 9, 40, 41), "缓存里记下了完整前缀")


def test_unlock_after_insert():
    """如果先解锁再插入，缓存可能在中间把旧前缀淘汰掉：检查调用顺序"""
    calls = []

    class Spy(ToyPrefixCache):
        def insert_prefix(self, ids, indices):
            calls.append("insert")
            return super().insert_prefix(ids, indices)

        def unlock(self, handle):
            calls.append("unlock")
            super().unlock(handle)

        def lock(self, handle):
            calls.append("lock")
            super().lock(handle)

    cache = Spy(1)
    h = Handle(0)
    cache.lock(h)
    calls.clear()
    run(cache, ReqState([1, 2], 2, [5, 6], h), finished=False)
    check(calls, ["insert", "unlock", "lock"], "调用顺序")
