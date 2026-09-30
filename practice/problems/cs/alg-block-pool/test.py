from checker import check, check_close
from solution import BlockPool


def test_example():
    p = BlockPool(2)
    a = p.allocate("h1")
    b = p.allocate("h1")
    check((a, b), (0, 0), "同一个哈希复用同一个块")
    p.free(a)
    p.free(b)
    check(p.num_free(), 2, "引用计数归零后回到空闲队列，加上从没用过的块 1")
    check(p.allocate("h1"), 0, "仍然命中缓存")
    check_close(p.hit_rate(), 2 / 3, rtol=1e-9, what="三次分配命中两次")


def test_none_hash_not_shared():
    p = BlockPool(3)
    a = p.allocate(None)
    b = p.allocate(None)
    check(a != b, True, "不可复用的块不能互相串用")
    check(p.hit_rate(), 0.0, "都没命中")


def test_exhaustion():
    p = BlockPool(2)
    p.allocate("a")
    p.allocate("b")
    check(p.allocate("c"), -1, "没有空闲块")
    check(p.num_free(), 0, "空闲数为 0")


def test_refcount_protects():
    p = BlockPool(2)
    a = p.allocate("a")
    p.allocate("a")                            # 引用计数 2
    p.free(a)
    check(p.num_free(), 1, "还有一个引用，不能回到空闲队列")
    p.free(a)
    check(p.num_free(), 2, "计数归零才回去")


def test_eviction_invalidates_hash():
    p = BlockPool(1)
    a = p.allocate("old")
    p.free(a)
    b = p.allocate("new")                      # 复用同一个块，旧哈希必须失效
    check(b, 0, "复用了同一个块")
    p.free(b)
    c = p.allocate("old")
    check(c, 0, "拿到的是块 0")
    check_close(p.hit_rate(), 0.0, rtol=1e-9, what="旧哈希已经失效，不该算命中")


def test_lru_order():
    p = BlockPool(3)
    ids = [p.allocate(f"h{i}") for i in range(3)]
    for i in ids:
        p.free(i)                              # 依次释放 0、1、2
    check(p.allocate(None), 0, "先释放的先被复用")
    check(p.allocate(None), 1, "然后是 1")


def test_many():
    p = BlockPool(64)
    for i in range(1000):
        bid = p.allocate(f"k{i % 64}")
        check(bid >= 0, True, "总能分配到")
        p.free(bid)
    check(p.hit_rate() > 0.9, True, f"重复访问同一批键，命中率很高（{p.hit_rate():.2f}）")
