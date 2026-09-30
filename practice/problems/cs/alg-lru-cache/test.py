from checker import check
from solution import LRUCache


def test_example():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    check(c.get(1), 1, "命中")
    c.put(3, 3)
    check(c.get(2), -1, "2 被淘汰")
    check(c.keys_lru_first(), [1, 3], "顺序：最久未使用在前")


def test_get_updates_order():
    c = LRUCache(3)
    for k in (1, 2, 3):
        c.put(k, k)
    c.get(1)                                   # 1 变成最近使用
    c.put(4, 4)                                # 淘汰 2
    check(c.get(2), -1, "2 被淘汰而不是 1")
    check(c.get(1), 1, "1 还在")
    check(c.keys_lru_first(), [3, 4, 1], "刚才 get(1) 又把 1 挪到了最近使用")


def test_update_existing():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.put(1, 100)                              # 更新已有的键
    check(c.get(1), 100, "值被更新")
    c.put(3, 3)
    check(c.get(2), -1, "被淘汰的是 2")
    check(c.keys_lru_first(), [1, 3], "1 因为刚被 put 过而更新")


def test_zero_capacity():
    c = LRUCache(0)
    c.put(1, 1)
    check(c.get(1), -1, "容量为 0 时存不下")
    check(c.keys_lru_first(), [], "空的")


def test_capacity_one():
    c = LRUCache(1)
    c.put(1, 1)
    c.put(2, 2)
    check(c.get(1), -1, "只能存一个")
    check(c.get(2), 2, "最新的还在")


def test_many_operations():
    c = LRUCache(100)
    for i in range(1000):
        c.put(i, i * 2)
        if i >= 100:
            check(c.get(i - 100), -1, "早期的键被淘汰")
    check(len(c.keys_lru_first()), 100, "始终不超过容量")
    check(c.get(999), 1998, "最后写入的还在")
