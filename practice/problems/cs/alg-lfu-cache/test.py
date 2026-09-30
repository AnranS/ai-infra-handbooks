from checker import check
from solution import LFUCache


def test_example():
    c = LFUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    check(c.get(1), 1, "命中")
    c.put(3, 3)
    check(c.get(2), -1, "次数最少的被淘汰")
    check(c.get(3), 3, "新键还在")


def test_tie_breaks_by_time():
    c = LFUCache(2)
    c.put(1, 1)
    c.put(2, 2)                                # 两个键次数都是 1
    c.put(3, 3)                                # 淘汰更久未使用的 1
    check(c.get(1), -1, "同次数时淘汰最久未使用的")
    check(c.get(2), 2, "2 还在")


def test_update_counts_as_use():
    c = LFUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.put(1, 100)                              # 更新也算一次使用
    c.put(3, 3)
    check(c.get(2), -1, "2 的次数最少")
    check(c.get(1), 100, "1 的值被更新了")


def test_zero_capacity():
    c = LFUCache(0)
    c.put(1, 1)
    check(c.get(1), -1, "容量为 0")
    check(c.keys_by_freq(), [], "什么都没有")


def test_freq_order():
    c = LFUCache(3)
    for k in (1, 2, 3):
        c.put(k, k)
    c.get(3)
    c.get(3)
    c.get(2)
    check(c.keys_by_freq(), [(1, 1), (2, 2), (3, 3)], "按次数升序")


def test_min_freq_moves():
    c = LFUCache(2)
    c.put(1, 1)
    c.get(1)
    c.get(1)                                   # 键 1 的次数是 3
    c.put(2, 2)                                # 键 2 的次数是 1
    c.put(3, 3)                                # 淘汰次数最少的 2
    check(c.get(2), -1, "2 被淘汰")
    check(c.get(1), 1, "1 因为次数高留下来")


def test_many():
    c = LFUCache(100)
    for i in range(1000):
        c.put(i, i)
        c.get(i)
    check(len(c.keys_by_freq()), 100, "不超过容量")
    check(c.get(999), 999, "最近的还在")
    check(c.get(0), -1, "早期的被淘汰")
