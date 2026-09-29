import random

from checker import check, time_limit
from solution import LRUCache


def test_example():
    """题目里的例子"""
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    check(cache.get(1), "a", "get(1)")
    cache.put(3, "c")
    check(cache.get(2), -1, "淘汰 2 之后 get(2)")
    check(len(cache), 2, "len(cache)")


def test_update_counts_as_use():
    """put 已有的键会更新值，也算一次使用"""
    cache = LRUCache(2)
    cache.put("x", 1)
    cache.put("y", 2)
    cache.put("x", 10)
    cache.put("z", 3)          # 淘汰 y
    check(cache.get("x"), 10, "get('x')")
    check(cache.get("y"), -1, "get('y')")
    check(cache.get("z"), 3, "get('z')")


def test_contains_does_not_touch():
    """in 只是查询，不改变使用顺序"""
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert 1 in cache
    cache.put(3, 3)            # 1 仍然是最久没用的
    assert 1 not in cache and 2 in cache and 3 in cache


def test_capacity_one():
    cache = LRUCache(1)
    cache.put(1, 1)
    cache.put(2, 2)
    check(cache.get(1), -1)
    check(cache.get(2), 2)
    check(len(cache), 1)


def test_random_against_reference():
    """和一个朴素实现对拍 2000 次随机操作"""
    rng = random.Random(0)
    cache, ref, order = LRUCache(5), {}, []
    for step in range(2000):
        k = rng.randrange(10)
        if rng.random() < 0.5:
            got = cache.get(k)
            want = ref.get(k, -1)
            if k in ref:
                order.remove(k)
                order.append(k)
            assert got == want, f"第 {step} 步 get({k})：期望 {want}，实际 {got}"
        else:
            cache.put(k, step)
            if k in ref:
                order.remove(k)
            ref[k] = step
            order.append(k)
            if len(order) > 5:
                del ref[order.pop(0)]
        assert len(cache) == len(ref), f"第 {step} 步之后 len 不对"


def test_performance():
    """20 万次操作要在 1 秒内完成（O(1)）"""
    cache = LRUCache(1000)
    with time_limit(1.0, "20 万次 get/put"):
        for i in range(100_000):
            cache.put(i, i)
            cache.get(i - 500)
