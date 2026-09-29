import random

from checker import check, raises
from solution import RadixCache


def dump(cache):
    out = []

    def rec(node, depth):
        for k in sorted(node.children):
            c = node.children[k]
            out.append((depth, c.key, c.value, c.ref_count))
            rec(c, depth + 1)

    rec(cache.root, 0)
    return out


def test_example_from_book():
    c = RadixCache()
    c.insert_prefix([1, 2, 3, 4], [10, 11, 12, 13])
    c.insert_prefix([1, 2, 5, 6], [99, 99, 20, 21])
    c.insert_prefix([7, 8], [30, 31])
    check(dump(c), [(0, [1, 2], [10, 11], 0), (1, [3, 4], [12, 13], 0), (1, [5, 6], [20, 21], 0), (0, [7, 8], [30, 31], 0)],
          "插入三个前缀之后的树")
    h, n = c.match_prefix([1, 2, 3, 9])
    check((n, c.matched_indices(h)), (3, [10, 11, 12]), "匹配 [1,2,3,9]")
    c.lock(h)
    check((c.evictable_size, c.protected_size), (5, 3), "加锁后 (可淘汰, 受保护)")
    check(c.evict(2), [13, 20, 21], "淘汰 2 个 token")
    check(c.evict(1), [30, 31], "再淘汰 1 个 token")
    check(dump(c), [(0, [1, 2], [10, 11], 1), (1, [3], [12], 1)], "剩下的树")
    with raises(ValueError, "没有可淘汰的了"):
        c.evict(1)


def test_insert_returns_prefix_len():
    c = RadixCache()
    check(c.insert_prefix([1, 2, 3], [5, 6, 7])[0], 0, "第一次插入")
    n, h = c.insert_prefix([1, 2, 3, 4, 5], [0, 0, 0, 8, 9])
    check((n, h.cached_len, c.matched_indices(h)), (3, 5, [5, 6, 7, 8, 9]), "已有 3 个，新增 2 个")
    n, h = c.insert_prefix([1, 2], [0, 0])
    check((n, c.matched_indices(h)), (2, [5, 6]), "完全被包含：分裂出 [1,2]")
    check(c.evictable_size, 5, "可淘汰总数")


def test_lock_unlock_counts():
    c = RadixCache()
    c.insert_prefix([1, 2, 3, 4], [1, 2, 3, 4])
    h1, _ = c.match_prefix([1, 2])
    h2, _ = c.match_prefix([1, 2, 3, 4])
    c.lock(h1)
    c.lock(h2)
    check((c.evictable_size, c.protected_size), (0, 4), "两个句柄都锁住")
    c.unlock(h2)
    check((c.evictable_size, c.protected_size), (2, 2), "解锁 h2：[3,4] 变成可淘汰，[1,2] 仍被 h1 锁着")
    check(c.evict(2), [3, 4], "只能淘汰 [3,4]")


def test_lru_order():
    c = RadixCache()
    c.insert_prefix([1], [1])
    c.insert_prefix([2], [2])
    c.insert_prefix([3], [3])
    c.match_prefix([1])                     # 访问 1，刷新时间戳
    check(c.evict(2), [2, 3], "最久没访问的先淘汰")


class Ref:
    """朴素实现：只记录每个被缓存的 token 前缀（按 token 粒度），用于检查匹配长度"""

    def __init__(self):
        self.prefixes = set()

    def insert(self, ids):
        for k in range(1, len(ids) + 1):
            self.prefixes.add(tuple(ids[:k]))

    def match(self, ids):
        n = 0
        while n < len(ids) and tuple(ids[:n + 1]) in self.prefixes:
            n += 1
        return n


def test_random_match_lengths_and_integrity():
    rng = random.Random(0)
    c, ref = RadixCache(), Ref()
    nxt = 1000
    for step in range(400):
        ids = [rng.randint(0, 3) for _ in range(rng.randint(1, 8))]
        if rng.random() < 0.5:
            vals = list(range(nxt, nxt + len(ids)))
            nxt += len(ids)
            n, h = c.insert_prefix(ids, vals)
            check(n, ref.match(ids), f"第 {step} 步插入前的已有长度")
            ref.insert(ids)
        else:
            h, n = c.match_prefix(ids)
            check(n, ref.match(ids), f"第 {step} 步的匹配长度")
            check(len(c.matched_indices(h)), n, f"第 {step} 步匹配到的位置数")
        total = sum(len(k) for _, k, _, _ in dump(c))
        check(c.evictable_size + c.protected_size, total, f"第 {step} 步：计数与树的大小一致")
