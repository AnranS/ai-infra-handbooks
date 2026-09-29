import random

from checker import check, time_limit
from solution import prefix_hits


def brute(reqs):
    out = []
    for i, r in enumerate(reqs):
        best = 0
        for p in reqs[:i]:
            n = 0
            while n < min(len(r), len(p)) and r[n] == p[n]:
                n += 1
            best = max(best, n)
        out.append(best)
    return out


def test_example():
    check(prefix_hits([[1, 2, 3], [1, 2, 4], [1, 2, 3, 5], [7]]), [0, 2, 3, 0])


def test_edge_cases():
    check(prefix_hits([]), [], "空列表")
    check(prefix_hits([[], [1], []]), [0, 0, 0], "空请求")
    check(prefix_hits([[5, 5], [5, 5], [5]]), [0, 2, 1], "完全相同、被包含")


def test_random_small():
    rng = random.Random(0)
    for _ in range(30):
        reqs = [[rng.randint(0, 3) for _ in range(rng.randint(0, 8))] for _ in range(rng.randint(1, 30))]
        check(prefix_hits(reqs), brute(reqs), f"随机用例 {reqs[:3]}…")


def ref_fast(reqs):
    """另一种线性做法：把每个请求的所有前缀哈希存进集合，二分查找最长命中"""
    seen, out = set(), []
    for r in reqs:
        hashes, h = [], 0
        for t in r:
            h = hash((h, t))
            hashes.append(h)
        lo, hi = 0, len(r)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if hashes[mid - 1] in seen:
                lo = mid
            else:
                hi = mid - 1
        out.append(lo)
        seen.update(hashes)
    return out


def test_large():
    """2000 个请求共享几段长的系统提示词，每个约 300 token"""
    rng = random.Random(1)
    systems = [[rng.randint(0, 50000) for _ in range(rng.randint(200, 250))] for _ in range(5)]
    reqs = []
    for i in range(2000):
        base = list(rng.choice(systems))
        if reqs and rng.random() < 0.3:          # 多轮对话：在之前某个请求后面接着说
            base = list(rng.choice(reqs))[:400]
        reqs.append(base + [rng.randint(0, 50000) for _ in range(rng.randint(10, 100))])
    with time_limit(1.0, "2000 个请求"):
        got = prefix_hits(reqs)
    check(got, ref_fast(reqs), "2000 个请求的命中长度")
