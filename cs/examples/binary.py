# 二分的三种用法：找插入位置、在答案空间上二分、以及和标准库对拍
import bisect
import random


def lower_bound(a, x):
    """第一个 >= x 的位置（等价于 bisect.bisect_left）"""
    lo, hi = 0, len(a)                          # 半开区间 [lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(a, x):
    """第一个 > x 的位置（等价于 bisect.bisect_right）"""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


rng = random.Random(0)
for _ in range(2000):                           # 和标准库随机对拍
    a = sorted(rng.randrange(10) for _ in range(rng.randrange(8)))
    x = rng.randrange(10)
    assert lower_bound(a, x) == bisect.bisect_left(a, x), (a, x)
    assert upper_bound(a, x) == bisect.bisect_right(a, x), (a, x)
print("和 bisect 对拍 2000 组：完全一致")
print("出现次数 = upper_bound - lower_bound，例如 [1,2,2,2,3] 里 2 出现",
      upper_bound([1, 2, 2, 2, 3], 2) - lower_bound([1, 2, 2, 2, 3], 2), "次")
print()


def min_chunk_size(lengths, budget, max_chunks):
    """二分答案：把一串请求按顺序切成若干块，每块的 token 数不超过 size，
    问 size 最小取多少才能不超过 max_chunks 块（这就是分块 prefill 的分块策略）"""
    def chunks_needed(size):
        used = cur = 0
        for x in lengths:
            if x > size:
                return float("inf")             # 单个请求就超了，这个 size 不可行
            if cur + x > size:
                used, cur = used + 1, 0
            cur += x
        return used + (1 if cur else 0)

    lo, hi = max(lengths), sum(lengths)         # 答案一定在这个范围里
    while lo < hi:
        mid = (lo + hi) // 2
        if chunks_needed(mid) <= max_chunks:
            hi = mid                            # 可行，试试更小的
        else:
            lo = mid + 1
    return lo if lo <= budget else -1


reqs = [512, 1024, 256, 2048, 128, 768]
for k in (2, 3, 4, 6):
    print(f"把 {len(reqs)} 个请求切成不超过 {k} 块时，每块至少要能装 {min_chunk_size(reqs, 99999, k)} 个 token")
