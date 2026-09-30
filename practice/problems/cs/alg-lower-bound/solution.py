def lower_bound(a, x):
    lo, hi = 0, len(a)                         # 半开区间 [lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1                       # mid 比 x 小，答案在右边
        else:
            hi = mid                           # mid 可能就是答案
    return lo


def upper_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def count_of(a, x):
    return upper_bound(a, x) - lower_bound(a, x)
