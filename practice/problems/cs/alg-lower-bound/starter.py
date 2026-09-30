def lower_bound(a, x):
    lo, hi = 0, len(a) - 1                     # 闭区间的边界配上半开区间的循环：会漏掉最后一个
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:                         # 应该是 <=，否则和 lower_bound 一样
            lo = mid + 1
        else:
            hi = mid
    return lo


def count_of(a, x):
    return upper_bound(a, x) - lower_bound(a, x)
