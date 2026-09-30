def split_array(a, k):
    def chunks(limit):
        used, cur = 1, 0                       # 至少一段
        for x in a:
            if cur + x > limit:
                used, cur = used + 1, 0
            cur += x
        return used

    lo, hi = max(a), sum(a)                    # 答案一定在这个区间里
    while lo < hi:
        mid = (lo + hi) // 2
        if chunks(mid) <= k:
            hi = mid                           # 可行，试试更小的
        else:
            lo = mid + 1
    return lo
