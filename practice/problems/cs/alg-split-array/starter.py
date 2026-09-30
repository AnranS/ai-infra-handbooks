def split_array(a, k):
    def chunks(limit):
        used, cur = 1, 0
        for x in a:
            if cur + x >= limit:               # 用 >=：正好等于上限也被切开了
                used, cur = used + 1, 0
            cur += x
        return used

    lo, hi = 0, sum(a)                         # 下界取 0：单个元素比上限大时死循环
    while lo < hi:
        mid = (lo + hi) // 2
        if chunks(mid) <= k:
            hi = mid
        else:
            lo = mid + 1
    return lo
