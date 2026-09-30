def max_subarray(a):
    best, best_range = 0, (0, 0)               # 初始化成 0：全是负数时会答错
    cur, start = 0, 0
    for i in range(len(a)):
        if cur + a[i] < a[i]:
            cur, start = a[i], i
        else:
            cur += a[i]
        if cur >= best:                        # 用 >=：会取到更长、更靠右的区间
            best, best_range = cur, (start, i + 1)
    return best, best_range[0], best_range[1]
