def max_subarray(a):
    best, best_range = a[0], (0, 1)
    cur, start = a[0], 0
    for i in range(1, len(a)):
        if cur + a[i] < a[i]:                  # 前面的和是累赘，另起一段
            cur, start = a[i], i
        else:
            cur += a[i]
        if cur > best:                         # 严格大于：保证起点最小、长度最短
            best, best_range = cur, (start, i + 1)
    return best, best_range[0], best_range[1]
