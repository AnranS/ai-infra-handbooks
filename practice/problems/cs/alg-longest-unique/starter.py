def longest_unique(s):
    last = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last:
            left = last[ch] + 1                # 没有 max：左边界可能往回跳，窗口里又混入重复字符
        last[ch] = right
        best = max(best, right - left + 1)
    return best
