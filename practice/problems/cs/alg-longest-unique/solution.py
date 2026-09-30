def longest_unique(s):
    last = {}                                  # 字符 -> 最后一次出现的下标
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last:
            left = max(left, last[ch] + 1)     # 左边界只能往右跳
        last[ch] = right
        best = max(best, right - left + 1)
    return best
