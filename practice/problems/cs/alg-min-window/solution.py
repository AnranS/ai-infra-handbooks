from collections import Counter


def min_window(s, t):
    if not t or len(s) < len(t):
        return ""
    need = Counter(t)
    missing = len(t)                           # 还欠多少个字符（含重复）
    best = (len(s) + 1, 0, 0)
    left = 0
    for right, ch in enumerate(s):
        if need[ch] > 0:                       # 只有还欠着的才算数
            missing -= 1
        need[ch] -= 1
        while missing == 0:                    # 已覆盖：尽量收缩
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right + 1)
            need[s[left]] += 1
            if need[s[left]] > 0:              # 左边这个字符是必需的，松开就不再覆盖
                missing += 1
            left += 1
    return "" if best[0] > len(s) else s[best[1]:best[2]]
