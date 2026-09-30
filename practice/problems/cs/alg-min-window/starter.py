from collections import Counter


def min_window(s, t):
    if not t:
        return ""
    need = Counter(t)
    missing = len(t)
    best = (len(s) + 1, 0, 0)
    left = 0
    for right, ch in enumerate(s):
        need[ch] -= 1
        if need[ch] >= 0:                      # 多余的重复字符也被算成了"补上一个"
            missing -= 1
        while missing == 0:
            if right - left + 1 <= best[0]:    # 用 <=：会取到更靠右的答案
                best = (right - left + 1, left, right + 1)
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
    return "" if best[0] > len(s) else s[best[1]:best[2]]
