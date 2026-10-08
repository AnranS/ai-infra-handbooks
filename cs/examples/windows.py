# 滑动窗口的三种形态：定长、最长满足条件、最短满足条件。三者共用同一套框架
from collections import Counter


def fixed_window_max_sum(a, k):
    """定长窗口：长度为 k 的子数组里最大的和"""
    cur = sum(a[:k])
    best = cur
    for right in range(k, len(a)):
        cur += a[right] - a[right - k]          # 进一个、出一个
        best = max(best, cur)
    return best


def longest_at_most_k_distinct(s, k):
    """最长窗口：最多含 k 种字符的最长子串长度"""
    cnt = Counter()
    left = best = 0
    for right, ch in enumerate(s):
        cnt[ch] += 1
        while len(cnt) > k:                     # 不满足条件就收缩左边界
            cnt[s[left]] -= 1
            if cnt[s[left]] == 0:
                del cnt[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best


def shortest_sum_at_least(a, target):
    """最短窗口：和 >= target 的最短子数组长度（全为正数）"""
    left = cur = 0
    best = len(a) + 1
    for right, x in enumerate(a):
        cur += x
        while cur >= target:                    # 满足了就尽量收缩
            best = min(best, right - left + 1)
            cur -= a[left]
            left += 1
    return best if best <= len(a) else 0


a = [2, 1, 5, 1, 3, 2]
print("定长窗口 k=3 的最大和：", fixed_window_max_sum(a, 3))
print("最长窗口（最多 2 种字符）：", longest_at_most_k_distinct("eceba", 2))
print("最短窗口（和 >= 7）：", shortest_sum_at_least(a, 7))
print()
print("三者的共同框架：右指针每步进一个元素，左指针只会往右走，所以总共 O(n)。")
print("区别只在收缩的时机：定长是右进左出，最长是不满足时收缩，最短是满足时收缩。")
