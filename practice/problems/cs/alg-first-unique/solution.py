from collections import Counter


def first_unique_char(s):
    cnt = Counter(s)
    for i, ch in enumerate(s):
        if cnt[ch] == 1:
            return i
    return -1


def top_k_frequent(nums, k):
    cnt = Counter(nums)                        # 保持第一次出现的顺序
    return sorted(cnt, key=lambda x: -cnt[x])[:k]   # 稳定排序：次数相同时保持原顺序
