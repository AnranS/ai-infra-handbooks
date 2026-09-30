from collections import Counter


def first_unique_char(s):
    cnt = Counter(s)
    for ch in s:
        if cnt[ch] == 1:
            return ch                          # 返回的是字符，题目要的是下标
    return -1


def top_k_frequent(nums, k):
    cnt = Counter(nums)
    return sorted(cnt, key=lambda x: cnt[x])[:k]   # 升序：拿到的是出现最少的 k 个
