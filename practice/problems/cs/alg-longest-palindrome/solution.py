def _expand(s, left, right):
    """从中心往两边扩，返回 (起点, 长度)"""
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return left + 1, right - left - 1


def longest_palindrome(s):
    best_start, best_len = 0, 0
    for i in range(len(s)):
        for left, right in ((i, i), (i, i + 1)):       # 奇数中心和偶数中心
            start, length = _expand(s, left, right)
            if length > best_len:                      # 严格大于：保证最靠左
                best_start, best_len = start, length
    return s[best_start:best_start + best_len]


def count_palindromes(s):
    total = 0
    for i in range(len(s)):
        for left, right in ((i, i), (i, i + 1)):
            l, r = left, right
            while l >= 0 and r < len(s) and s[l] == s[r]:
                total += 1                             # 每扩一次就是一个回文子串
                l -= 1
                r += 1
    return total
