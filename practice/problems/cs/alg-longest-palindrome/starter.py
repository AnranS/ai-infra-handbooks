def longest_palindrome(s):
    best = ""
    for i in range(len(s)):
        for j in range(i, len(s)):             # 枚举所有子串再判断：O(n³)
            sub = s[i:j + 1]
            if sub == sub[::-1] and len(sub) >= len(best):   # >= 会取到更靠右的
                best = sub
    return best


def count_palindromes(s):
    total = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            sub = s[i:j + 1]
            if sub == sub[::-1]:
                total += 1
    return total
