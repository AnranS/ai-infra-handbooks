from collections import Counter, defaultdict


def is_anagram(s, t):
    return len(s) == len(t) and Counter(s) == Counter(t)


def group_anagrams(words):
    groups = defaultdict(list)                 # 规范形式 -> 原词列表
    for w in words:
        groups["".join(sorted(w))].append(w)
    return list(groups.values())               # dict 保持插入顺序
