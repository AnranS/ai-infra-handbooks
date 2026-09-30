from collections import Counter, defaultdict


def is_anagram(s, t):
    return Counter(s) == Counter(t)            # 长度不同时 Counter 也可能相等？其实不会，但下面的分组有问题


def group_anagrams(words):
    groups = defaultdict(list)
    for w in words:
        groups[len(w)].append(w)               # 用长度当键：长度相同但不是异位词的会被分到一组
    return list(groups.values())
