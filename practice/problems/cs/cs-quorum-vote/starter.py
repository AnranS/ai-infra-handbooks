def fresh_read(n, w, r):
    return w + r >= n                                  # 等于 n 时读写集合可能没有交集


def tolerated_failures(n):
    return n // 2                                      # 对偶数副本算多了


def can_elect(total, alive):
    return alive > total // 2 + 1                      # 正好是多数派时也该能选


def grant_vote(voter, candidate):
    if candidate["term"] <= voter["term"]:
        return False
    return candidate["last_log_index"] >= voter["last_log_index"]   # 只比长度，没比任期
