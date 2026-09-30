def fresh_read(n, w, r):
    return w + r > n


def tolerated_failures(n):
    return n - (n // 2 + 1)


def can_elect(total, alive):
    return alive >= total // 2 + 1


def grant_vote(voter, candidate):
    if candidate["term"] < voter["term"]:
        return False
    if candidate["term"] == voter["term"] and voter["voted_for"] not in (None, candidate["id"]):
        return False
    mine = (voter["last_log_term"], voter["last_log_index"])
    theirs = (candidate["last_log_term"], candidate["last_log_index"])
    return theirs >= mine                              # 先比任期，任期相同再比长度
