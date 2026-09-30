from checker import check
from solution import can_elect, fresh_read, grant_vote, tolerated_failures


def node(term, idx, log_term, voted_for=None, nid=0):
    return {"term": term, "voted_for": voted_for, "last_log_index": idx, "last_log_term": log_term, "id": nid}


def test_example():
    check(fresh_read(5, 3, 3), True, "多数派读写")
    check(tolerated_failures(5), 2, "5 个副本容忍 2 台")
    check(can_elect(5, 2), False, "2 台不是多数派")
    check(grant_vote(node(1, 3, 1), {"term": 2, "last_log_index": 3, "last_log_term": 1, "id": 1}), True,
          "任期更大、日志一样新")


def test_quorum_table():
    check([fresh_read(3, w, r) for w, r in [(2, 2), (3, 1), (1, 3), (1, 1)]], [True, True, True, False],
          "W + R > N")
    check([tolerated_failures(n) for n in (1, 2, 3, 4, 5, 6, 7)], [0, 0, 1, 1, 2, 2, 3],
          "偶数副本不划算")


def test_elect():
    check([can_elect(5, a) for a in (1, 2, 3, 5)], [False, False, True, True], "3 台就够")
    check(can_elect(3, 2), True, "3 个副本挂一台还能选")
    check(can_elect(4, 2), False, "4 个副本挂两台就选不出来了")


def test_vote_term_rules():
    v = node(3, 5, 2, voted_for=None, nid=0)
    check(grant_vote(v, {"term": 2, "last_log_index": 9, "last_log_term": 9, "id": 1}), False, "任期比我小")
    check(grant_vote(node(3, 5, 2, voted_for=7), {"term": 3, "last_log_index": 5, "last_log_term": 2, "id": 1}),
          False, "同任期里已经投给别人了")
    check(grant_vote(node(3, 5, 2, voted_for=1), {"term": 3, "last_log_index": 5, "last_log_term": 2, "id": 1}),
          True, "同任期里重复请求同一个候选人：可以再投一次")


def test_vote_log_rules():
    v = node(3, 5, 4)
    check(grant_vote(v, {"term": 4, "last_log_index": 9, "last_log_term": 3, "id": 1}), False,
          "日志更长但最后一条任期更旧：不投")
    check(grant_vote(v, {"term": 4, "last_log_index": 2, "last_log_term": 5, "id": 1}), True,
          "最后一条任期更新：即使更短也投")
    check(grant_vote(v, {"term": 4, "last_log_index": 4, "last_log_term": 4, "id": 1}), False,
          "同任期但更短：不投")
    check(grant_vote(v, {"term": 4, "last_log_index": 5, "last_log_term": 4, "id": 1}), True,
          "一样新：投")


def test_split_vote_impossible():
    # 两个候选人不可能在同一任期里都拿到多数票：5 个节点、每人一票
    voters = [node(1, 1, 1, nid=i) for i in range(5)]
    a = {"term": 2, "last_log_index": 1, "last_log_term": 1, "id": 100}
    b = {"term": 2, "last_log_index": 1, "last_log_term": 1, "id": 200}
    votes_a = 0
    for v in voters:
        if grant_vote(v, a):
            v["voted_for"], v["term"] = a["id"], a["term"]
            votes_a += 1
    votes_b = sum(grant_vote(v, b) for v in voters)
    check(votes_a, 5, "A 先拿到全部选票")
    check(votes_b, 0, "同一任期里 B 一票也拿不到")
