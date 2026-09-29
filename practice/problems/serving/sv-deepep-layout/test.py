import numpy as np

from checker import check, raises
from solution import dispatch_layout, ll_buffer_bytes, ll_slots, recv_plan


def test_example():
    topk = np.array([[0, 3], [1, 2], [3, -1]])
    per_rank, per_expert, in_rank = dispatch_layout(topk, 4, 2)
    check(np.asarray(per_rank), np.array([2, 3]), "每张卡收到的 token 数（token 1 的专家 1、2 分在两张卡上，各一份）")
    check(np.asarray(per_expert), np.array([1, 1, 1, 2]), "每个专家被选中的次数")
    check(np.asarray(in_rank), np.array([[True, True], [True, True], [False, True]]), "in_rank")


def brute(topk, E, R):
    local = E // R
    in_rank = [[any(e >= 0 and e // local == r for e in row) for r in range(R)] for row in topk]
    per_expert = [sum(int(e == x) for row in topk for e in row) for x in range(E)]
    return [sum(col) for col in zip(*in_rank)], per_expert, in_rank


def rand_topk(rng, T, E, k, mask=0.1):
    topk = np.array([rng.choice(E, k, replace=False) for _ in range(T)])
    topk[rng.random((T, k)) < mask] = -1
    return topk


def test_layout_random():
    rng = np.random.default_rng(0)
    for E, R, k in ((8, 2, 2), (16, 4, 4), (64, 8, 8), (12, 12, 3)):
        topk = rand_topk(rng, 50, E, k)
        per_rank, per_expert, in_rank = dispatch_layout(topk, E, R)
        want = brute(topk.tolist(), E, R)
        check(np.asarray(per_rank).tolist(), want[0], f"E={E} R={R}：tokens_per_rank")
        check(np.asarray(per_expert).tolist(), want[1], f"E={E} R={R}：tokens_per_expert")
        check(np.asarray(in_rank).tolist(), want[2], f"E={E} R={R}：in_rank")
    per_rank, per_expert, _ = dispatch_layout(np.full((3, 2), -1), 4, 2)
    check(np.asarray(per_rank).tolist(), [0, 0], "全部被掩掉时不发送")


def test_recv_plan():
    rng = np.random.default_rng(1)
    E, R = 16, 4
    all_topk = [rand_topk(rng, int(rng.integers(0, 20)) + 1, E, 3) for _ in range(R)]
    counts, offsets, totals = recv_plan(all_topk, E)
    counts, offsets, totals = np.asarray(counts), np.asarray(offsets), np.asarray(totals)
    for s in range(R):
        check(counts[s].tolist(), brute(all_topk[s].tolist(), E, R)[0], f"第 {s} 张卡发往各卡的 token 数")
    for d in range(R):
        check(offsets[:, d].tolist(), [int(counts[:s, d].sum()) for s in range(R)], f"第 {d} 张卡的接收缓冲区按来源依次排")
        check(int(totals[d]), int(counts[:, d].sum()), f"第 {d} 张卡一共收多少行")
    c2, o2, t2 = recv_plan([np.array([[0, 1]]), np.array([[1, 3]])], 4)
    check((np.asarray(c2).tolist(), np.asarray(o2).tolist(), np.asarray(t2).tolist()),
          ([[1, 0], [1, 1]], [[0, 0], [1, 0]], [2, 1]), "两张卡的小例子")


def test_ll_slots():
    topk = np.array([[0, 2], [2, 3], [-1, 0], [2, 0]])
    check(np.asarray(ll_slots(topk, 0, 4, 2, 8)).tolist(), [[0, 0], [1, 0], [-1, 1], [2, 2]], "按 token 顺序，每个专家的第几份写第几个槽位")
    rng = np.random.default_rng(2)
    topk = rand_topk(rng, 200, 32, 4)
    slots = np.asarray(ll_slots(topk, 5, 32, 8, 200))
    for e in range(32):
        got = slots[topk == e]
        check(sorted(got.tolist()), list(range(int((topk == e).sum()))), f"专家 {e} 的槽位是 0..n-1、没有重复")
    check(bool(np.all(slots[topk < 0] == -1)), True, "被掩掉的位置槽位为 -1")
    with raises(OverflowError, "某个专家收到的份数超过 max_tokens"):
        ll_slots(np.array([[1], [1], [1]]), 0, 4, 2, 2)


def test_ll_buffer_bytes():
    msg = 7168 + 7168 // 128 * 4 + 16
    check(ll_buffer_bytes(256, 64, 128, msg), 4 * 64 * 128 * msg, "本章的例子：4 个本地专家 × 64 个来源 × 128 个槽位")
    check(round(ll_buffer_bytes(256, 64, 128, msg) / 2**20), 232, "约 232 MiB")
    check(ll_buffer_bytes(256, 32, 256, 1000), 8 * 32 * 256 * 1000, "32 张卡、每卡 256 个 token")
