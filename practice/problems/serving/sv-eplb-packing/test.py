import random

from checker import check, check_close
from solution import balanced_packing, imbalance, replicate


def test_example():
    check(replicate([8, 1, 1, 2], 6), [3, 1, 1, 1], "最热的专家拿到两个额外副本")
    check(balanced_packing([5, 1, 4, 2], 2), [[0, 1], [2, 3]], "5+1 与 4+2")


def test_replicate():
    check(replicate([1, 1, 1], 3), [1, 1, 1], "不需要冗余")
    check(replicate([4, 4], 4), [2, 2], "并列时轮流加")
    cnt = replicate([10, 3, 3, 3, 1], 9)
    check(sum(cnt), 9, "副本总数")
    check(cnt[0], 4, "10 分成 4 份（每份 2.5）之后才轮到别人")


def test_packing():
    packs = balanced_packing([9, 8, 7, 6, 5, 4, 3, 2], 4)
    check([len(p) for p in packs], [2, 2, 2, 2], "每包物品数相同")
    check(sorted(i for p in packs for i in p), list(range(8)), "每个物品恰好一次")
    check(packs, [[0, 7], [1, 6], [2, 5], [3, 4]], "重的与轻的搭配")


def test_end_to_end():
    rng = random.Random(0)
    load = [rng.paretovariate(1.5) for _ in range(64)]
    naive_cnt = [1] * 64
    naive = imbalance(load, naive_cnt, [list(range(i * 8, i * 8 + 8)) for i in range(8)], list(range(64)))
    cnt = replicate(load, 72)
    phys = [e for e in range(64) for _ in range(cnt[e])]
    packs = balanced_packing([load[e] / cnt[e] for e in phys], 8)
    balanced = imbalance(load, cnt, packs, phys)
    assert balanced < naive, f"EPLB 之后应该更均衡：{balanced:.2f} vs {naive:.2f}"
    assert balanced < 1.1, f"8 个冗余副本之后最忙的卡应接近平均：{balanced:.2f}"
