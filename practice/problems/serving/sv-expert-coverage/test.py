from fractions import Fraction

from checker import check
from solution import lost_experts, min_failures_to_lose, prob_no_loss, replica_counts

CARDS = [[0, 1], [2, 3], [0, 2], [1, 3]]


def test_example():
    check(lost_experts(CARDS, {0, 2}), [0], "专家 0 的两份都在坏卡上")
    check(min_failures_to_lose(CARDS), 2, "每个专家两份")
    check(prob_no_loss(CARDS, 2), Fraction(1, 3), "坏两张卡")


def test_lost():
    check(lost_experts(CARDS, set()), [], "没有坏卡")
    check(lost_experts(CARDS, {0}), [], "坏一张卡：每个专家都还有一份")
    check(lost_experts(CARDS, {0, 1, 2}), [0, 2], "只剩卡 3")
    check(lost_experts(CARDS, {0, 1, 2, 3}), [0, 1, 2, 3], "全坏")
    single = [[0, 1, 2], [3, 4, 5], [6, 7, 0]]
    check(lost_experts(single, {1}), [3, 4, 5], "只有一份的专家")
    check(lost_experts([[5], [], [5, 9]], {2}), [9], "有的卡上没放专家")


def test_min_failures():
    check(min_failures_to_lose([[0, 1, 2], [3, 4, 5], [6, 7, 0]]), 1, "有只放了一份的专家")
    check(min_failures_to_lose([[0, 1], [0, 1], [0, 1]]), 3, "每个专家三份")
    check(min_failures_to_lose([[0, 0], [1]]), 1, "同一张卡上放两份不算两个副本")


def test_probability():
    check(prob_no_loss(CARDS, 1), Fraction(1), "坏一张卡一定不丢")
    check(prob_no_loss(CARDS, 3), Fraction(0), "坏三张卡一定丢")
    one_each = [[e] for e in range(8)] + [[0], [1]]          # 10 张卡：8 个专家各一份，0、1 各多一份
    check(prob_no_loss(one_each, 1), Fraction(4, 10), "坏一张卡：只有放 0、1 的 4 张卡坏了不丢")
    check(prob_no_loss(one_each, 2), Fraction(4, 45), "坏两张卡：只能在放 0、1 的 4 张卡里挑，且不能是同一个专家的两份")
    big = [[i, (i + 1) % 20] for i in range(20)]              # 20 张卡，每个专家两份放在相邻的卡上
    check(prob_no_loss(big, 2), Fraction(170, 190), "20 张卡的环形放置：只有坏相邻两张才丢")


def test_replicas():
    check(replica_counts(4, [10, 1, 1, 1], 3), [4, 1, 1, 1], "最热的专家拿走全部副本")
    check(replica_counts(4, [8, 6, 1, 1], 3), [3, 2, 1, 1], "依次给专家 0（8）、1（6）、0（8/2 = 4 > 6/2 = 3）")
    check(replica_counts(3, [5, 5, 5], 2), [2, 2, 1], "负载相等时编号小的优先")
    check(replica_counts(5, [1, 2, 3, 4, 5], 0), [1, 1, 1, 1, 1], "没有额外副本")
    counts = replica_counts(256, [1000 // (e + 1) for e in range(256)], 32)
    check(sum(counts), 288, "256 个专家 + 32 份额外副本")
    check(sum(c == 1 for c in counts), 256 - sum(c > 1 for c in counts), "大部分专家仍只有一份")
    check(sum(c > 1 for c in counts) <= 32, True, "拿到额外副本的专家不超过 32 个")
