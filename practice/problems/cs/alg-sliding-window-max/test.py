from checker import check
from solution import sliding_max


def test_example():
    check(sliding_max([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7], "经典例子")
    check(sliding_max([1], 1), [1], "单元素")


def test_edges():
    check(sliding_max([], 3), [], "空数组")
    check(sliding_max([1, 2], 0), [], "k 为 0")
    check(sliding_max([1, 2, 3], 5), [3], "k 比数组长")
    check(sliding_max([1, 2, 3], 1), [1, 2, 3], "窗口为 1")


def test_monotone():
    check(sliding_max([1, 2, 3, 4], 2), [2, 3, 4], "递增")
    check(sliding_max([4, 3, 2, 1], 2), [4, 3, 2], "递减")


def test_duplicates():
    check(sliding_max([2, 2, 2], 2), [2, 2], "全相同")
    check(sliding_max([1, 3, 3, 1], 2), [3, 3, 3], "重复的最大值")


def test_max_leaves_window():
    check(sliding_max([9, 1, 2, 3], 2), [9, 2, 3], "最大值滑出窗口后要换人")
    check(sliding_max([9, 1, 1, 1, 1], 3), [9, 1, 1], "连续滑出")


def test_against_brute_force():
    import random
    rng = random.Random(17)
    for _ in range(200):
        a = [rng.randrange(-20, 20) for _ in range(rng.randrange(1, 30))]
        k = rng.randrange(1, len(a) + 1)
        want = [max(a[i:i + k]) for i in range(len(a) - k + 1)]
        check(sliding_max(a, k), want, f"与暴力一致：{a}, k={k}")


def test_large():
    n = 100000
    a = [(i * 7919) % 1000 for i in range(n)]
    got = sliding_max(a, 1000)
    check(len(got), n - 999, "结果长度")
    check(max(got) <= 999, True, "值域正确")
