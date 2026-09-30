from checker import check
from solution import lis_dp, lis_fast


def test_example():
    check(lis_dp([10, 9, 2, 5, 3, 7, 101, 18]), 4, "经典例子")
    check(lis_fast([0, 1, 0, 3, 2, 3]), 4, "快速版")


def test_edges():
    check(lis_dp([]), 0, "空数组")
    check(lis_fast([]), 0, "空数组")
    check(lis_dp([5]), 1, "单元素")
    check(lis_dp([7, 7, 7]), 1, "严格递增：全相同只算 1")
    check(lis_fast([7, 7, 7]), 1, "快速版也是")


def test_monotone():
    check(lis_dp([1, 2, 3, 4]), 4, "已经递增")
    check(lis_dp([4, 3, 2, 1]), 1, "递减")
    check(lis_fast([4, 3, 2, 1]), 1, "快速版")


def test_replace_matters():
    # 只追加不替换的写法在这里会答错：结尾必须保持尽量小
    check(lis_fast([3, 4, 5, 1, 2, 3, 4, 5, 6]), 6, "后半段更长")
    check(lis_dp([3, 4, 5, 1, 2, 3, 4, 5, 6]), 6, "两种写法一致")


def test_two_methods_agree():
    import random
    rng = random.Random(7)
    for _ in range(200):
        a = [rng.randrange(30) for _ in range(rng.randrange(1, 25))]
        check(lis_fast(a), lis_dp(a), f"两种写法必须一致：{a}")


def test_large():
    n = 30000
    a = list(range(n))
    check(lis_fast(a), n, "三万个递增元素")
    a = [(i * 7919) % n for i in range(n)]     # 伪随机排列
    check(lis_fast(a) > 100, True, "随机排列的 LIS 约为 2√n")
