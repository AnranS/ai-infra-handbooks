import bisect
import random

from checker import check
from solution import count_of, lower_bound, upper_bound


def test_example():
    a = [1, 2, 2, 2, 3]
    check((lower_bound(a, 2), upper_bound(a, 2), count_of(a, 2)), (1, 4, 3), "重复元素")
    check(count_of([1, 3], 2), 0, "不存在")


def test_edges():
    check(lower_bound([], 1), 0, "空数组")
    check(upper_bound([], 1), 0, "空数组")
    check(lower_bound([2], 1), 0, "比所有元素都小")
    check(lower_bound([2], 3), 1, "比所有元素都大")
    check(upper_bound([2], 2), 1, "正好等于唯一的元素")


def test_all_same():
    a = [5] * 7
    check((lower_bound(a, 5), upper_bound(a, 5)), (0, 7), "全相同")
    check(count_of(a, 5), 7, "出现 7 次")


def test_against_bisect():
    rng = random.Random(1)
    for _ in range(3000):
        a = sorted(rng.randrange(8) for _ in range(rng.randrange(10)))
        x = rng.randrange(9)
        check(lower_bound(a, x), bisect.bisect_left(a, x), f"lower_bound 与标准库一致：{a}, {x}")
        check(upper_bound(a, x), bisect.bisect_right(a, x), f"upper_bound 与标准库一致：{a}, {x}")


def test_large():
    a = list(range(0, 200000, 2))
    check(lower_bound(a, 199998), 99999, "最后一个元素")
    check(lower_bound(a, 199999), 100000, "比最后一个大")
    check(count_of(a, 4), 1, "每个偶数出现一次")
