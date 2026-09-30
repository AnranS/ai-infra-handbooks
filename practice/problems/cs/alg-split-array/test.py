from checker import check
from solution import split_array


def test_example():
    check(split_array([7, 2, 5, 10, 8], 2), 18, "[7,2,5] 和 [10,8]")
    check(split_array([1, 2, 3, 4, 5], 2), 9, "[1,2,3,4] 和 [5]")
    check(split_array([1, 4, 4], 3), 4, "每段一个")


def test_edges():
    check(split_array([5], 1), 5, "只有一段")
    check(split_array([1, 1, 1, 1], 4), 1, "段数等于元素个数")
    check(split_array([1, 1, 1, 1], 1), 4, "只能分一段")


def test_zeros():
    check(split_array([0, 0, 5], 2), 5, "有零")
    check(split_array([0, 0, 0], 2), 0, "全是零")


def test_exact_boundary():
    check(split_array([2, 2, 2, 2], 2), 4, "正好等于上限时不该切开")


def test_chunked_prefill():
    reqs = [512, 1024, 256, 2048, 128, 768]
    check(split_array(reqs, 3), 2048, "三块时每块至少能装 2048 个 token")
    check(split_array(reqs, 2), 2944, "两块时每块至少 2944")


def test_large():
    a = [i % 100 + 1 for i in range(20000)]
    got = split_array(a, 50)
    check(got * 50 >= sum(a), True, "50 段装得下所有元素")
    check(got <= sum(a) // 50 + 100, True, "接近平均值")
