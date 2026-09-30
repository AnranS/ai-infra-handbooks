from checker import check
from solution import move_zeroes


def run(a):
    move_zeroes(a)
    return a


def test_example():
    check(run([0, 1, 0, 3, 12]), [1, 3, 12, 0, 0], "保持相对顺序")


def test_edges():
    check(run([]), [], "空数组")
    check(run([0]), [0], "只有一个零")
    check(run([1]), [1], "只有一个非零")
    check(run([0, 0, 0]), [0, 0, 0], "全是零")
    check(run([1, 2, 3]), [1, 2, 3], "没有零")


def test_consecutive_zeros():
    check(run([0, 0, 1, 0, 0, 2]), [1, 2, 0, 0, 0, 0], "连续的零")
    check(run([1, 0, 0, 2, 0, 3]), [1, 2, 3, 0, 0, 0], "零在中间")


def test_in_place():
    a = [0, 1, 0, 2]
    before = id(a)
    move_zeroes(a)
    check(id(a), before, "必须原地修改同一个列表")
    check(a, [1, 2, 0, 0], "结果正确")


def test_large():
    a = [0 if i % 3 == 0 else i for i in range(30000)]
    move_zeroes(a)
    check(a[:4], [1, 2, 4, 5], "大数组的前几个")
    check(a[-1], 0, "末尾是零")
    check(sum(1 for x in a if x == 0), 10000, "零的个数不变")
