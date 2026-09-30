from checker import check
from solution import search_rotated


def test_example():
    a = [4, 5, 6, 7, 0, 1, 2]
    check(search_rotated(a, 0), 4, "找到")
    check(search_rotated(a, 3), -1, "不存在")
    check(search_rotated([1], 1), 0, "单元素")


def test_two_elements():
    check(search_rotated([3, 1], 1), 1, "两个元素，旋转过")
    check(search_rotated([3, 1], 3), 0, "两个元素，找第一个")
    check(search_rotated([1, 3], 3), 1, "两个元素，没旋转")


def test_right_end():
    a = [5, 6, 7, 0, 1, 2, 4]
    check(search_rotated(a, 4), 6, "答案在最右端")
    check(search_rotated(a, 5), 0, "答案在最左端")


def test_not_rotated():
    a = list(range(10))
    for i, x in enumerate(a):
        check(search_rotated(a, x), i, f"没旋转时也要对：{x}")
    check(search_rotated(a, 99), -1, "不存在")


def test_all_positions():
    base = list(range(20))
    for k in range(20):                        # 所有旋转位置都试一遍
        a = base[k:] + base[:k]
        for x in base:
            check(a[search_rotated(a, x)], x, f"旋转 {k} 位后找 {x}")


def test_empty():
    check(search_rotated([], 1), -1, "空数组")
