from checker import check
from solution import can_finish, find_order


def test_example():
    check(can_finish(2, [(1, 0)]), True, "简单依赖")
    check(can_finish(2, [(1, 0), (0, 1)]), False, "互相依赖")
    check(find_order(4, [(1, 0), (2, 0), (3, 1), (3, 2)]), [0, 1, 2, 3], "一种可行顺序")


def test_edges():
    check(can_finish(0, []), True, "没有课")
    check(find_order(0, []), [], "没有课")
    check(can_finish(3, []), True, "没有依赖")
    check(find_order(3, []), [0, 1, 2], "按编号")


def test_self_loop():
    check(can_finish(1, [(0, 0)]), False, "自己依赖自己")


def test_direction():
    # (1, 0) 表示先修 0 再修 1，所以顺序必须是 0 在前
    order = find_order(2, [(1, 0)])
    check(order, [0, 1], "方向不能反")
    order = find_order(2, [(0, 1)])
    check(order, [1, 0], "反过来的依赖")


def test_chain():
    prereq = [(i + 1, i) for i in range(5)]
    check(find_order(6, prereq), [0, 1, 2, 3, 4, 5], "一条链")


def test_cycle_in_part():
    check(can_finish(4, [(1, 0), (2, 1), (1, 2), (3, 0)]), False, "一部分成环")
    check(find_order(4, [(1, 0), (2, 1), (1, 2), (3, 0)]), [], "返回空列表")


def test_large():
    n = 20000
    prereq = [(i + 1, i) for i in range(n - 1)]
    order = find_order(n, prereq)
    check(len(order), n, "两万门课")
    check(order[:3], [0, 1, 2], "顺序正确")
