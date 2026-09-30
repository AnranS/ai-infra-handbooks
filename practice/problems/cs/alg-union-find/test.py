from checker import check
from solution import DSU, count_components, has_cycle


def test_example():
    d = DSU(5)
    d.union(0, 1)
    d.union(3, 4)
    check(d.count(), 3, "三个组")
    check(d.size(0), 2, "组的大小")
    check(count_components(5, [(0, 1), (3, 4)]), 3, "连通分量")
    check(has_cycle(3, [(0, 1), (1, 2), (2, 0)]), True, "三角形有环")


def test_union_returns():
    d = DSU(3)
    check(d.union(0, 1), True, "第一次合并")
    check(d.union(1, 0), False, "已经同组")
    check(d.union(0, 2), True, "合并第三个")
    check(d.count(), 1, "全部连通")


def test_size_from_any_member():
    d = DSU(4)
    d.union(0, 1)
    d.union(1, 2)
    for x in (0, 1, 2):
        check(d.size(x), 3, f"从任何成员查组大小都应该是 3（查的是 {x}）")
    check(d.size(3), 1, "孤立元素")


def test_edges():
    check(count_components(0, []), 0, "没有元素")
    check(count_components(3, []), 3, "没有边")
    check(has_cycle(3, []), False, "没有边就没有环")
    check(has_cycle(2, [(0, 1)]), False, "一条边")
    check(has_cycle(2, [(0, 1), (0, 1)]), True, "重复的边构成环")


def test_self_loop():
    check(has_cycle(1, [(0, 0)]), True, "自环")


def test_deep_chain():
    n = 50000
    d = DSU(n)
    for i in range(n - 1):                     # 按顺序合并，不压缩路径的实现会退化
        d.union(i, i + 1)
    check(d.count(), 1, "五万个元素连成一组")
    check(d.size(0), n, "组大小")
    check(d.find(0), d.find(n - 1), "首尾同组")
