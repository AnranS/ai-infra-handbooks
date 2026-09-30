from checker import check
from solution import longest_path, topo_order


def test_example():
    check(topo_order(4, [(0, 1), (0, 2), (1, 3), (2, 3)]), [0, 1, 2, 3], "菱形依赖")
    check(topo_order(2, [(0, 1), (1, 0)]), None, "有环")
    check(longest_path(4, [(0, 1), (0, 2), (1, 3), (2, 3)]), 3, "三层")


def test_edges():
    check(topo_order(0, []), [], "没有节点")
    check(topo_order(3, []), [0, 1, 2], "没有边时按编号")
    check(longest_path(3, []), 1, "全部并行，只要一层")


def test_self_loop():
    check(topo_order(1, [(0, 0)]), None, "自环也是环")
    check(longest_path(1, [(0, 0)]), -1, "有环返回 -1")


def test_chain():
    edges = [(i, i + 1) for i in range(5)]
    check(topo_order(6, edges), [0, 1, 2, 3, 4, 5], "一条链")
    check(longest_path(6, edges), 6, "链长就是层数")


def test_lexicographic():
    check(topo_order(4, [(2, 3)]), [0, 1, 2, 3], "可选时按编号从小到大")
    check(topo_order(3, [(2, 0)]), [1, 2, 0], "1 没有依赖，排在最前")


def test_partial_cycle():
    check(topo_order(4, [(0, 1), (1, 2), (2, 1), (2, 3)]), None, "图里有一个环")


def test_large():
    n = 20000
    edges = [(i, i + 1) for i in range(n - 1)]
    check(longest_path(n, edges), n, "两万个节点的长链")
    order = topo_order(n, edges)
    check(order[:3], [0, 1, 2], "顺序正确")
