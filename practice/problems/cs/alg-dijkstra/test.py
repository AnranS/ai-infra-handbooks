from checker import check
from solution import dijkstra, network_delay


def test_example():
    edges = [(0, 1, 1), (1, 2, 2), (0, 2, 5), (2, 3, 1)]
    check(dijkstra(4, edges, 0), [0, 1, 3, 4], "绕路更短")
    check(network_delay(4, edges, 0), 4, "最远的节点")


def test_edges():
    check(dijkstra(1, [], 0), [0], "只有起点")
    check(dijkstra(2, [], 0), [0, None], "不可达")
    check(network_delay(2, [], 0), -1, "有节点到不了")
    check(dijkstra(0, [], 0), [], "没有节点")


def test_direction():
    check(dijkstra(2, [(0, 1, 5)], 1), [None, 0], "有向图，反方向走不通")


def test_multiple_edges():
    check(dijkstra(2, [(0, 1, 5), (0, 1, 2)], 0), [0, 2], "重边取小的")


def test_zero_weight():
    check(dijkstra(3, [(0, 1, 0), (1, 2, 0)], 0), [0, 0, 0], "零权边")


def test_longer_path_is_shorter():
    # 直达要 100，绕三步只要 3
    edges = [(0, 3, 100), (0, 1, 1), (1, 2, 1), (2, 3, 1)]
    check(dijkstra(4, edges, 0)[3], 3, "多跳但更短")


def test_large():
    n = 20000
    edges = [(i, i + 1, 1) for i in range(n - 1)] + [(0, n - 1, 10 ** 6)]
    dist = dijkstra(n, edges, 0)
    check(dist[n - 1], n - 1, "沿链走更短")
    check(network_delay(n, edges, 0), n - 1, "最远距离")
