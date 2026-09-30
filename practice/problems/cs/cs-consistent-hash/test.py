from collections import Counter

from checker import check, check_close
from solution import Ring, h


def keys(n, prefix="k"):
    return [f"{prefix}{i}" for i in range(n)]


def test_example_rebalance():
    r8, r9 = Ring([f"node{i}" for i in range(8)]), Ring([f"node{i}" for i in range(9)])
    check_close(r8.rebalance_ratio(r9, keys(4000)), 1 / 9, rtol=0.3, what="加第 9 台约迁移 1/9")


def test_wraparound():
    r = Ring(["a", "b"], vnodes=4)
    biggest = max(r.positions)
    # 构造一个哈希值大于环上所有点的键：它必须回绕到第一个点
    key = next(k for k in keys(3000, "wrap") if h(k) > biggest)
    check(r.route(key), r.points[0][1], "超过最大点要回绕到环的开头")


def test_deterministic_and_stable():
    r = Ring(["a", "b", "c"])
    ks = keys(200)
    check([r.route(k) for k in ks] == [Ring(["a", "b", "c"]).route(k) for k in ks], True, "同样的输入结果一样")
    check(set(r.route(k) for k in ks) == {"a", "b", "c"}, True, "三个节点都用上了")


def test_vnodes_improve_balance():
    ks = keys(6000)
    spread = {}
    for v in (1, 100):
        load = Counter(Ring([f"n{i}" for i in range(8)], vnodes=v).route(k) for k in ks)
        spread[v] = max(load.values()) / (len(ks) / 8)
    check(spread[100] < spread[1], True, f"虚拟节点多的更均衡（{spread[1]:.2f} -> {spread[100]:.2f}）")
    check(spread[100] < 1.4, True, "100 个虚拟节点时最重的节点不超过平均的 1.4 倍")


def test_bounded_load():
    nodes = [f"n{i}" for i in range(4)]
    r = Ring(nodes)
    load = Counter()
    hot = ["hot"] * 400
    for k in hot:                                       # 全是同一个热键
        load[r.route_bounded(k, load, cap=120)] += 1
    check(max(load.values()) <= 120, True, "谁都不会超过上限")
    check(sum(load.values()), 400, "所有请求都被安排了")


def test_bounded_falls_back():
    r = Ring(["a", "b"])
    load = {"a": 10, "b": 10}
    check(r.route_bounded("x", load, cap=5) in ("a", "b"), True, "全都超上限时退回普通路由")
    check(r.route_bounded("x", load, cap=5), r.route("x"), "退回的是本来该去的那台")


def test_remove_node():
    r8 = Ring([f"node{i}" for i in range(8)])
    r7 = Ring([f"node{i}" for i in range(7)])
    ratio = r8.rebalance_ratio(r7, keys(4000))
    check_close(ratio, 1 / 8, rtol=0.3, what="删一台只影响它自己的键")
