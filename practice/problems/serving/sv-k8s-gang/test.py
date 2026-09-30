from checker import check
from solution import defrag_gain, fragmentation, place_group


def nodes(*pairs):
    return [{"name": name, "free": free} for name, free in pairs]


def test_example():
    ns = nodes(("a", 2), ("b", 2), ("c", 4))
    check(place_group(ns, 2, 2), [("a", 2), ("b", 2)], "装箱：先用剩得少的")
    check([n["free"] for n in ns], [0, 0, 4], "资源被扣减")
    check(fragmentation(nodes(("a", 2), ("b", 2), ("c", 4)), 4), (8, 4), "只有 c 能整除出 4 卡")
    check(defrag_gain(nodes(("a", 2), ("b", 2), ("c", 4)), 4), 1, "整理后能多放一个")


def test_rollback():
    ns = nodes(("a", 2), ("b", 1))
    check(place_group(ns, 2, 2), None, "第二个成员放不下")
    check([n["free"] for n in ns], [2, 1], "整组回滚，资源没被占用")


def test_single_member():
    ns = nodes(("a", 8))
    check(place_group(ns, 1, 8), [("a", 8)], "整机一个成员")
    check(ns[0]["free"], 0, "占满")


def test_binpack_order():
    ns = nodes(("a", 8), ("b", 2))
    check(place_group(ns, 1, 2), [("b", 2)], "优先用剩得最少但够用的节点，保住整机")
    check(ns[0]["free"], 8, "a 还是完整的")


def test_fragmentation():
    check(fragmentation(nodes(("a", 3), ("b", 3)), 2), (6, 4), "每个节点只能整除出 2")
    check(fragmentation(nodes(("a", 8)), 8), (8, 8), "整机没有碎片")
    check(fragmentation(nodes(("a", 1), ("b", 1)), 2), (2, 0), "都放不下")
    check(defrag_gain(nodes(("a", 1), ("b", 1)), 2), 1, "整理后能放下一个")


def test_no_gain_when_clean():
    ns = nodes(("a", 8), ("b", 8))
    check(defrag_gain(ns, 8), 0, "本来就不碎")
    check(defrag_gain(ns, 4), 0, "4 卡也不碎")


def test_many_members():
    ns = nodes(("a", 8), ("b", 8), ("c", 8))
    placed = place_group(ns, 6, 4)
    check(len(placed), 6, "六个成员都放下了")
    check(sum(n["free"] for n in ns), 0, "24 张卡刚好用完")
    check(place_group(ns, 1, 1), None, "再也放不下了")
