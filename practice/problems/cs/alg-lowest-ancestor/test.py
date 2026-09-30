from checker import check
from solution import T, lowest_ancestor, max_depth


def sample():
    n4, n7 = T(4), T(7)
    n2 = T(2, n7, n4)
    n6 = T(6)
    n5 = T(5, n6, n2)
    n0, n8 = T(0), T(8)
    n1 = T(1, n0, n8)
    root = T(3, n5, n1)
    return root, {n.val: n for n in (root, n5, n1, n6, n2, n7, n4, n0, n8)}


def test_example():
    root, n = sample()
    check(lowest_ancestor(root, n[5], n[1]) is root, True, "根节点")
    check(lowest_ancestor(root, n[5], n[4]) is n[5], True, "一个是另一个的祖先")
    check(lowest_ancestor(root, n[7], n[4]) is n[2], True, "在子树内部")


def test_same_node():
    root, n = sample()
    check(lowest_ancestor(root, n[6], n[6]) is n[6], True, "两个是同一个节点")


def test_two_nodes():
    a, b = T(1), T(2)
    root = T(0, a, b)
    check(lowest_ancestor(root, a, b) is root, True, "最简单的情形")


def test_duplicate_values():
    leaf1, leaf2 = T(1), T(1)
    left = T(2, leaf1)
    right = T(3, leaf2)
    root = T(0, left, right)
    check(lowest_ancestor(root, leaf1, leaf2) is root, True, "值重复时要按节点判断")
    check(lowest_ancestor(root, leaf1, left) is left, True, "同一棵子树里")


def test_max_depth():
    root, n = sample()
    check(max_depth(root), 4, "最大深度")
    check(max_depth(None), 0, "空树")
    check(max_depth(T(1)), 1, "单节点")


def test_deep():
    root = T(0)
    cur = root
    for i in range(1, 20000):
        cur.left = T(i)
        cur = cur.left
    check(max_depth(root), 20000, "两万层的树也要能算")
