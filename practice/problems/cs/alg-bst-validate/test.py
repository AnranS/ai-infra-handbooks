from checker import check
from solution import T, is_bst, kth_smallest


def test_example():
    check(is_bst(T(2, T(1), T(3))), True, "合法的 BST")
    check(is_bst(T(5, T(1), T(4, T(3), T(6)))), False, "右子树里有比根小的")
    check(kth_smallest(T(3, T(1, None, T(2)), T(4)), 2), 2, "第二小")


def test_edges():
    check(is_bst(None), True, "空树")
    check(is_bst(T(1)), True, "单节点")
    check(kth_smallest(None, 1), None, "空树")
    check(kth_smallest(T(1), 0), None, "k 从 1 开始")
    check(kth_smallest(T(1), 2), None, "k 越界")


def test_strict():
    check(is_bst(T(2, T(2), None)), False, "等于根不算合法")
    check(is_bst(T(2, None, T(2))), False, "右边等于根也不行")


def test_deep_violation():
    # 违规的节点藏在很深的地方：只比较父子会漏掉
    root = T(10, T(5, T(3), T(7, T(6), T(12))), T(15))
    check(is_bst(root), False, "12 在 10 的左子树里")


def test_kth_all():
    root = T(4, T(2, T(1), T(3)), T(6, T(5), T(7)))
    check([kth_smallest(root, k) for k in range(1, 8)], [1, 2, 3, 4, 5, 6, 7], "从小到大")


def test_deep_tree():
    root = T(0)
    cur = root
    for i in range(1, 20000):                  # 按有序数据插入退化成的链
        cur.right = T(i)
        cur = cur.right
    check(is_bst(root), True, "两万层的链也是合法 BST")
    check(kth_smallest(root, 3), 2, "第三小")
