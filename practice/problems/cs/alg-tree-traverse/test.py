from checker import check
from solution import T, inorder, level_order, preorder


def sample():
    return T(1, T(2, T(4), T(5)), T(3, None, T(6)))


def test_example():
    root = sample()
    check(preorder(root), [1, 2, 4, 5, 3, 6], "前序")
    check(inorder(root), [4, 2, 5, 1, 3, 6], "中序")
    check(level_order(root), [[1], [2, 3], [4, 5, 6]], "层序")


def test_empty():
    check(preorder(None), [], "空树")
    check(inorder(None), [], "空树")
    check(level_order(None), [], "空树")


def test_single():
    root = T(9)
    check((preorder(root), inorder(root), level_order(root)), ([9], [9], [[9]]), "单节点")


def test_left_chain():
    root = T(1, T(2, T(3)))
    check(preorder(root), [1, 2, 3], "只有左子树")
    check(inorder(root), [3, 2, 1], "中序是倒序")
    check(level_order(root), [[1], [2], [3]], "每层一个")


def test_bst_inorder_sorted():
    root = T(5, T(3, T(1), T(4)), T(8, T(7), T(9)))
    check(inorder(root), [1, 3, 4, 5, 7, 8, 9], "二叉搜索树的中序是升序")


def test_deep_tree():
    root = T(0)
    cur = root
    for i in range(1, 20000):                  # 两万层的退化树：递归会爆栈
        cur.right = T(i)
        cur = cur.right
    check(preorder(root)[:3], [0, 1, 2], "迭代能处理很深的树")
    check(len(inorder(root)), 20000, "中序也是")
    check(len(level_order(root)), 20000, "层序有两万层")
