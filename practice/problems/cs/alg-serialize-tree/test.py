from checker import check
from solution import T, deserialize, serialize


def shape(node):
    """把树转成嵌套元组，方便比较结构"""
    if node is None:
        return None
    return (node.val, shape(node.left), shape(node.right))


def test_example():
    root = T(1, T(2), T(3, T(4), T(5)))
    check(serialize(root), "1,2,3,#,#,4,5", "层序加占位符")
    check(shape(deserialize("1,2,3,#,#,4,5")), shape(root), "还原成同一棵树")
    check(serialize(None), "", "空树")


def test_empty():
    check(deserialize(""), None, "空串还原成空树")
    check(serialize(deserialize("")), "", "来回一遍还是空")


def test_single():
    check(serialize(T(7)), "7", "单节点没有多余的占位符")
    check(shape(deserialize("7")), (7, None, None), "还原")


def test_left_chain():
    root = T(1, T(2, T(3)))
    check(serialize(root), "1,2,#,3", "只有左子树")
    check(shape(deserialize(serialize(root))), shape(root), "来回一致")


def test_right_chain():
    root = T(1, None, T(2, None, T(3)))
    check(serialize(root), "1,#,2,#,3", "只有右子树")
    check(shape(deserialize(serialize(root))), shape(root), "来回一致")


def test_negative_values():
    root = T(-1, T(-2), T(3))
    check(serialize(root), "-1,-2,3", "负数")
    check(shape(deserialize(serialize(root))), shape(root), "来回一致")


def test_round_trip_random():
    import random
    rng = random.Random(13)

    def build(depth):
        if depth == 0 or rng.random() < 0.3:
            return None
        return T(rng.randrange(-50, 50), build(depth - 1), build(depth - 1))

    for _ in range(100):
        root = build(5)
        check(shape(deserialize(serialize(root))), shape(root), "随机树来回一致")


def test_large():
    root = T(0)
    nodes = [root]
    for i in range(1, 5000):                   # 完全二叉树
        parent = nodes[(i - 1) // 2]
        node = T(i)
        if i % 2:
            parent.left = node
        else:
            parent.right = node
        nodes.append(node)
    data = serialize(root)
    check(shape(deserialize(data)), shape(root), "五千个节点来回一致")
    check(data.count("#") < 5000, True, "末尾的占位符被裁掉了")
