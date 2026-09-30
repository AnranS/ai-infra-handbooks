from checker import check
from solution import Node, reverse, reverse_between


def build(vals):
    head = None
    for v in reversed(vals):
        head = Node(v, head)
    return head


def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


def test_example():
    check(to_list(reverse(build([1, 2, 3, 4, 5]))), [5, 4, 3, 2, 1], "整体反转")
    check(to_list(reverse_between(build([1, 2, 3, 4, 5]), 2, 4)), [1, 4, 3, 2, 5], "局部反转")


def test_edges():
    check(to_list(reverse(None)), [], "空链表")
    check(to_list(reverse(build([1]))), [1], "单节点")
    check(to_list(reverse(build([1, 2]))), [2, 1], "两个节点")


def test_between_edges():
    check(to_list(reverse_between(build([1, 2, 3]), 1, 3)), [3, 2, 1], "从头反转到尾")
    check(to_list(reverse_between(build([1, 2, 3]), 2, 2)), [1, 2, 3], "只有一个节点时不变")
    check(to_list(reverse_between(build([5]), 1, 1)), [5], "单节点")
    check(to_list(reverse_between(build([1, 2, 3, 4]), 1, 2)), [2, 1, 3, 4], "包含头节点")


def test_no_new_nodes():
    head = build([1, 2, 3])
    nodes = set()
    cur = head
    while cur:
        nodes.add(id(cur))
        cur = cur.next
    new_head = reverse(head)
    cur = new_head
    while cur:
        check(id(cur) in nodes, True, "必须原地反转，不能新建节点")
        cur = cur.next


def test_long_list():
    n = 50000
    head = build(list(range(n)))
    got = to_list(reverse(head))               # 递归写法会在这里 RecursionError
    check(got[0], n - 1, "五万个节点的长链表")
    check(got[-1], 0, "末尾")
