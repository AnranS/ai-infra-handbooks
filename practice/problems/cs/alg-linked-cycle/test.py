from checker import check
from solution import Node, cycle_start, has_cycle


def build(vals, pos=-1):
    """pos 是入环点的下标，-1 表示无环"""
    nodes = [Node(v) for v in vals]
    for a, b in zip(nodes, nodes[1:]):
        a.next = b
    if pos >= 0 and nodes:
        nodes[-1].next = nodes[pos]
    return (nodes[0] if nodes else None), nodes


def test_example():
    head, nodes = build([1, 2, 3, 4], pos=1)
    check(has_cycle(head), True, "有环")
    check(cycle_start(head) is nodes[1], True, "入环点是第二个节点")


def test_no_cycle():
    head, _ = build([1, 2, 3])
    check(has_cycle(head), False, "无环")
    check(cycle_start(head), None, "无环时返回 None")


def test_empty_and_single():
    check(has_cycle(None), False, "空链表")
    head, nodes = build([1])
    check(has_cycle(head), False, "单节点无环")
    head, nodes = build([1], pos=0)
    check(has_cycle(head), True, "自环")
    check(cycle_start(head) is nodes[0], True, "自环的入环点是它自己")


def test_duplicate_values():
    head, _ = build([1, 1, 1, 1])
    check(has_cycle(head), False, "值重复但无环")


def test_whole_list_is_cycle():
    head, nodes = build([1, 2, 3], pos=0)
    check(cycle_start(head) is nodes[0], True, "整条链表成环")


def test_long():
    n = 20000
    head, nodes = build(list(range(n)), pos=n // 2)
    check(has_cycle(head), True, "长链表有环")
    check(cycle_start(head) is nodes[n // 2], True, "入环点正确")
