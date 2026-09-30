from checker import check
from solution import Node, merge_k_lists


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
    lists = [build([1, 4, 5]), build([1, 3, 4]), build([2, 6])]
    check(to_list(merge_k_lists(lists)), [1, 1, 2, 3, 4, 4, 5, 6], "三条链表")


def test_edges():
    check(to_list(merge_k_lists([])), [], "没有链表")
    check(to_list(merge_k_lists([None])), [], "一条空链表")
    check(to_list(merge_k_lists([None, build([1]), None])), [1], "中间夹着空链表")
    check(to_list(merge_k_lists([build([1, 2, 3])])), [1, 2, 3], "只有一条")


def test_equal_values():
    lists = [build([1, 1]), build([1, 1]), build([1])]
    check(to_list(merge_k_lists(lists)), [1] * 5, "值全相同时不能比较节点")


def test_disjoint_ranges():
    lists = [build([7, 8, 9]), build([1, 2, 3]), build([4, 5, 6])]
    check(to_list(merge_k_lists(lists)), list(range(1, 10)), "区间互不重叠")


def test_terminated():
    merged = merge_k_lists([build([1, 3]), build([2])])
    node = merged
    steps = 0
    while node and steps < 10:                 # 末尾必须是 None，不能成环
        node = node.next
        steps += 1
    check(steps, 3, "链表长度正确且正常结束")


def test_large():
    k = 50
    lists = [build(list(range(i, 20000, k))) for i in range(k)]
    got = to_list(merge_k_lists(lists))
    check(len(got), 20000, "两万个节点")
    check(got == sorted(got), True, "结果有序")
