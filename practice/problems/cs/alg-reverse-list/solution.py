class Node:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt


def reverse(head):
    prev = None
    while head:
        nxt = head.next                        # 先存后继
        head.next = prev
        prev, head = head, nxt
    return prev


def reverse_between(head, left, right):
    dummy = Node(None, head)
    pre = dummy
    for _ in range(left - 1):
        pre = pre.next
    cur = pre.next
    for _ in range(right - left):              # 头插法：把 cur 后面的节点依次挪到 pre 后面
        moved = cur.next
        cur.next = moved.next
        moved.next = pre.next
        pre.next = moved
    return dummy.next
