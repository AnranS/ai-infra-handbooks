class Node:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt


def reverse(head):
    prev = None
    while head:
        head.next = prev                       # 先改指向：后继丢了，循环停在第二个节点
        prev = head
        head = head.next
    return prev


def reverse_between(head, left, right):
    pre = head                                 # 没有虚拟头节点：left == 1 时会错
    for _ in range(left - 1):
        pre = pre.next
    cur = pre.next
    for _ in range(right - left):
        moved = cur.next
        cur.next = moved.next
        moved.next = pre.next
        pre.next = moved
    return head
