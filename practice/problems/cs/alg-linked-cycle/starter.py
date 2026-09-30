class Node:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt


def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        if slow.val == fast.val:               # 比较值而不是节点：值重复的无环链表会误判
            return True
        slow, fast = slow.next, fast.next.next
    return False


def cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            return slow                        # 相遇点不是入环点
    return None
