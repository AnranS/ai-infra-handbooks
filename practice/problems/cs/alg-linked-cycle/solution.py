class Node:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt


def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            return True
    return False


def cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:                       # 相遇：让一个指针回到头，同速再走
            probe = head
            while probe is not slow:
                probe, slow = probe.next, slow.next
            return probe
    return None
