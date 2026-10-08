# 链表的三板斧：虚拟头节点、快慢指针、原地反转
class Node:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt


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


def remove_all(head, target):
    """删除所有值为 target 的节点：虚拟头节点让"删除头节点"不用特判"""
    dummy = Node(None, head)
    prev = dummy
    while prev.next:
        if prev.next.val == target:
            prev.next = prev.next.next         # 跳过它，prev 不动
        else:
            prev = prev.next
    return dummy.next


def middle(head):
    """快慢指针：快指针一次两步，慢指针一次一步，快到头时慢正好在中点"""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow.val if slow else None


def has_cycle(head):
    """判环：有环时快指针一定会追上慢指针"""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            return True
    return False


def reverse(head):
    """原地反转：三个指针，prev 是新的后继"""
    prev = None
    while head:
        head.next, prev, head = prev, head, head.next
    return prev


print("删除所有 2：", to_list(remove_all(build([2, 1, 2, 3, 2]), 2)))
print("删除头节点：", to_list(remove_all(build([1, 1, 1]), 1)), "（虚拟头节点省掉了特判）")
print("中间节点：", middle(build([1, 2, 3, 4, 5])), "（偶数个时取靠后的：", middle(build([1, 2, 3, 4])), "）")
print("反转：", to_list(reverse(build([1, 2, 3, 4, 5]))))

loop = build([1, 2, 3, 4])
tail = loop
while tail.next:
    tail = tail.next
tail.next = loop.next                          # 4 -> 2，成环
print("有环吗：", has_cycle(loop), "；无环的链表：", has_cycle(build([1, 2, 3])))
