import heapq


class Node:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt


def merge_k_lists(lists):
    heap = []
    for node in lists:
        if node:
            heapq.heappush(heap, (node.val, node))      # 值相等时会去比较 Node，抛 TypeError
    dummy = Node(None)
    tail = dummy
    while heap:
        _, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, node.next))
    return dummy.next
