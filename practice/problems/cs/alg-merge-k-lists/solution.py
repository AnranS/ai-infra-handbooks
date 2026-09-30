import heapq


class Node:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt


def merge_k_lists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))   # 序号用来打破平局，避免比较 Node
    dummy = Node(None)
    tail = dummy
    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    tail.next = None
    return dummy.next
