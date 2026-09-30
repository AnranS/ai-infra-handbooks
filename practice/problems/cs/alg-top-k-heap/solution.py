import heapq
import random


def top_k_heap(nums, k):
    if k <= 0 or not nums:
        return []
    heap = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)         # 一次操作完成弹出和压入
    return sorted(heap, reverse=True)


def kth_largest(nums, k):
    if not nums or k <= 0 or k > len(nums):
        return None
    rng = random.Random(0)
    items = list(nums)
    while True:
        pivot = items[rng.randrange(len(items))]
        bigger = [x for x in items if x > pivot]
        equal = [x for x in items if x == pivot]
        if k <= len(bigger):
            items = bigger                     # 第 k 大在"更大"那一份里
        elif k <= len(bigger) + len(equal):
            return pivot
        else:
            k -= len(bigger) + len(equal)      # 去"更小"那一份里找第 k 大
            items = [x for x in items if x < pivot]
