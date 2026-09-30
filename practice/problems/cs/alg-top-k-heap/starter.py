import heapq


def top_k_heap(nums, k):
    heap = []
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)
    return sorted(heap)                        # 忘了逆序，而且 k <= 0 时行为不对


def kth_largest(nums, k):
    return sorted(nums)[k - 1]                 # 排序的是升序，取的位置也不对
