import heapq


class KthLargest:
    def __init__(self, k, nums=()):
        self.k = k
        self.data = list(nums)

    def add(self, x):
        self.data.append(x)
        self.data.sort(reverse=True)           # 每次都排序：O(n log n)
        return self.data[self.k - 1] if len(self.data) >= self.k else None


class MedianFinder:
    def __init__(self):
        self.heap = []

    def add(self, x):
        heapq.heappush(self.heap, x)

    def median(self):
        if not self.heap:
            return None
        return self.heap[len(self.heap) // 2]  # 堆里的顺序不是有序的
