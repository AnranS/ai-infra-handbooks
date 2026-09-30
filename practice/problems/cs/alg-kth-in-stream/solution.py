import heapq


class KthLargest:
    def __init__(self, k, nums=()):
        self.k = k
        self.heap = []
        for x in nums:
            self._push(x)

    def _push(self, x):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, x)
        elif self.k > 0 and x > self.heap[0]:
            heapq.heapreplace(self.heap, x)

    def add(self, x):
        self._push(x)
        return self.heap[0] if len(self.heap) == self.k and self.k > 0 else None


class MedianFinder:
    def __init__(self):
        self.small = []                        # 大顶堆（存负值），放较小的一半
        self.large = []                        # 小顶堆，放较大的一半

    def add(self, x):
        heapq.heappush(self.small, -x)
        heapq.heappush(self.large, -heapq.heappop(self.small))   # 先过一遍另一个堆
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def median(self):
        if not self.small:
            return None
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        return (-self.small[0] + self.large[0]) / 2
