import heapq
import itertools
from collections import deque


def chunked(iterable, n):
    if n <= 0:
        raise ValueError("n 必须是正整数")

    def gen():
        it = iter(iterable)
        while chunk := tuple(itertools.islice(it, n)):
            yield chunk

    return gen()


def sliding_window(iterable, n):
    if n <= 0:
        raise ValueError("n 必须是正整数")

    def gen():
        it = iter(iterable)
        window = deque(itertools.islice(it, n - 1), maxlen=n)
        for x in it:
            window.append(x)
            yield tuple(window)

    return gen()


def merge_sorted(*iterables):
    heap = []
    for i, iterable in enumerate(iterables):
        it = iter(iterable)
        for first in it:
            heap.append((first, i, it))
            break
    heapq.heapify(heap)
    while heap:
        value, i, it = heap[0]
        yield value
        for nxt in it:
            heapq.heapreplace(heap, (nxt, i, it))
            break
        else:
            heapq.heappop(heap)
