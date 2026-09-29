import heapq
from collections import Counter


def top_k_words(words: list[str], k: int) -> list[str]:
    counts = Counter(words)
    return heapq.nsmallest(k, counts, key=lambda w: (-counts[w], w))
