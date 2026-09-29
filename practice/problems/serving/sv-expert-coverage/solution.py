from fractions import Fraction
from itertools import combinations


def lost_experts(cards, dead):
    alive = {e for g, c in enumerate(cards) if g not in dead for e in c}
    return sorted({e for c in cards for e in c} - alive)


def min_failures_to_lose(cards):
    where = {}
    for g, c in enumerate(cards):
        for e in c:
            where.setdefault(e, set()).add(g)
    return min(len(s) for s in where.values())


def prob_no_loss(cards, k):
    n = len(cards)
    combos = list(combinations(range(n), k))
    good = sum(not lost_experts(cards, set(dead)) for dead in combos)
    return Fraction(good, len(combos))


def replica_counts(num_experts, load, extra):
    counts = [1] * num_experts
    for _ in range(extra):
        best = max(range(num_experts), key=lambda e: (load[e] / counts[e], -e))
        counts[best] += 1
    return counts
