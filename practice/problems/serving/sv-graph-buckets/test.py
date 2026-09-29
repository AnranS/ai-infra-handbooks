import itertools
import random

from checker import check, check_close, time_limit
from solution import best_sizes, expected_waste, pad_to


def brute(dist, n, k):
    best = None
    for combo in itertools.combinations(range(1, n), k - 1):
        sizes = list(combo) + [n]
        w = expected_waste(dist, sizes)
        key = (round(w, 12), sizes)
        if best is None or key < best:
            best = key
    return best[1]


def test_example():
    dist = {1: 10, 2: 5, 3: 30, 7: 25, 8: 5}
    check(best_sizes(dist, 8, 3), [3, 7, 8], "best_sizes")
    check(pad_to(5, [1, 2, 4, 8]), 8, "pad_to(5)")
    check(pad_to(9, [1, 2, 4, 8]), None, "超过最大值")


def test_expected_waste():
    dist = {1: 1, 3: 1, 5: 2, 20: 1}
    check_close(expected_waste(dist, [1, 4, 8]), (0 + 1 + 3 * 2) / 5, what="expected_waste")
    check(expected_waste({}, [1, 2]), 0.0, "空分布")


def test_small_against_brute_force():
    rng = random.Random(0)
    for trial in range(40):
        n = rng.randint(2, 9)
        k = rng.randint(1, n)
        dist = {b: rng.randint(0, 20) for b in rng.sample(range(1, n + 1), rng.randint(1, n))}
        got = best_sizes(dist, n, k)
        check(got, brute(dist, n, k), f"随机用例 {trial}：max_bs={n}, k={k}, dist={dist}")


def test_must_include_max_and_k():
    dist = {b: 1 for b in range(1, 33)}
    s = best_sizes(dist, 32, 6)
    check((len(s), s[-1], s == sorted(set(s))), (6, 32, True), "k 个、包含 max_bs、升序不重复")


def test_large_fast():
    rng = random.Random(1)
    dist = {}
    for _ in range(5000):
        b = min(256, max(1, int(rng.expovariate(1 / 30))))
        dist[b] = dist.get(b, 0) + 1
    with time_limit(3.0, "max_bs=256, k=20 的动态规划"):
        s = best_sizes(dist, 256, 20)
    naive = [1, 2, 4, 8] + list(range(16, 257, 16))
    assert expected_waste(dist, s) <= expected_waste(dist, naive[:19] + [256]) + 1e-12, "应该不比常见的固定档位差"
