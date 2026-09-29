import random

from checker import check
from solution import h2o_keep, simulate_h2o, streaming_keep


def test_example():
    check(streaming_keep(10, 2, 3), [0, 1, 7, 8, 9], "streaming_keep(10, sink=2, window=3)")
    check(h2o_keep([5, 1, 9, 2, 3, 0.5], budget=4, recent=2), [0, 2, 4, 5], "h2o_keep")


def test_small_sequences():
    check(streaming_keep(4, 2, 3), [0, 1, 2, 3], "不超过 sink + window")
    check(h2o_keep([1, 2, 3], 5, 1), [0, 1, 2], "不超过 budget")
    check(h2o_keep([1, 1, 1, 1, 1], 3, 1), [0, 1, 4], "同分取靠前的")


def test_simulate_small():
    steps = [{0: 1.0}, {0: 0.9, 1: 0.1}, {0: 0.1, 1: 0.8, 2: 0.1}, {0: 0.2, 1: 0.1, 2: 0.3, 3: 0.4}]
    # 第 3 步后缓存 4 个 > 3：分数 0:2.2, 1:1.0, 2:0.4, 3:0.4；保留最近 1 个（3）+ 分数最高的 2 个（0、1）
    check(simulate_h2o(steps, budget=3, recent=1), [0, 1, 3], "4 步之后的缓存")


def ref_sim(steps, budget, recent):
    cache, score = [], {}
    for t, row in enumerate(steps):
        cache.append(t)
        score[t] = 0.0
        for p, w in row.items():
            score[p] += w
        if len(cache) > budget:
            rec = cache[-recent:] if recent else []
            rest = sorted(cache[:len(cache) - recent], key=lambda p: (-score[p], p))[:budget - recent]
            keep = sorted(set(rec) | set(rest))
            for p in set(cache) - set(keep):
                del score[p]
            cache = keep
    return cache


def test_simulate_random():
    rng = random.Random(0)
    for trial in range(20):
        budget, recent = rng.randint(2, 8), rng.randint(0, 2)
        cache, steps = [], []
        for t in range(40):
            cache.append(t)
            ws = [rng.random() ** 3 for _ in cache]
            s = sum(ws)
            steps.append({p: w / s for p, w in zip(cache, ws)})
            cache = ref_sim(steps, budget, recent)
        check(simulate_h2o(steps, budget, recent), ref_sim(steps, budget, recent), f"随机用例 {trial}")
