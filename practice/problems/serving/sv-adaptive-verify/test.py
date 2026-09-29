import itertools
import random

from checker import check, check_close
from solution import plan, survival


def rate(confs, counts, step_time, draft_time):
    gain = sum(sum(survival(c)[:k]) for c, k in zip(confs, counts))
    return (len(confs) + gain) / (draft_time + step_time(len(confs) + sum(counts)))


def test_example():
    got = survival([0.9, 0.8, 0.5])
    check(len(got), 3, "每个位置一个存活概率")
    for g, want in zip(got, [0.9, 0.72, 0.36]):
        check_close(g, want, what="存活概率是前缀乘积")
    confs = [[0.9] * 4, [0.5] * 4]
    check(plan(confs, lambda n: 1.0, 0.1), [4, 4], "访存受限：全部验证")
    counts = plan(confs, lambda n: 0.4 * n, 0.0)
    check(counts[0] >= counts[1], True, f"准的请求放行得更多：{counts}")


def test_compute_bound():
    # 每个 token 固定耗时 1：只有存活概率大于"当前平均产出率"的位置才值得放行
    confs = [[0.95, 0.9, 0.9], [0.3, 0.9, 0.9]]
    check(plan(confs, lambda n: float(n), 0.0), [0, 0], "纯算力受限、没有草稿开销时：多验证一个位置的收益不超过它的代价")
    counts = plan([[0.99, 0.99, 0.99]], lambda n: max(1.0, n / 2), 0.0)
    check(counts, [1], "单个请求：前两个 token（1 个真实 + 1 个草稿）不增加耗时，第三个起每个 token 多 0.5")


def test_brute_force():
    rng = random.Random(0)
    for _ in range(40):
        confs = [[round(rng.uniform(0.2, 0.99), 2) for _ in range(3)] for _ in range(rng.randint(1, 3))]
        knee = rng.uniform(1, 8)
        step = lambda n, knee=knee: max(1.0, n / knee)
        draft = rng.choice([0.0, 0.2])
        got = plan(confs, step, draft)
        best = max(rate(confs, list(c), step, draft) for c in itertools.product(range(4), repeat=len(confs)))
        check_close(rate(confs, got, step, draft), best, rtol=1e-9, what=f"和枚举所有放行组合的最优值一致（{confs}）")
        for c, k in zip(confs, got):
            check(0 <= k <= len(c), True, "放行个数在 0 到 K 之间")
