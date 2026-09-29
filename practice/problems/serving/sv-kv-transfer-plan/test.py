import random

import numpy as np

from checker import check
from solution import apply_plan, transfer_plan


def slots(n, table, bs):
    return [table[p // bs] * bs + p % bs for p in range(n)]


def min_segments(n, st, sbs, dt, dbs):
    s, d = slots(n, st, sbs), slots(n, dt, dbs)
    return sum(1 for i in range(n) if i == 0 or s[i] != s[i - 1] + 1 or d[i] != d[i - 1] + 1)


def test_example():
    plan = transfer_plan(40, [5, 6, 2], 16, [3, 1], 32)
    check(plan, [(80, 96, 32), (32, 32, 8)], "块大小 16 → 32：源块 5、6 相邻，目标块 3 里也连续，合并成一段")


def test_contiguous_blocks_merge():
    check(transfer_plan(48, [7, 8, 9], 16, [0, 1, 2], 16), [(112, 0, 48)], "两边块号都连续：一段搞定")
    check(transfer_plan(5, [3], 16, [9], 32), [(48, 288, 5)], "不足一块")


def test_random_plans():
    rng = random.Random(0)
    for trial in range(100):
        sbs, dbs = rng.choice([4, 8, 16]), rng.choice([4, 8, 16, 32])
        n = rng.randint(0, 100)
        nsb, ndb = -(-n // sbs), -(-n // dbs)
        st = rng.sample(range(40), nsb) if rng.random() < 0.5 else list(range(10, 10 + nsb))
        dt = rng.sample(range(40), ndb) if rng.random() < 0.5 else list(range(3, 3 + ndb))
        plan = transfer_plan(n, st, sbs, dt, dbs)
        src = np.arange(40 * sbs) * 10 + 7
        dst = np.full(40 * 32, -1)
        apply_plan(src, dst, plan)
        s, d = slots(n, st, sbs), slots(n, dt, dbs)
        check(dst[d].tolist(), src[s].tolist(), f"随机用例 {trial}：拷贝结果")
        check(sum(x[2] for x in plan), n, f"随机用例 {trial}：拷贝的 token 总数")
        check(len(plan), min_segments(n, st, sbs, dt, dbs), f"随机用例 {trial}：段数最少")
