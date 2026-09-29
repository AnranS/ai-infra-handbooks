import random

from checker import check
from solution import CacheAwareRouter


def test_example():
    r = CacheAwareRouter(num_replicas=2, block_size=2, capacity_blocks=100, load_weight=0.5)
    sys_prompt = [1, 2, 3, 4, 5, 6]
    check(r.route(sys_prompt + [7, 8]), (0, 0), "第一个请求：都没有缓存，负载相同，选 0")
    check(r.route([9, 9, 9, 9]), (1, 0), "不相关的请求：0 号负载更高，选 1")
    check(r.route(sys_prompt + [10, 11]), (0, 3), "共享系统提示词：0 号命中 3 块（3 - 0.5 > 0 - 0.5）")


def test_load_breaks_hotspot():
    r = CacheAwareRouter(2, 2, 100, load_weight=1.5)
    p = [1, 2, 3, 4]
    check(r.route(p + [5, 6])[0], 0, "第 1 个")
    check(r.route(p + [7, 8])[0], 0, "第 2 个：0 号命中 2 块，2 - 1.5×1 = 0.5 > 1 号的 0")
    check(r.route(p + [9, 9])[0], 1, "第 3 个：0 号 2 - 1.5×2 = -1 < 0，改去 1 号")
    r.finish(0)
    r.finish(0)
    check(r.route(p + [3, 3]), (0, 2), "0 号请求结束后，又回到 0 号")


def test_lru_capacity():
    r = CacheAwareRouter(1, 1, capacity_blocks=3, load_weight=0.0)
    r.route([1, 2, 3])            # 索引：h(1) h(12) h(123)
    r.route([5])                  # 淘汰 h(1)
    check(r.route([1, 2, 3])[1], 0, "第一个块被淘汰后，整条前缀都不能命中（链式哈希）")


def ref_run(n, bs, cap, w, reqs, finishes):
    from collections import OrderedDict

    load, index, out = [0] * n, [OrderedDict() for _ in range(n)], []

    def hs(t):
        res, p = [], None
        for s in range(0, len(t) - bs + 1, bs):
            p = hash((p, tuple(t[s:s + bs])))
            res.append(p)
        return res

    for i, t in enumerate(reqs):
        h = hs(t)
        cands = []
        for rr in range(n):
            m = 0
            while m < len(h) and h[m] in index[rr]:
                m += 1
            cands.append((-(m - w * load[rr]), load[rr], rr, m))
        _, _, rr, m = min(cands)
        for x in h:
            if x in index[rr]:
                index[rr].move_to_end(x)
            else:
                index[rr][x] = None
                if len(index[rr]) > cap:
                    index[rr].popitem(last=False)
        load[rr] += 1
        out.append((rr, m))
        for f in finishes.get(i, []):
            load[f] -= 1
    return out


def test_random_against_reference():
    rng = random.Random(0)
    prefixes = [[rng.randint(0, 9) for _ in range(rng.randint(2, 12))] for _ in range(5)]
    reqs = [rng.choice(prefixes) + [rng.randint(0, 9) for _ in range(rng.randint(0, 6))] for _ in range(200)]
    for n, bs, cap, w in [(3, 2, 20, 0.5), (4, 3, 8, 1.0), (2, 1, 50, 0.0)]:
        want = ref_run(n, bs, cap, w, reqs, {})
        r = CacheAwareRouter(n, bs, cap, w)
        got = []
        for t in reqs:
            rep, m = r.route(t)
            got.append((rep, m))
        check(got, want, f"随机请求序列（{n} 个副本，block={bs}，容量 {cap}，权重 {w}）")
