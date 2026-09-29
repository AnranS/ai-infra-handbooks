import random
from collections import OrderedDict

from checker import check
from solution import TieredKVCache


def test_example():
    c = TieredKVCache(gpu_blocks=2, cpu_blocks=2)
    check([c.access(x) for x in "abc"], ["miss", "miss", "miss"], "前三次访问")
    check((c.gpu_contents(), c.cpu_contents()), (["b", "c"], ["a"]), "a 被卸载到 CPU")
    check(c.access("a"), "cpu", "a 从 CPU 加载回来")
    check((c.gpu_contents(), c.cpu_contents()), (["c", "a"], ["b"]), "b 被卸载")
    check(c.access("c"), "gpu", "c 在 GPU 上")
    check(c.stats, {"gpu_hits": 1, "cpu_hits": 1, "misses": 3, "offloads": 2, "loads": 1, "drops": 0}, "统计")


def test_cpu_full_drops():
    c = TieredKVCache(1, 1)
    for x in "abc":
        c.access(x)
    check((c.gpu_contents(), c.cpu_contents()), (["c"], ["b"]), "CPU 满了，a 被丢弃")
    check(c.stats["drops"], 1, "drops")
    check(c.access("a"), "miss", "被丢弃的块只能重算")


def test_no_cpu_tier():
    c = TieredKVCache(2, 0)
    for x in "abca":
        c.access(x)
    check(c.stats, {"gpu_hits": 0, "cpu_hits": 0, "misses": 4, "offloads": 0, "loads": 0, "drops": 2}, "没有 CPU 缓存")


def ref(gcap, ccap, seq):
    g, cp, out = OrderedDict(), OrderedDict(), []
    st = dict.fromkeys(("gpu_hits", "cpu_hits", "misses", "offloads", "loads", "drops"), 0)

    def put(h):
        if len(g) >= gcap:
            x, _ = g.popitem(last=False)
            if ccap == 0:
                st["drops"] += 1
            else:
                if len(cp) >= ccap:
                    cp.popitem(last=False)
                    st["drops"] += 1
                cp[x] = None
                st["offloads"] += 1
        g[h] = None

    for h in seq:
        if h in g:
            g.move_to_end(h)
            st["gpu_hits"] += 1
            out.append("gpu")
        elif h in cp:
            del cp[h]
            st["cpu_hits"] += 1
            st["loads"] += 1
            put(h)
            out.append("cpu")
        else:
            st["misses"] += 1
            put(h)
            out.append("miss")
    return out, st, list(g), list(cp)


def test_multi_turn_workload():
    rng = random.Random(0)
    for gcap, ccap in [(4, 8), (8, 0), (3, 3), (16, 64)]:
        seq = [rng.randint(0, 20) if rng.random() < 0.7 else rng.randint(0, 200) for _ in range(500)]
        c = TieredKVCache(gcap, ccap)
        got = [c.access(h) for h in seq]
        out, st, g, cp = ref(gcap, ccap, seq)
        check(got, out, f"GPU {gcap} / CPU {ccap} 的访问结果")
        check((c.stats, c.gpu_contents(), c.cpu_contents()), (st, g, cp), f"GPU {gcap} / CPU {ccap} 的最终状态")
