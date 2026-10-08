# 有界负载的一致性哈希：沿环顺延，直到找到一台负载没超过上限的机器
import bisect
import hashlib
from collections import Counter


def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class Ring:
    def __init__(self, nodes, vnodes=100):
        self.points = sorted((h(f"{n}#{v}"), n) for n in nodes for v in range(vnodes))
        self.keys = [p for p, _ in self.points]

    def route(self, key, load=None, cap=None):
        i = bisect.bisect(self.keys, h(key))
        for step in range(len(self.points)):                  # 顺时针找第一台没超过上限的
            node = self.points[(i + step) % len(self.points)][1]
            if load is None or load[node] < cap:
                return node
        return self.points[i % len(self.points)][1]


nodes = [f"node{i}" for i in range(8)]
ring = Ring(nodes)
# 会话的热度差异很大：少数会话占了大部分请求
reqs = [f"session-{i % 20}" for i in range(4000)] + [f"session-{i}" for i in range(2000)]
for name, factor in [("不限上限", None), ("上限 = 平均 × 1.25", 1.25), ("上限 = 平均 × 1.05", 1.05)]:
    load = Counter({n: 0 for n in nodes})
    cap = None if factor is None else int(len(reqs) / len(nodes) * factor)
    hit = 0
    for r in reqs:
        node = ring.route(r, None if cap is None else load, cap)
        hit += node == ring.route(r)                          # 是否还落在"本来该去"的那台
        load[node] += 1
    hi = max(load.values()) / (len(reqs) / len(nodes))
    print(f"{name:18s} 最重的机器是平均的 {hi:.2f} 倍，亲和性（还去原来那台）{hit / len(reqs):5.1%}")
