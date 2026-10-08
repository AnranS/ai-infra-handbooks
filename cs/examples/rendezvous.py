# 最高随机权重（rendezvous hashing）：对每个候选机器算一个分数，取最高的那台。
# 不需要环和虚拟节点，天然支持按权重分配，删除一台机器时只有它的键会迁移
import hashlib
from collections import Counter


def score(key, node):
    return int.from_bytes(hashlib.blake2b(f"{key}@{node}".encode(), digest_size=8).digest(), "big")


def route(key, nodes, weights=None):
    if weights is None:
        return max(nodes, key=lambda n: score(key, n))
    # 带权重时用 -weight / ln(score)：分数越高、权重越大越容易被选中
    import math
    return max(nodes, key=lambda n: -weights[n] / math.log(score(key, n) / 2 ** 64))


keys = [f"session-{i}" for i in range(50000)]
eight = [f"node{i}" for i in range(8)]
nine = eight + ["node8"]
seven = eight[:-1]

load = Counter(route(k, eight) for k in keys)
print(f"8 台均分：最重 {max(load.values()) / (len(keys) / 8):.3f} 倍，最轻 {min(load.values()) / (len(keys) / 8):.3f} 倍")
print(f"加到 9 台：迁移 {sum(1 for k in keys if route(k, eight) != route(k, nine)) / len(keys):.1%}（理想 {1 / 9:.1%}）")
print(f"减到 7 台：迁移 {sum(1 for k in keys if route(k, eight) != route(k, seven)) / len(keys):.1%}（理想 {1 / 8:.1%}）")

w = {n: (3 if n == "node0" else 1) for n in eight}            # node0 的容量是别人的 3 倍
load = Counter(route(k, eight, w) for k in keys)
print(f"\n按权重（node0 权重 3，其余 1）：node0 拿到 {load['node0'] / len(keys):.1%}，理想 {3 / 10:.1%}")
