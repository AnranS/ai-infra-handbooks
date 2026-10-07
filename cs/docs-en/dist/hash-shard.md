# Consistent hashing and sharding: which machine should the data live on

<p class="lead">Add one machine to a KV cache cluster and, with modulo sharding, 88% of the keys have to move, as measured in this chapter. The cache is wiped out and the back end is hit all at once. Consistent hashing brings that number down to the theoretical best of 11%. This chapter works through the three ways to shard (modulo, consistent hashing, highest random weight), what virtual nodes actually solve, how bounded loads trade affinity against balance, and where all of this lands in an inference system: prefix-cache routing, distributed KV storage, and the distribution of experts under expert parallelism.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. With `hash(key) % n` sharding, roughly what fraction of keys has to migrate when the machine count goes from 8 to 9? What is the ideal figure?
    2. Why does consistent hashing need virtual nodes? What goes wrong with too few, and with too many?
    3. How does consistent hashing handle the skew caused by one particularly hot key?
    4. What is highest random weight (rendezvous hashing) better at than consistent hashing?
    5. Where is all of this used in an inference system?

??? success "Answers for the self-test (answer first, then open this)"
    1. Modulo sharding migrates about 88.8% (measured in this chapter), because the divisor changed and almost every key's owner changed with it. Ideally only 1/9, about 11.1%, has to move: the share the new machine ought to take.
    2. With one point per machine on the ring, the arcs the machines receive differ widely: this chapter measures the heaviest machine at 2.34 times the average and the lightest at only 0.49. Putting many virtual nodes per machine is like sampling repeatedly, which smooths the load out: at 100 virtual nodes the heaviest is 1.24 times, at 500 it is 1.05. The price of too many is memory and lookup cost (the points on the ring are the machine count times the virtual-node count) plus more points to update when scaling.
    3. Hashing alone cannot solve it: a hot key's hash is fixed and always lands on the same machine. The answer is **bounded loads**, moving along the ring to the next machine that has not exceeded a load cap. The tighter the cap the more even the load, but the worse the affinity (going back to the original machine): this chapter measures 91% affinity with the cap at 1.25 times the average and only 79% at 1.05 times.
    4. It needs no ring and no virtual nodes, and the implementation is a few lines; the load is naturally even (this chapter measures the heaviest at 1.024 times); weighting falls out naturally; and removing a machine moves only its own keys. The price is one hash per candidate machine on every routing decision, which is slower than a binary search on the ring once there are many machines, so it suits a small candidate set (replica selection, a cache cluster of a few dozen machines).
    5. Cache-aware routing for the prefix cache (sending the same session or prefix to the same instance), distributed KV storage (a KV pool sharded by key, as in Mooncake Store), the mapping of experts to GPUs under expert parallelism, and any case of pinning requests to a fixed instance by some id (multi-turn conversations, LoRA adapter affinity).

## The problem with modulo sharding {#取模分片的问题}

The simplest sharding: `node = hash(key) % n`. It has exactly one problem, and it is fatal:

```python title="modulo.py"
# modulo sharding: add one machine, and how many keys have to move?
import hashlib


def h(key):
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


keys = [f"session-{i}" for i in range(100000)]
print("机器数变化    需要迁移的键")
for n in (4, 8, 16, 32):
    moved = sum(1 for k in keys if h(k) % n != h(k) % (n + 1))
    print(f"{n:3d} -> {n + 1:3d}    {moved / len(keys):6.1%}")
print("\n理想情况：加一台机器只需要迁移 1/(n+1)：", ", ".join(f"{1 / (n + 1):.1%}" for n in (4, 8, 16, 32)))
```

```text title="output"
机器数变化    需要迁移的键
  4 ->   5     80.2%
  8 ->   9     88.8%
 16 ->  17     94.2%
 32 ->  33     97.0%

理想情况：加一台机器只需要迁移 1/(n+1)： 20.0%, 11.1%, 5.9%, 3.0%
```

Change the divisor and almost every key's owner changes. For a cache cluster this means the hit rate drops to zero the moment you scale out and every request falls through to the back end; for stateful storage it means moving nearly all of the data.

## Consistent hashing {#一致性哈希}

Picture the hash space as a ring (0 to 2^64). Each machine occupies a number of points on the ring, each key hashes to a point as well, and **the first machine point found clockwise** owns it. When a machine is added it takes one arc from its neighbour on the ring and no other key moves.

Try it first: how many keys have to move when a machine is added, and how the virtual-node count flattens the load.

<div class="aig-widget" data-widget="hashring"></div>

```python title="ring.py"
# the consistent hash ring: each machine puts vnodes virtual nodes on the ring, and the first virtual node a key finds clockwise owns it
import bisect
import hashlib
from collections import Counter


def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class Ring:
    def __init__(self, nodes, vnodes=1):
        self.vnodes = vnodes
        self.points = sorted((h(f"{n}#{v}"), n) for n in nodes for v in range(vnodes))
        self.keys = [p for p, _ in self.points]

    def route(self, key):
        i = bisect.bisect(self.keys, h(key))
        return self.points[i % len(self.points)][1]


keys = [f"session-{i}" for i in range(100000)]
print("每台机器的虚拟节点数   负载最重/最轻   加一台机器要迁移的键")
for vnodes in (1, 10, 100, 500):
    before = Ring([f"node{i}" for i in range(8)], vnodes)
    after = Ring([f"node{i}" for i in range(9)], vnodes)
    load = Counter(before.route(k) for k in keys)
    moved = sum(1 for k in keys if before.route(k) != after.route(k))
    hi, lo = max(load.values()), min(load.values())
    print(f"{vnodes:12d} {hi / (len(keys) / 8):16.2f} / {lo / (len(keys) / 8):.2f} {moved / len(keys):14.1%}")
print("\n加第 9 台机器时，理想的迁移比例是 1/9 =", f"{1 / 9:.1%}")
```

```text title="output"
每台机器的虚拟节点数   负载最重/最轻   加一台机器要迁移的键
           1             2.34 / 0.49           8.8%
          10             1.20 / 0.74          11.1%
         100             1.24 / 0.86          11.7%
         500             1.05 / 0.98          11.2%

加第 9 台机器时，理想的迁移比例是 1/9 = 11.1%
```

Two things to read together:

- **The migration volume**: whatever the virtual-node count, adding the 9th machine moves about 11% of the keys, close to the theoretical best of 1/9. That is the whole point of consistent hashing.
- **The balance**: with one point per machine the ring is cut very unevenly and the heaviest machine is 2.34 times the average. Virtual nodes are the answer: scatter 100 to 500 points per machine, which is like sampling repeatedly, and the heaviest machine falls to 1.05 to 1.24 times.

How many virtual nodes to use is a trade-off between balance and the size of the ring: the points on the ring are the machine count times the virtual-node count, each routing decision is one binary search (O(log points)), and scaling inserts or removes the matching points. In a cluster of a few dozen machines, 100 to 500 virtual nodes each is a common choice.

Virtual nodes have another benefit: the count can be allocated by a machine's capacity (more for the faster machines), which gives weighted sharding.

## Hot keys and bounded loads {#热键与有界负载}

Consistent hashing only guarantees an even load when the keys are even. In an inference service the keys are often far from even: a popular system prompt, a session accessed over and over, a hit LoRA adapter. A hot key's hash is fixed, so it always lands on the same machine.

**Consistent hashing with bounded loads** adds one rule: if the machine found clockwise is already over a cap (say 1.25 times the average load), keep going until one that is not is found.

```python title="bounded.py"
# consistent hashing with bounded loads: walk along the ring until a machine under the load cap is found
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
        for step in range(len(self.points)):                  # find the first one clockwise that is under the cap
            node = self.points[(i + step) % len(self.points)][1]
            if load is None or load[node] < cap:
                return node
        return self.points[i % len(self.points)][1]


nodes = [f"node{i}" for i in range(8)]
ring = Ring(nodes)
# sessions differ widely in popularity: a few of them take most of the requests
reqs = [f"session-{i % 20}" for i in range(4000)] + [f"session-{i}" for i in range(2000)]
for name, factor in [("不限上限", None), ("上限 = 平均 × 1.25", 1.25), ("上限 = 平均 × 1.05", 1.05)]:
    load = Counter({n: 0 for n in nodes})
    cap = None if factor is None else int(len(reqs) / len(nodes) * factor)
    hit = 0
    for r in reqs:
        node = ring.route(r, None if cap is None else load, cap)
        hit += node == ring.route(r)                          # whether it still lands on the machine it was supposed to go to
        load[node] += 1
    hi = max(load.values()) / (len(reqs) / len(nodes))
    print(f"{name:18s} 最重的机器是平均的 {hi:.2f} 倍，亲和性（还去原来那台）{hit / len(reqs):5.1%}")
```

```text title="output"
不限上限               最重的机器是平均的 1.69 倍，亲和性（还去原来那台）100.0%
上限 = 平均 × 1.25     最重的机器是平均的 1.25 倍，亲和性（还去原来那台）90.9%
上限 = 平均 × 1.05     最重的机器是平均的 1.05 倍，亲和性（还去原来那台）79.2%
```

The tighter the cap the more even the load, but the more requests get reassigned to another machine and the worse the affinity (going back to the original machine). For the prefix cache, a reassignment means a cache miss and a fresh prefill. So the cap is a product parameter: loosen it (1.25 to 1.5) when the cache gain is large, tighten it (1.05 to 1.1) when the load matters more.

Two details that go with it:

- **A reassignment needs a memory**: once a session has been reassigned to a machine, its later requests should go there too (otherwise it bounces between two machines and builds a cache on both). The usual approach is to cache the session-to-instance mapping for a short while.
- **A reassignment needs hysteresis**: when the load flaps around the threshold, do not decide again for every request, or you get a herd effect.

## Highest random weight (rendezvous hashing) {#最高随机权重rendezvous-hashing}

There is a simpler way to shard: compute a score `hash(key + node)` for each candidate machine and take the highest.

```python title="rendezvous.py"
# highest random weight (rendezvous hashing): compute a score for each candidate machine and take the highest.
# no ring and no virtual nodes needed, weighting falls out naturally, and removing a machine moves only its own keys
import hashlib
from collections import Counter


def score(key, node):
    return int.from_bytes(hashlib.blake2b(f"{key}@{node}".encode(), digest_size=8).digest(), "big")


def route(key, nodes, weights=None):
    if weights is None:
        return max(nodes, key=lambda n: score(key, n))
    # with weights, use -weight / ln(score): a higher score and a larger weight make it likelier to be picked
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

w = {n: (3 if n == "node0" else 1) for n in eight}            # node0 has 3 times the capacity of the others
load = Counter(route(k, eight, w) for k in keys)
print(f"\n按权重（node0 权重 3，其余 1）：node0 拿到 {load['node0'] / len(keys):.1%}，理想 {3 / 10:.1%}")
```

```text title="output"
8 台均分：最重 1.024 倍，最轻 0.984 倍
加到 9 台：迁移 11.1%（理想 11.1%）
减到 7 台：迁移 12.8%（理想 12.5%）

按权重（node0 权重 3，其余 1）：node0 拿到 29.9%，理想 30.0%
```

- **The load is naturally even**: no virtual nodes needed, and with 8 machines the heaviest is only 2.4% above the average.
- **The migration volume is optimal**: adding a machine moves 1/(n+1), and removing one moves only the keys on that machine.
- **Weighting is natural**: use `-weight / ln(score)` as the comparison value and the distribution follows the weights (node0 with weight 3 above gets 29.9%, against an ideal 30%).
- **It can also pick the top k directly**: sort by score and take the first k machines, which are this key's k replicas, and removing one of them leaves the others unchanged.

The price is a hash per candidate machine on every routing decision, O(n). With a few dozen candidates this is no problem at all (a few dozen hashes is nanoseconds); with thousands, use the ring. A common combination in practice is consistent hashing on the outside to find a shard group, and highest random weight inside the group to pick a replica.

## Where it lands in an inference system {#在推理系统里的落点}

| Case | How it is used | What to watch |
| --- | --- | --- |
| Cache-aware routing | hash the session id or the prefix to route to a fixed instance, hitting the prefix cache and saving a whole prefill | use bounded loads so a hot session does not blow one machine up; on scaling, only 1/N of the sessions lose their cache (see [The prefix cache](serving://engine/prefix-cache/)) |
| Distributed KV storage | KV blocks sharded by key across machines (Mooncake Store, LMCache and others) | a block's key is usually a prefix hash, which is naturally content-addressable; pick replicas with highest random weight (see [The KV transfer engine and distributed KV storage](serving://comm/kv-storage/)) |
| Expert parallelism | the mapping of experts to GPUs; hot experts have to be replicated across cards | the key here is an expert id, few and fixed in number, so an explicit mapping table plus a load-balancing algorithm is usual rather than hashing (see [Large-scale expert-parallel deployment](serving://moe/ep-deploy/)) |
| Multi-LoRA serving | requests for one adapter go to one instance to avoid loading it repeatedly | many adapters with very uneven popularity, which suits bounded loads |
| Sharded metrics and logs | written by instance id | time-series writes are usually sharded by time and instance hash together, to avoid a hot spot |

One premise deserves emphasis: **sharding only works if the key is the right one**. Prefix-cache routing shards better on the hash of the first N tokens than on a session id, because different sessions may share a system prompt; sharding by user id pins all of one user's conversations to one machine, which gives cache locality but lets a large customer create skew.

## Beyond sharding: rebalancing and metadata {#分片之外再平衡与元数据}

A real system has two more things to handle:

- **How the rebalancing proceeds**. The migration is not instantaneous: either write to both places, move gradually, then switch the reads, or allow reads from the old location for a while (falling back to the source on a miss). For cache-like data (KV blocks, the prefix cache), the easiest approach is not to migrate at all: let the old location expire naturally and let new requests rebuild the cache in the new one, at the cost of a period of lower hit rate.
- **Who knows the current sharding**. Every client has to agree on which machines exist now, or they will route to different places. The usual approach is to keep the membership list in a strongly consistent metadata service (etcd, ZooKeeper) and have clients watch for changes. That needs the next chapter's consensus algorithms.

!!! interview "How to answer in an interview"
    Asked about sharding: start with modulo's problem, that going from 8 machines to 9 moves about 89% of the keys and wipes the cache. Consistent hashing hashes both keys and machines onto a ring and takes the first machine point clockwise, so adding a machine affects only the neighbouring arc and the migration falls to the theoretical best of 1/(n+1); balance comes from virtual nodes (100 to 500 each takes the heaviest machine from 2.3 times to 1.05), and the virtual-node count doubles as a weight. Hot keys are handled with bounded loads: go to the next machine past the cap, where a tighter cap means more balance and less affinity, which is a product parameter. The other option is highest random weight: compute hash(key+node) per machine and take the maximum, with no virtual nodes, naturally even load, support for weights and for taking the top k replicas, at the price of O(n) routing, which suits a small candidate set. Finish on inference: cache-aware routing, distributed KV storage and multi-LoRA affinity are all this machinery, and all of it needs a strongly consistent membership list.

## Exercises {#练习}

**1. Compute the migration.** A 16-machine prefix-cache cluster is scaled to 20. With consistent hashing, roughly what fraction of sessions changes owner? If each of those sessions' next request has to prefill again at 300 ms, what does the moment of scaling do to the p99? How would you soften it?

??? success "Answer"
    The migrated fraction is about the share of new capacity: 4/20, so 20% of sessions move to a new machine. Their next request loses the prefix cache and its time to first token grows by 300 ms. If that 20% arrives within a few seconds of the scale-up, the p99 clearly degrades.

    To soften it: (1) add the new machines in batches (one at a time, minutes apart) to spread the shock; (2) keep traffic off a new machine at first, or give it a weight that ramps up (a virtual-node count growing from few to many); (3) allow a fallback: when a new machine finds nothing locally, try pulling the KV from the previous owner (possible with distributed KV storage); (4) accept it, since scaling usually happens while load is rising and this cost buys overall capacity.

**2. How many virtual nodes.** A 200-machine cluster with 500 virtual nodes each. How many points are on the ring? How many comparisons does one routing decision take? And with highest random weight? What order of magnitude is each one's memory?

??? success "Answer"
    100,000 points on the ring, about 17 comparisons for the binary search, and at 16 bytes per point about 1.6 MB of memory. Highest random weight computes a hash for each of the 200 machines (200 of them), an order of magnitude slower than the binary search, but it stores no ring and its memory is just the machine list.

    So a large cluster uses the ring and a small one uses highest random weight; you can also do both in two layers: the ring to find a shard group (a few dozen groups), then highest random weight inside the group to pick the machine and the replicas.

**3. Design a cache route.** Design a routing policy for the prefix cache: the input is a request's first 512 tokens and the current queue depth of each instance, and the output is the target instance. Write the pseudocode and explain how to choose three of the parameters.

??? success "Answer"
    <!-- i18n:diagram f7e3db8b34 -->
    ```text
    key = hash(the first 512 tokens, chunked)   # chunked hashing, so requests sharing a system prompt land together
    candidates = the next 3 machines clockwise from key on the consistent hash ring
    sort the candidates by (estimated matched length, -queue depth)
    if the best candidate's queue depth > the average x C: move on to the next candidate
    record session -> chosen instance, and reuse it for later requests within the TTL
    ```

    The three parameters: (1) **the prefix truncation length** (512): too short and different requests are mistaken for sharing a prefix, too long and requests that differ only at the end go to different instances; set it from your system prompt's length. (2) **The load cap coefficient C**: loosen it to 1.5 when the cache gain is large (a long system prompt), tighten it to 1.1 when latency matters. (3) **The session mapping's TTL**: slightly longer than a typical gap between conversation turns (5 minutes, say); too long and it keeps pointing at the old instance after a scaling event.

**4. Judge a claim.** "We shard our KV storage with consistent hashing, so scaling out does not have to migrate data." Is that right?

??? success "Answer"
    Not entirely. Consistent hashing guarantees that **the amount of data to migrate is minimal** (1/(n+1)), not that there is none. For persistent storage, that 1/(n+1) really does have to be moved, otherwise the new machine cannot find it and the data on the old machine is never accessed again.

    Only **a rebuildable cache** (KV cache blocks, the prefix cache) can skip the migration: if the new location does not have it, compute it again, and the data in the old location expires by itself. The difference is whether the data can be regenerated after it is lost.

## Summary {#小结}

- [x] Modulo sharding migrates almost every key on a scale-up (about 89% from 8 machines to 9); consistent hashing brings it to the theoretical best of 1/(n+1).
- [x] Virtual nodes solve uneven distribution: 100 to 500 each takes the heaviest machine from 2.3 times to 1.05 to 1.24, and the count doubles as a weight.
- [x] Hot keys are handled with bounded loads: go past the cap to the next machine, where a tighter cap means more balance and less affinity; reassignment needs a memory and hysteresis.
- [x] Highest random weight is simpler to implement, naturally even, and supports weights and multiple replicas, at the price of O(n) routing, which suits a small candidate set.
- [x] Where it lands in an inference system: cache-aware routing, distributed KV storage, multi-LoRA affinity; a rebuildable cache can skip the migration, persistent data cannot.
- [x] The sharding scheme itself needs a strongly consistent membership list, which needs consensus algorithms.
