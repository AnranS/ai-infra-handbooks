# Global scheduling in disaggregated architectures: KV-aware routing, xPyD ratios and overload control

<p class="lead">A single inference engine's scheduler decides "which requests run this step"; a disaggregated architecture adds a layer of <b>global scheduling</b>: which prefill instance and which decode instance a request goes to, how many instances prefill and decode each need, and whom to reject under overload. Mooncake's Conductor, NVIDIA Dynamo's Router and Planner, llm-d's inference gateway and SGLang's Router all do this. This chapter answers three questions with two simulations: should routing look at the cache or at load, why the xPyD ratio must change over time, and why "rejecting early" under overload actually lets more requests meet their targets.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What goes wrong with routing that looks only at cache hits? And only at load?
    2. How does "minimum predicted TTFT" routing weigh cache and queueing together?
    3. Why reject requests early under overload? On what basis?
    4. Why can't the ratio of prefill to decode instances be set once?
    5. How does Dynamo's Router know which blocks each instance has cached?

??? success "Answers (try first, then expand to compare)"
    1. Cache only: everyone's requests flood the few instances that cached hot prefixes, forming hotspots with growing queues while other instances sit empty; load only: requests are spread onto instances without the cache, and the same prefix is prefilled again on many instances, wasting compute and lengthening TTFT too.
    2. For each candidate instance, estimate "predicted TTFT = queueing time + compute time to prefill the part that missed", and pick the smallest: instances with more hits have less to compute but longer queues; idle instances queue less but compute more. One unified cost compares both on the same scale.
    3. Under overload requests queue without bound, every request times out, and everyone's SLO collapses together; by rejecting early the requests that cannot finish in time according to the predicted TTFT, the rest all meet the target (in this chapter's simulation the attainment rate rises from 86% to 98%).
    4. The shape of the load (prompt length, output length, hit rate) keeps changing, and the bottleneck jumps between prefill and decode, so any fixed ratio is wrong some of the time; a planner must watch the load at minute granularity and adjust the number of each kind of instance, accounting for the cost of switching roles.
    5. Workers publish KV events (which block hashes were stored or removed) when they allocate and evict KV blocks, and the Router subscribes to them to maintain a global index. The index is approximate; routing mistakes only cost extra compute, never wrong results.

## What global scheduling does {#全局调度要做的事}

In a KV-centric disaggregated architecture (KVCache-centric, in Mooncake's terms), the global scheduler sees a cluster: some prefill instances, some decode instances, and a KV cache pool across instances (the [previous part](../comm/kv-storage.md)). It must decide:

1. **Where prefill goes**: which instance has cached this request's prefix, and which has a short queue;
2. **Where decode goes**: which instance still has KV space, and whether TPOT can meet the target;
3. **Whether to admit**: when the predicted TTFT or TPOT misses the target, wait in the queue or reject immediately;
4. **The ratio**: how many prefill and decode instances each, and how to adjust as the load changes.

## Routing: trading cache off against queueing {#路由缓存与排队的权衡}

Simulate 8 prefill instances with a multi-turn conversation workload (20 system prompts, 400 sessions, each turn appending 300 tokens), with requests arriving as a Poisson process; each instance has its own prefix cache (LRU, about 190 thousand tokens), and prefill computes only the part that missed. Compare four routing policies, plus "early rejection under overload":

```python
import random
from collections import OrderedDict

BLOCK, INSTANCES, SPEED, SLO = 64, 8, 20000, 2.0     # block size; prefill instances; prefill tokens per second per instance; TTFT target (s)


def keys(tokens):
    out, h = [], 0
    for i in range(0, len(tokens) // BLOCK * BLOCK, BLOCK):
        h = hash((h, tuple(tokens[i:i + BLOCK])))
        out.append(h)
    return out


def workload(rate, seconds, seed=0):
    """多轮对话：20 种系统提示词（各 2K token）、400 个会话，每轮追加 300 个 token；请求按泊松过程到达"""
    rng = random.Random(seed)
    systems = [[rng.randrange(10**6) for _ in range(2048)] for _ in range(20)]
    convs = [list(rng.choice(systems)) for _ in range(400)]
    t, reqs = 0.0, []
    while t < seconds:
        t += rng.expovariate(rate)
        c = rng.randrange(len(convs))
        convs[c] = convs[c] + [rng.randrange(10**6) for _ in range(300)]
        reqs.append((t, list(convs[c])))
    return reqs


def simulate(reqs, policy, reject=False):
    rng = random.Random(1)
    cache = [OrderedDict() for _ in range(INSTANCES)]
    busy = [0.0] * INSTANCES                          # when each instance's queue runs out
    ttfts, rejected, hit_tokens, total_tokens = [], 0, 0, 0
    for t, tokens in reqs:
        ks = keys(tokens)

        def hit(i):
            n = 0
            for k in ks:
                if k not in cache[i]:
                    break
                n += 1
            return n * BLOCK

        def ttft(i):                                  # predicted time to first token: queueing + computing the part that missed
            return max(busy[i] - t, 0) + (len(tokens) - hit(i)) / SPEED

        if policy == "随机":
            i = rng.randrange(INSTANCES)
        elif policy == "最少排队":
            i = min(range(INSTANCES), key=lambda j: busy[j])
        elif policy == "只看缓存":
            i = max(range(INSTANCES), key=lambda j: (hit(j), -busy[j]))
        else:                                         # minimum predicted TTFT: weighs cache hits and queueing together
            i = min(range(INSTANCES), key=ttft)
        if reject and ttft(i) > SLO:                  # early rejection under overload: requests doomed to time out never enter the queue
            rejected += 1
            continue
        h = hit(i)
        busy[i] = max(busy[i], t) + (len(tokens) - h) / SPEED
        ttfts.append(busy[i] - t)
        hit_tokens, total_tokens = hit_tokens + h, total_tokens + len(tokens)
        for k in ks:
            cache[i][k] = True
            cache[i].move_to_end(k)
        while len(cache[i]) > 3000:                   # each instance can cache 3000 blocks (about 190 thousand tokens)
            cache[i].popitem(last=False)
    ttfts.sort()
    good = sum(x <= SLO for x in ttfts)
    return (f"命中率 {hit_tokens / total_tokens:5.1%}，TTFT 中位数 {ttfts[len(ttfts) // 2]:5.2f} s、"
            f"P99 {ttfts[int(len(ttfts) * 0.99)]:6.2f} s，达标 {good / len(reqs):5.1%}" + (f"，拒绝 {rejected / len(reqs):.1%}" if reject else ""))


for rate in (40, 70):
    reqs = workload(rate, 120)
    print(f"== 每秒 {rate} 个请求（{len(reqs)} 个）")
    for policy in ("随机", "最少排队", "只看缓存", "预计 TTFT 最小"):
        print(f"  {policy}：{simulate(reqs, policy)}")
    print(f"  预计 TTFT 最小 + 提前拒绝：{simulate(reqs, '预计 TTFT 最小', reject=True)}")
```

```text title="output"
== 每秒 40 个请求（4814 个）
  随机：命中率 53.5%，TTFT 中位数  0.12 s、P99   1.74 s，达标 99.5%
  最少排队：命中率 53.3%，TTFT 中位数  0.09 s、P99   0.42 s，达标 100.0%
  只看缓存：命中率 90.9%，TTFT 中位数  0.02 s、P99   0.17 s，达标 100.0%
  预计 TTFT 最小：命中率 84.4%，TTFT 中位数  0.02 s、P99   0.23 s，达标 100.0%
  预计 TTFT 最小 + 提前拒绝：命中率 84.4%，TTFT 中位数  0.02 s、P99   0.23 s，达标 100.0%，拒绝 0.0%
== 每秒 70 个请求（8456 个）
  随机：命中率 38.9%，TTFT 中位数  4.50 s、P99  79.64 s，达标 44.0%
  最少排队：命中率 39.1%，TTFT 中位数  3.76 s、P99  74.19 s，达标 47.1%
  只看缓存：命中率 81.7%，TTFT 中位数  0.02 s、P99  31.56 s，达标 85.3%
  预计 TTFT 最小：命中率 75.8%，TTFT 中位数  0.03 s、P99  10.96 s，达标 86.4%
  预计 TTFT 最小 + 提前拒绝：命中率 78.6%，TTFT 中位数  0.02 s、P99   1.97 s，达标 97.9%，拒绝 2.1%
```

- **Under light load**, both cache-aware policies do well: a high hit rate means fewer tokens to compute per request, so queues are naturally short; random and least-queue can hit only the system prompts, for a hit rate of only about half;
- **Under heavy load**, the two policies that ignore the cache collapse completely (compute is wasted on repeated prefill, and queues keep growing); **cache only** piles sessions of the same kind onto the same instance, forming hotspots with a P99 of 31 seconds; **minimum predicted TTFT** (queueing time plus "how long the missed part takes to compute") trades hit rate against load automatically, bringing P99 down to 11 seconds;
- **Early rejection**: requests whose predicted TTFT already exceeds the target at routing time are doomed to time out if queued, and they drag down the requests behind them. Rejecting 2.1% of them outright brings the rest's P99 under 2 seconds, raising attainment from 86% to 98%.

This is the shared idea behind every global scheduler:

- **Mooncake's Conductor** picks both a prefill and a decode instance for each request, based on how much of the prefix cache can be reused, the instances' queues, and the TTFT / TPOT targets; under overload it rejects early based on predictions, looking not only at prefill but also predicting whether the decode side will have room by then, so decode is not unable to take over after prefill finishes;
- **NVIDIA Dynamo's KV Router**: each inference worker publishes events when KV blocks are stored or evicted, and the Router maintains a global index of "which worker has which blocks" (a tree organized by prefix), combining "the number of overlapping prefix blocks" and "each worker's current load" into one cost at routing time;
- **llm-d** does the same in Kubernetes' inference gateway: its endpoint picker scores each replica by prefix cache and load;
- **SGLang Router** (written in Rust) keeps an approximate radix tree per worker for its cache-aware policy, and falls back to shortest queue when the load gap exceeds a threshold (the practice problem bank has a simplified version of this problem).

The index a router maintains is always approximate: instances evict blocks on their own and events lag, so hit counts are only estimates. Fortunately a routing mistake only costs some extra prefill, never a wrong result.

## The xPyD ratio: following the load {#xpyd-配比跟着负载变}

PD disaggregation splits prefill and decode into two resource pools; the benefit is optimizing each separately, and the cost is that the **ratio** becomes a quantity that needs continuous adjustment. When the shape of the load changes, the bottleneck jumps from one side to the other:

```python
import math

P_CAP, D_CAP, TOTAL = 20000, 3000, 24        # prompt tokens per second one prefill instance computes; tokens per second one decode instance outputs under the TPOT target; total instances
PHASES = [                                   # (period, requests per second, average prompt length, average output length, prefix cache hit rate)
    ("白天：对话", 100, 2000, 500, 0.6),
    ("傍晚：长文档总结", 30, 16000, 300, 0.2),
    ("夜间：代码智能体", 25, 6000, 2000, 0.8),
]


def need(rate, isl, osl, hit):
    return math.ceil(rate * isl * (1 - hit) / P_CAP), math.ceil(rate * osl / D_CAP)


static = (12, 12)                            # 12P12D, set once by "the average"
print(f"固定 {static[0]}P{static[1]}D 与按时段调整的对比（共 {TOTAL} 个实例）：")
for name, rate, isl, osl, hit in PHASES:
    p, d = need(rate, isl, osl, hit)
    load = (rate * isl * (1 - hit) / (static[0] * P_CAP), rate * osl / (static[1] * D_CAP))
    verdict = "；".join(f"{side}负载 {x:.0%}" + ("（过载）" if x > 1 else "") for side, x in zip(("prefill ", "decode "), load))
    print(f"  {name}：需要 {p}P{d}D（{'够用' if p + d <= TOTAL else '不够'}）；固定配比下 {verdict}")
```

```text title="output"
固定 12P12D 与按时段调整的对比（共 24 个实例）：
  白天：对话：需要 4P17D（够用）；固定配比下 prefill 负载 33%；decode 负载 139%（过载）
  傍晚：长文档总结：需要 20P3D（够用）；固定配比下 prefill 负载 160%（过载）；decode 负载 25%
  夜间：代码智能体：需要 2P17D（够用）；固定配比下 prefill 负载 12%；decode 负载 139%（过载）
```

With the same 24 instances, each of the three periods has enough in total, but no fixed ratio satisfies all three: chat and agents (long outputs, high cache hits) need decode, while long documents (long prompts, low hits) need prefill. So disaggregated systems need a **planner**: Dynamo's Planner adds and removes prefill / decode workers based on metrics such as TTFT and TPOT and on queueing; Mooncake and other systems also support switching instances between prefill and decode roles. Switching costs time to load weights and warm up (CUDA Graphs, JIT kernels), and requests running on a decode instance must be migrated or run to completion, so planners usually adjust at minute granularity, leaving second-level fluctuations to routing and queueing.

Turn the dials on the shape of the load and the fixed ratio:

<div class="aig-widget" data-widget="pd-ratio"></div>

Another approach is **not fully disaggregating**: under light load or with short prompts, run prefill and decode on the same instance with chunked prefill and mixed scheduling, and use PD disaggregation only for very long prompts (some deployment modes of vLLM and SGLang support such "conditional disaggregation"). The gains of disaggregation come mainly from the interference between long prompts and decode; when the interference is mild, the transfer and ratio problems that disaggregation brings may not be worth it.

!!! interview "In an interview"
    After PD disaggregation comes up in a system design question, interviewers often follow up with "how do you route" and "how do you set the ratio". Routing: cache only creates hotspots, load only repeats prefill, so use a unified cost like "predicted TTFT = queueing + compute time of the part that missed", and under overload reject early based on predictions (in this chapter's simulation, attainment rises from 86% to 98%); the index is maintained from KV events published by workers and is approximate. The ratio: it shifts with the shape of the load, the bottleneck jumps between the two sides, and a planner must adjust at minute granularity, noting the cost of switching roles. Mentioning Mooncake's Conductor and Dynamo's Router and Planner shows you know the industry's solutions.

## Exercises {#练习}

**1. Why is "cache only" best under light load?** In this chapter's simulation, under light load "cache only" even has a lower P99 than "minimum predicted TTFT". Explain why, and say when this advantage disappears.

??? success "Answer"
    Under light load there is almost no queueing, and TTFT is set mainly by "the number of tokens to compute", so the policy with the highest hit rate is naturally fastest; "minimum predicted TTFT" occasionally gives up a hit to avoid a little queueing and ends up computing more. As load rises, sessions of the same kind concentrate on the same instance and form hotspots, queueing time starts to dominate, and the cache-only policy cannot spread requests out, so P99 deteriorates sharply (31 seconds at 70 requests per second in this chapter). A common practice in real systems: prefer the cache while the load gap is small and switch to load once the gap exceeds a threshold (SGLang Router's balance threshold), or use a unified cost function directly.

**2. The cost of early rejection.** Early rejection raises attainment from 86% to 98%, but rejects 2.1% of requests. In a real service, what does "rejection" mean? What alternatives are there?

??? success "Approach"
    On the user side, rejection usually shows up as "service busy, please retry", or a downgrade to a smaller model or a shorter output. Alternatives include: rejecting by priority (paying users and interactive requests first, batch requests deferred); placing requests in a "slow queue" with a relaxed SLO; triggering the planner to scale up (though scaling takes minutes); and rate limiting at the entrance so overload never actually happens. The key to early rejection is "early": decide before the request consumes any prefill compute, or you both waste compute and make the user wait a long time only to fail.

## Summary {#小结}

- [x] Global scheduling decides the choice of prefill / decode instances, admission and the ratio; Mooncake's Conductor, Dynamo's Router and Planner, the llm-d gateway and SGLang Router all do this.
- [x] Routing must trade cache hits against queueing: "predicted TTFT = queueing + compute for the missed part" unifies the two; rejecting early under overload based on predictions greatly improves attainment.
- [x] A router's cache index is maintained from workers' KV events and is approximate; routing mistakes only cost extra compute, never wrong results.
- [x] Changes in the load's shape make the bottleneck jump between prefill and decode, so the xPyD ratio needs a planner adjusting at minute granularity; when interference is mild, you can also skip disaggregation or disaggregate conditionally.
