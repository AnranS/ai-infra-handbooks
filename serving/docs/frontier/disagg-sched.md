# 分离式架构的全局调度：KV 感知路由、xPyD 配比与过载控制

<p class="lead">单个推理引擎的调度器决定"这一步跑哪些请求"；分离式架构还多了一层<b>全局调度</b>：一个请求该发给哪个 prefill 实例、哪个 decode 实例，prefill 和 decode 各要多少个实例，过载时拒绝谁。Mooncake 的 Conductor、NVIDIA Dynamo 的 Router 与 Planner、llm-d 的推理网关、SGLang 的 Router 都在做这件事。这一章用两个模拟回答三个问题：路由该看缓存还是看负载、xPyD 的配比为什么要随时间调整、过载时为什么"提前拒绝"反而让更多请求达标。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 只看缓存命中的路由会出什么问题？只看负载呢？
    2. "预计 TTFT 最小"的路由是怎么同时考虑缓存和排队的？
    3. 过载时为什么要提前拒绝请求？拒绝的依据是什么？
    4. 为什么 prefill 和 decode 实例的配比不能一次配好？
    5. Dynamo 的 Router 怎样知道每个实例缓存了哪些块？

## 全局调度要做的事

以 KV 为中心的分离式架构（Mooncake 的说法是 KVCache-centric）里，全局调度器看到的是一个集群：若干 prefill 实例、若干 decode 实例、一个跨实例的 KV 缓存池（[上一部分](../comm/kv-storage.md)）。它要决定：

1. **prefill 发给谁**：哪个实例缓存了这个请求的前缀、哪个实例排队短；
2. **decode 发给谁**：哪个实例还有 KV 空间、TPOT 能否达标；
3. **收不收**：预计 TTFT、TPOT 达不到目标时，是排队等、还是立刻拒绝；
4. **配比**：prefill 和 decode 实例各要多少，负载变化时怎样调整。

## 路由：缓存与排队的权衡

模拟 8 个 prefill 实例，负载是多轮对话（20 种系统提示词、400 个会话，每轮追加 300 个 token），请求按泊松过程到达；每个实例有自己的前缀缓存（LRU，约 19 万 token），prefill 只算没命中的部分。比较四种路由，以及"过载时提前拒绝"：

```python
import random
from collections import OrderedDict

BLOCK, INSTANCES, SPEED, SLO = 64, 8, 20000, 2.0     # 块大小；prefill 实例数；每个实例每秒 prefill 的 token 数；TTFT 目标（秒）


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
    busy = [0.0] * INSTANCES                          # 每个实例的队列排到什么时候
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

        def ttft(i):                                  # 预计首 token 时间：排队 + 算没命中的部分
            return max(busy[i] - t, 0) + (len(tokens) - hit(i)) / SPEED

        if policy == "随机":
            i = rng.randrange(INSTANCES)
        elif policy == "最少排队":
            i = min(range(INSTANCES), key=lambda j: busy[j])
        elif policy == "只看缓存":
            i = max(range(INSTANCES), key=lambda j: (hit(j), -busy[j]))
        else:                                         # 预计 TTFT 最小：同时考虑缓存命中和排队
            i = min(range(INSTANCES), key=ttft)
        if reject and ttft(i) > SLO:                  # 过载时提前拒绝：注定超时的请求不进队列
            rejected += 1
            continue
        h = hit(i)
        busy[i] = max(busy[i], t) + (len(tokens) - h) / SPEED
        ttfts.append(busy[i] - t)
        hit_tokens, total_tokens = hit_tokens + h, total_tokens + len(tokens)
        for k in ks:
            cache[i][k] = True
            cache[i].move_to_end(k)
        while len(cache[i]) > 3000:                   # 每个实例能缓存 3000 块（约 19 万 token）
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

```text title="输出"
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

- **轻载时**，看缓存的两种策略都好：命中率高意味着每个请求要算的 token 少，排队自然短；随机和最少排队只能命中系统提示词，命中率只有一半；
- **重载时**，不看缓存的两种策略彻底崩溃（算力都浪费在重复 prefill 上，队列越排越长）；**只看缓存**会把同一类会话压在同一个实例上形成热点，P99 达到 31 秒；**预计 TTFT 最小**——排队时间加上"没命中的部分要算多久"——在命中率和负载之间自动权衡，P99 降到 11 秒；
- **提前拒绝**：路由时预计 TTFT 已经超过目标的请求，排进去也注定超时，还会拖累后面的请求。直接拒绝其中的 2.1%，其余请求的 P99 降到 2 秒以内，达标率从 86% 升到 98%。

这就是各家全局调度器的共同思路：

- **Mooncake 的 Conductor** 为每个请求同时选择 prefill 和 decode 实例，依据是前缀缓存能复用多少、实例的排队情况和 TTFT / TPOT 目标；过载时基于预测提前拒绝——不只看 prefill，也预测 decode 侧到时候有没有空间，避免 prefill 做完了 decode 却接不住；
- **NVIDIA Dynamo 的 KV Router**：每个推理 worker 在 KV 块被存入、淘汰时发布事件，Router 据此维护一个"哪个 worker 有哪些块"的全局索引（按前缀组织的树），路由时把"前缀重叠的块数"和"各 worker 当前的负载"合成一个代价；
- **llm-d** 在 Kubernetes 的推理网关里做同样的事：端点选择器按前缀缓存和负载给各个副本打分；
- **SGLang Router**（Rust 实现）的 cache-aware 策略为每个 worker 维护一棵近似的基数树，负载差距超过阈值时退化成最短队列（练习题库里有这道题的简化版）。

路由器维护的索引永远是近似的：实例自己会淘汰块，事件有延迟，所以命中数只是估计。好在路由错了只会多算一些 prefill，不会算错结果。

## xPyD 配比：跟着负载变

PD 分离把 prefill 和 decode 拆成两个资源池，好处是各自优化，代价是**配比**成了一个需要持续调整的量。负载的形态一变，瓶颈就从一侧跳到另一侧：

```python
import math

P_CAP, D_CAP, TOTAL = 20000, 3000, 24        # 一个 prefill 实例每秒算多少提示词 token；一个 decode 实例在 TPOT 目标下每秒出多少 token；实例总数
PHASES = [                                   # (时段, 每秒请求数, 平均提示词长度, 平均输出长度, 前缀缓存命中率)
    ("白天：对话", 100, 2000, 500, 0.6),
    ("傍晚：长文档总结", 30, 16000, 300, 0.2),
    ("夜间：代码智能体", 25, 6000, 2000, 0.8),
]


def need(rate, isl, osl, hit):
    return math.ceil(rate * isl * (1 - hit) / P_CAP), math.ceil(rate * osl / D_CAP)


static = (12, 12)                            # 按"平均"一次性配好的 12P12D
print(f"固定 {static[0]}P{static[1]}D 与按时段调整的对比（共 {TOTAL} 个实例）：")
for name, rate, isl, osl, hit in PHASES:
    p, d = need(rate, isl, osl, hit)
    load = (rate * isl * (1 - hit) / (static[0] * P_CAP), rate * osl / (static[1] * D_CAP))
    verdict = "；".join(f"{side}负载 {x:.0%}" + ("（过载）" if x > 1 else "") for side, x in zip(("prefill ", "decode "), load))
    print(f"  {name}：需要 {p}P{d}D（{'够用' if p + d <= TOTAL else '不够'}）；固定配比下 {verdict}")
```

```text title="输出"
固定 12P12D 与按时段调整的对比（共 24 个实例）：
  白天：对话：需要 4P17D（够用）；固定配比下 prefill 负载 33%；decode 负载 139%（过载）
  傍晚：长文档总结：需要 20P3D（够用）；固定配比下 prefill 负载 160%（过载）；decode 负载 25%
  夜间：代码智能体：需要 2P17D（够用）；固定配比下 prefill 负载 12%；decode 负载 139%（过载）
```

同样 24 个实例，三个时段各自都够用，但没有一个固定的配比能同时满足三个时段：对话和智能体（输出长、缓存命中高）要的是 decode，长文档（提示词长、命中低）要的是 prefill。所以分离式系统需要一个**规划器**：Dynamo 的 Planner 根据 TTFT、TPOT 等指标和排队情况增减 prefill / decode worker；Mooncake 等系统也支持实例在 prefill 和 decode 角色之间切换。切换的代价是加载权重、预热（CUDA Graph、JIT kernel）的时间，以及 decode 实例上正在运行的请求要迁走或跑完——所以规划器通常按分钟级别调整，秒级的波动交给路由和排队去吸收。

另一个思路是**不完全分离**：负载轻或者提示词短时，prefill 和 decode 放在同一个实例上用分块 prefill 混合调度，只在提示词很长时才走 PD 分离（vLLM、SGLang 的一些部署方式都支持"条件分离"）。分离的收益主要来自长提示词与 decode 的相互干扰；干扰不严重时，分离带来的传输和配比问题可能得不偿失。

!!! interview "面试怎么答"
    系统设计题里讲到 PD 分离后，面试官常追问"路由怎么做""配比怎么定"。路由：只看缓存会热点、只看负载会重复 prefill，要用"预计 TTFT = 排队 + 未命中部分的计算时间"这样的统一代价，过载时按预测提前拒绝（本章模拟：达标率从 86% 到 98%）；索引靠 worker 发布的 KV 事件维护，是近似的。配比：随负载形态变化，瓶颈会在两侧之间跳动，需要规划器按分钟级调整，并说明切换角色的代价。能提到 Mooncake Conductor、Dynamo 的 Router 和 Planner，说明你了解业界方案。

## 练习

**1. 为什么"只看缓存"在轻载时最好？** 在本章的模拟里，轻载时"只看缓存"的 P99 甚至比"预计 TTFT 最小"更低。解释原因，并说明这个优势在什么情况下会消失。

??? success "参考答案"
    轻载时几乎不排队，TTFT 主要由"要算的 token 数"决定，命中率最高的策略自然最快；"预计 TTFT 最小"偶尔会为了避开一点排队而放弃命中，反而多算。负载升高后，同一类会话集中到同一个实例上形成热点，排队时间开始主导，只看缓存的策略无法把请求分散出去，P99 急剧恶化（本章 70 请求/秒时 31 秒）。实际系统里常见的做法是：负载差距不大时优先缓存，差距超过阈值时改为看负载（SGLang Router 的平衡阈值），或者直接用统一的代价函数。

**2. 提前拒绝的代价。** 提前拒绝让达标率从 86% 升到 98%，但拒绝了 2.1% 的请求。在真实服务里，"拒绝"意味着什么？还有什么替代办法？

??? success "参考思路"
    拒绝在用户侧通常表现为"服务繁忙，请重试"，或者降级到更小的模型、更短的输出。替代办法包括：按优先级拒绝（付费用户、交互式请求优先，批处理请求延后）；把请求排到一个"慢队列"里、放宽它的 SLO；触发规划器扩容（但扩容需要分钟级的时间）；以及在入口做限流，避免过载真的发生。提前拒绝的关键在于"提前"：在请求消耗任何 prefill 算力之前就做出决定，否则既浪费了算力、又让用户等了很久才失败。

## 小结

- [x] 全局调度决定 prefill / decode 实例的选择、准入和配比；Mooncake Conductor、Dynamo 的 Router 与 Planner、llm-d 网关、SGLang Router 都在做这件事。
- [x] 路由要在缓存命中和排队之间权衡：用"预计 TTFT = 排队 + 未命中部分的计算"统一两者；过载时基于预测提前拒绝，能大幅提高达标率。
- [x] 路由器的缓存索引靠 worker 的 KV 事件维护，是近似的；路由错了只会多算，不会算错。
- [x] 负载形态变化会让瓶颈在 prefill 和 decode 之间跳动，xPyD 配比需要规划器按分钟级调整；干扰不严重时也可以不分离或条件分离。
