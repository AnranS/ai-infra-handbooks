# 压测、SLO 与容量规划

<p class="lead">"这个服务能扛多少 QPS？"是推理岗最常被问到的实际问题。回答它需要三样东西：清楚的指标定义、正确的压测方法、对"延迟随负载如何变化"的理解。这一章先把指标和压测工具讲清楚，再写一个基于屋顶线模型的服务模拟器，画出延迟与吞吐的关系、找到满足 SLO 的最大负载，最后用它做容量规划。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. TPOT 和 ITL 有什么区别？压测报告里的 P99 TTFT 是什么意思？
    2. 按固定并发压测和按固定请求速率（泊松到达）压测，结果为什么不一样？各适合回答什么问题？
    3. 为什么吞吐量达到最大值的时候，goodput 往往已经崩溃了？
    4. 给定目标 QPS 和 SLO，怎样估算需要多少张卡？

??? success "自测参考答案（先自己答，再展开对照）"
    1. TPOT 是首 token 之后平均每个 token 的时间（对一个请求取平均），ITL 是相邻两个 token 之间的每一个间隔（看卡顿，关注它的尾部）。P99 TTFT：99% 的请求的首 token 延迟都不超过这个值。
    2. 固定并发是闭环的：一个请求完成才发下一个，服务变慢时发送也跟着变慢，过载被掩盖了；固定请求速率（泊松到达）是开环的，更接近真实流量，过载时排队会一直增长。前者适合测"满负载下最多多快"，后者适合容量规划和 SLO 验证。
    3. 吞吐在负载越过拐点之后还会继续上升一点（batch 更大），但这时排队已经爆炸，大部分请求的延迟超出了 SLO，满足 SLO 的请求（goodput）反而崩溃。
    4. 先确定 SLO，再用开环压测（或模拟器）找出单卡在满足 SLO 时能扛的最大请求速率，用目标 QPS 除以它，再加 20%～30% 的余量应对突发。本章的例子里单卡约 22.6 req/s，100 req/s 需要 5 张，加上余量约 6 张。

## 指标

| 指标 | 定义 | 关注什么 |
| --- | --- | --- |
| TTFT | 请求发出到收到第一个 token 的时间 | 排队 + prefill；交互体验的第一印象 |
| TPOT | (最后一个 token 时间 − 第一个 token 时间) / (输出 token 数 − 1)，每个请求一个值 | 平均生成速度 |
| ITL | 相邻两个 token 之间的间隔，每个请求有很多个值 | 卡顿：一次长 prefill 插进来，ITL 会出现尖峰，而 TPOT 平均后看不出来 |
| E2E 延迟 | 请求发出到最后一个 token | 非流式场景、Agent 的一次调用 |
| 吞吐 | 每秒完成的请求数、输出 token 数、总 token 数 | 成本 |
| goodput | 每秒完成的**满足 SLO 的**请求数 | 真正有用的吞吐 |

延迟指标一定要看**分位数**：P50 反映典型情况，P99 反映最差的 1%。服务的 SLO 通常这样定义："P99 TTFT < 1 s 且 P99 TPOT < 40 ms"。

## 怎样压测

两种负载模式回答两个不同的问题：

- **固定并发**（闭环）：始终保持 N 个请求在途，一个结束就立刻发下一个。它回答"N 个用户同时使用时体验如何"，但系统越慢、发出的请求越少，会**掩盖过载**（称为 coordinated omission）；
- **固定到达速率**（开环）：请求按泊松过程到达，平均每秒 λ 个，不管系统是否跟得上。它回答"每秒来 λ 个请求时能否满足 SLO"，能暴露排队导致的延迟爆炸，是做容量规划的正确方法。

用一个小模拟器看两种模式测出的东西有什么不同（同样的系统，同样的负载强度）：

<div class="aig-widget" data-widget="open-closed"></div>

两个引擎都自带压测工具：

```bash
# vLLM：随机数据集，输入 1024、输出 256，泊松到达 10 req/s，统计 SLO 下的 goodput
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --num-prompts 2000 \
    --request-rate 10 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --goodput ttft:1000 tpot:40

# SGLang（0.5.20 起入口为 sglang.benchmark.serving，旧名 sglang.bench_serving）
python -m sglang.benchmark.serving --backend sglang --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --num-prompts 2000 --request-rate 10
```

容易踩的坑：

- **不预热**：第一次请求会触发编译、CUDA Graph 捕获、内存分配，要先跑一轮再计时；
- **前缀缓存干扰**：用真实数据集重复压测时，第二轮会大量命中缓存；随机数据集要确保各请求内容不同；
- **输出长度不受控**：模型提前输出 EOS 会让结果不可比，用 `--ignore-eos` 固定输出长度；
- **客户端成为瓶颈**：高 QPS 时单个压测进程可能跟不上，TTFT 被客户端的排队拉长；
- **只看平均值**：平均 TPOT 正常，P99 ITL 可能很糟。

## 一个服务模拟器

真实压测需要 GPU。为了理解"延迟如何随负载变化"，可以写一个模拟器：按连续批处理与分块 prefill 的规则调度，每一步的耗时用屋顶线模型估算（取"算完"和"读完权重与 KV"两者中较大的，见大模型手册的[延迟下限](llm://inference/estimation/#延迟的下限)）：

```python title="sim.py"
"""sim.py —— 推理服务的离散事件模拟器：连续批处理 + 分块 prefill，每一步的耗时用屋顶线模型估算。

它不运行模型，只模拟调度与时间，用来回答"在这个负载下 TTFT/TPOT 会是多少、一张卡能扛多少 QPS"。
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class Setup:
    params: float                  # 每个 token 参与计算的参数量（MoE 取激活参数）
    weight_bytes: float            # 每步要读的权重字节数
    kv_bytes_per_token: float
    kv_capacity_tokens: int        # 显存能放下的 KV token 总数
    peak_flops: float = 989e12     # H100 BF16 稠密峰值
    bandwidth: float = 3.35e12     # H100 HBM3
    mfu: float = 0.5               # prefill 能达到的算力利用率
    step_overhead: float = 0.5e-3  # 每步的固定开销（调度、kernel 启动，已用 CUDA Graph）
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256


@dataclass
class SimRequest:
    arrival: float
    input_len: int
    output_len: int
    computed: int = 0              # 已计算的提示词 token
    generated: int = 0
    first_token: float | None = None
    finish: float | None = None
    token_times: list = field(default_factory=list)


def step_time(s: Setup, num_tokens: int, context_tokens: int) -> float:
    compute = 2 * s.params * num_tokens / (s.peak_flops * s.mfu)
    memory = (s.weight_bytes + s.kv_bytes_per_token * context_tokens) / s.bandwidth
    return max(compute, memory) + s.step_overhead


def simulate(s: Setup, rate: float, num_requests: int, input_len: int, output_len: int, seed: int = 0):
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for _ in range(num_requests):                                  # 泊松到达：间隔服从指数分布
        t += rng.expovariate(rate)
        reqs.append(SimRequest(t, input_len, output_len))
    now, i, waiting, running, kv_used = 0.0, 0, [], [], 0
    while i < len(reqs) or waiting or running:
        while i < len(reqs) and reqs[i].arrival <= now:
            waiting.append(reqs[i])
            i += 1
        if not waiting and not running:
            now = reqs[i].arrival                                  # 空闲：直接跳到下一个请求到达
            continue
        budget = s.max_num_batched_tokens
        batch = []                                                 # (请求, 本步 token 数)
        for r in running:                                          # 运行中的请求：decode 或继续分块 prefill
            n = 1 if r.computed == r.input_len else min(r.input_len - r.computed, budget)
            if n <= 0:
                break
            batch.append((r, n))
            budget -= n
        while waiting and budget > 0 and len(running) < s.max_num_seqs:
            r = waiting[0]
            if kv_used + r.input_len + r.output_len > s.kv_capacity_tokens:
                break                                              # 显存不够：按最终长度预留，不接收
            waiting.pop(0)
            running.append(r)
            kv_used += r.input_len + r.output_len
            n = min(r.input_len, budget)
            batch.append((r, n))
            budget -= n
        context = sum(r.computed + r.generated for r in running)
        now += step_time(s, sum(n for _, n in batch), context)
        for r, n in batch:
            if r.computed < r.input_len:
                r.computed += n
                if r.computed < r.input_len:
                    continue                                       # 分块 prefill 还没算完，不产生 token
            r.generated += 1
            r.token_times.append(now)
            if r.first_token is None:
                r.first_token = now
            if r.generated == r.output_len:
                r.finish = now
                running.remove(r)
                kv_used -= r.input_len + r.output_len
    return reqs


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(math.ceil(p / 100 * len(values))) - 1)]


def summarize(reqs, ttft_slo: float, tpot_slo: float) -> dict:
    ttft = [r.first_token - r.arrival for r in reqs]
    tpot = [(r.finish - r.first_token) / (r.output_len - 1) for r in reqs]
    duration = max(r.finish for r in reqs) - min(r.arrival for r in reqs)
    good = sum(a <= ttft_slo and b <= tpot_slo for a, b in zip(ttft, tpot))
    return {"ttft_p50": percentile(ttft, 50), "ttft_p99": percentile(ttft, 99),
            "tpot_p50": percentile(tpot, 50), "tpot_p99": percentile(tpot, 99),
            "output_tps": sum(r.output_len for r in reqs) / duration, "goodput": good / duration,
            "slo_ok": good / len(reqs)}
```

先检查它在低负载下的表现是否合理，再逐步提高请求速率。模型取 Qwen2.5-7B（BF16，权重约 15 GB，每 token KV 56 KB），单张 H100，KV 显存约 55 GB：

```python
import math
from dataclasses import replace
from sim import Setup, percentile, simulate, summarize

qwen7b = Setup(params=7.6e9, weight_bytes=15.2e9, kv_bytes_per_token=57344, kv_capacity_tokens=int(55e9 / 57344))
print("速率   TTFT P50/P99 (ms)   TPOT P50/P99 (ms)   输出吞吐 (tok/s)   goodput (req/s)   满足 SLO")
for rate in (1, 5, 10, 15, 20, 25, 30, 40):
    m = summarize(simulate(qwen7b, rate, 2000, 1024, 256), ttft_slo=1.0, tpot_slo=0.04)
    print(f"{rate:4d}   {m['ttft_p50'] * 1e3:6.0f} / {m['ttft_p99'] * 1e3:6.0f}      "
          f"{m['tpot_p50'] * 1e3:5.1f} / {m['tpot_p99'] * 1e3:5.1f}      {m['output_tps']:8.0f}          "
          f"{m['goodput']:6.1f}         {m['slo_ok']:.0%}")
```

```text
速率   TTFT P50/P99 (ms)   TPOT P50/P99 (ms)   输出吞吐 (tok/s)   goodput (req/s)   满足 SLO
   1       34 /     60        5.2 /   5.6           252             1.0         100%
   5       35 /     90        6.0 /   6.9          1256             4.9         100%
  10       37 /    121        7.5 /   8.9          2504             9.8         100%
  15       39 /    166       10.0 /  12.8          3742            14.6         100%
  20       65 /    242       15.5 /  22.9          4966            19.4         100%
  25      756 /   2012       40.1 /  40.7          6122             8.2         34%
  30     7342 /  13321       40.3 /  40.9          6178             1.8         8%
  40    14677 /  29698       40.3 /  41.2          6187             0.4         1%
```

先做合理性检查：低负载时 TTFT 约 34 ms，正好是 prefill 1024 个 token 的计算时间（2 × 7.6G × 1024 / 495 TFLOPS ≈ 31 ms，加上每步的固定开销）；TPOT 约 5 ms，正好是读一遍 15 GB 权重的时间。模拟器的基本行为是对的。

这张表展示了推理服务最重要的一个现象：

- 负载较低时，延迟几乎不随速率变化，吞吐随速率线性增长：批处理让更多的请求分摊权重读取，几乎是免费的；
- 超过某个点（这里在 20～25 req/s 之间）之后，TTFT 突然爆炸：GPU 的处理能力达到上限，请求开始在队列里越积越多，排队时间无限增长；
- **吞吐在崩溃之后仍然维持在最大值**（约 6200 tok/s），但 goodput 迅速跌向零：几乎没有请求满足 SLO。只看吞吐做容量规划，会把系统规划在崩溃点上。

## 容量规划

在满足 SLO 的前提下，一张卡最多能扛多少 req/s？用二分查找：

```python
lo, hi = 1.0, 40.0
for _ in range(12):
    mid = (lo + hi) / 2
    ok = summarize(simulate(qwen7b, mid, 2000, 1024, 256), 1.0, 0.04)["slo_ok"] >= 0.99   # 99% 的请求满足 SLO
    lo, hi = (mid, hi) if ok else (lo, mid)
target_qps = 100
print(f"单卡满足 SLO 的最大速率约 {lo:.1f} req/s；支撑 {target_qps} req/s 需要 {math.ceil(target_qps / lo)} 张卡"
      f"（再留 20%～30% 余量应对突发，约 {math.ceil(target_qps / lo * 1.25)} 张）")
```

```text
单卡满足 SLO 的最大速率约 22.6 req/s；支撑 100 req/s 需要 5 张卡（再留 20%～30% 余量应对突发，约 6 张）
```

容量的另一半是显存：一张卡在给定上下文下能同时放下多少请求（大模型手册的同一个工具）：

<div class="aig-widget" data-widget="kv-calc"></div>

实际工作中，这个数字来自真实压测而不是模拟器，但思路完全相同：**先确定 SLO，再用开环压测找到满足 SLO 的最大速率，然后按目标流量和冗余系数算卡数**。模拟器的价值在于快速回答"如果……会怎样"：换模型、换卡、改参数、改负载，几秒钟就能看到趋势，再用真实压测确认。

例如，提示词变长到 4096 时，token 预算对 ITL 的影响（每秒 4 个请求）：

```python
print("token 预算   TTFT P50/P99 (ms)   TPOT P99 (ms)   ITL P99 / 最大 (ms)")
for budget in (512, 2048, 8192):
    reqs = simulate(replace(qwen7b, max_num_batched_tokens=budget), 4, 1000, 4096, 256)
    m = summarize(reqs, 2.0, 0.05)
    itl = [b - a for r in reqs for a, b in zip(r.token_times, r.token_times[1:])]
    print(f"{budget:8d}     {m['ttft_p50'] * 1e3:5.0f} / {m['ttft_p99'] * 1e3:5.0f}        {m['tpot_p99'] * 1e3:5.1f}"
          f"          {percentile(itl, 99) * 1e3:5.1f} / {max(itl) * 1e3:5.1f}")
```

```text
token 预算   TTFT P50/P99 (ms)   TPOT P99 (ms)   ITL P99 / 最大 (ms)
     512       152 /   612         12.6           16.2 /  16.2
    2048       194 /   611         19.0           63.5 /  63.5
    8192       133 /   725         20.7          127.3 / 252.3
```

预算为 8192 时，一个 4096 token 的 prefill 在一步内完成，同批次的 decode 请求会遇到一次 250 ms 的卡顿；平均 TPOT 看起来还不错，ITL 的尾部却很糟。预算为 512 时 ITL 平稳，代价是 TTFT 的中位数略长（P99 反而更短：排队中的请求也不会被一个超长的步拖住）。这与[调度器一章](../engine/scheduler.md#token-预算的取舍)在迷你引擎上测到的趋势一致。

!!! source "源码对照"
    - **vLLM**：压测工具在 `vllm/benchmarks/`（`serve.py` 在线压测，`throughput.py` 离线吞吐，`latency.py` 单批延迟，`sweep/` 参数扫描），命令行入口是 `vllm bench serve|throughput|latency`。`--goodput` 的定义直接引用了 DistServe 论文。服务端的指标通过 Prometheus 格式在 `/metrics` 暴露（`vllm/v1/metrics/`），压测时应同时观察 KV Cache 使用率、排队请求数、抢占次数。
    - **SGLang**：`sglang/benchmark/serving.py`（在线）、`offline_throughput.py`、`one_batch.py`（单批次，不启动服务）；`--enable-metrics` 开启 Prometheus 指标。

!!! interview "怎么讲清楚"
    "怎么评估一个推理服务的容量？"标准答案的结构：**定义 SLO**（P99 TTFT、P99 TPOT 或 ITL）→ **构造贴近真实的负载**（输入输出长度分布、前缀共享比例、到达模式）→ **开环压测**，逐步提高请求速率，画出延迟–吞吐曲线 → 找到 **goodput 最大、满足 SLO 的速率** → 按目标流量加冗余算出卡数。加分项：解释为什么吞吐最大时 goodput 已经崩溃；提到闭环压测的 coordinated omission 问题；提到要同时看服务端指标来定位瓶颈（排队？KV 满了？抢占？）。

## 练习

**1. 换成 INT4 权重。** 把模拟器中的 `weight_bytes` 改为约 4.5 GB（INT4 权重 + 缩放因子），prefill 的算力不变，单卡满足 SLO 的最大速率会怎样变化？为什么变化没有 decode 速度的提升那么大？

??? success "参考思路"
    decode 读权重的时间缩短到约 1/3，低负载时 TPOT 大幅下降；但在接近饱和的区间，每一步都混有 prefill（计算受限，INT4 weight-only 量化不能加速计算），而且 KV 的读取量不变，所以最大速率的提升远小于 3 倍。可以直接用 `replace(qwen7b, weight_bytes=4.5e9)` 跑一遍二分查找验证。要进一步提升，需要 FP8（W8A8）这类同时加速计算的方案，或者 PD 分离。

**2. 突发流量。** 泊松到达假设请求彼此独立，真实流量常常是突发的（例如整点大量用户同时发起请求）。突发对 TTFT 有什么影响？压测时如何模拟？

??? success "参考答案"
    同样的平均速率，突发流量会让瞬时速率远超容量，队列快速堆积，TTFT 的 P99 显著变差，而平均吞吐可能没有变化。模拟方法是让到达间隔服从方差更大的分布：`vllm bench serve` 的 `--burstiness` 参数就是 Gamma 分布的形状参数，1 对应泊松过程，小于 1 表示更突发。容量规划时要按突发情况留余量，或者用限流与排队超时来保护 SLO。

## 小结

- [x] TTFT、TPOT、ITL、E2E、吞吐、goodput 各有侧重；延迟看 P99，卡顿看 ITL 的尾部。
- [x] 固定并发压测会掩盖过载，容量规划要用固定到达速率（开环）压测。
- [x] 延迟–吞吐曲线有一个拐点：之前延迟平稳、吞吐线性增长，之后排队爆炸；吞吐最大时 goodput 往往已经崩溃。
- [x] 容量规划：确定 SLO → 找满足 SLO 的最大速率 → 按目标流量加冗余算卡数；模拟器可以快速评估各种"如果"。
