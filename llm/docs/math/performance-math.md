# 性能与服务中的数学

<p class="lead">推理优化的日常决策大多是几个简单公式的应用：数 FLOPs 和字节得到算术强度，与硬件的屋脊点比较判断瓶颈；用 Amdahl 定律估计一项优化能带来多少整体收益；用 Little 定律和排队论理解为什么服务接近满载时延迟会爆炸；用统计学判断压测数字的波动有多大。这一章把这些工具集中讲清楚，并用模拟验证每一个公式。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. $[m, k] \times [k, n]$ 的矩阵乘法算术强度是多少？decode 时 $m$ 很小意味着什么？
    2. 注意力占总时间的 30%，把它加速 2 倍，整体快多少？
    3. 平均每秒到达 20 个请求，每个请求平均耗时 8 秒，系统中平均有多少个请求在处理？
    4. 利用率从 80% 升到 95%，平均排队时间会怎样变化？
    5. 一个页面并行调用 10 次模型，每次调用的 P99 是 1 秒，页面的"P99"是多少？
    6. 200 个样本估计出的 P99 延迟，可信度有多高？

??? success "自测参考答案（先自己答，再展开对照）"
    1. $\dfrac{2mnk}{2(mk + kn + mn)}$ FLOP/字节（BF16）；$k$、$n$ 很大、$m$ 很小时约等于 $m$。decode 时 $m$ 就是 batch 大小，强度只有个位数到几十，远低于屋脊点，是访存瓶颈。
    2. Amdahl 定律：$1 / (0.7 + 0.3/2) \approx 1.18$，整体只快约 18%。
    3. Little 定律 $L = \lambda W = 20 \times 8 = 160$ 个。
    4. 按 M/M/1，平均停留时间是服务时间的 $1/(1-\rho)$ 倍：80% 时是 5 倍、95% 时是 20 倍，约变成原来的 4 倍，尾部延迟涨得更多。
    5. 10 次都在 1 秒内完成的概率是 $0.99^{10} \approx 0.90$，所以 1 秒只是页面的 P90；页面的 P99 对应单次调用的 P99.9 左右。
    6. 不太可信：P99 由最大的约 2 个样本决定，换一批样本波动很大，bootstrap 的置信区间很宽。要让 P99 稳定，通常需要几千个样本。

## FLOPs、字节与算术强度

矩阵乘法 $[m, k] \times [k, n]$ 需要 $2mnk$ 次浮点运算，至少读写 $(mk + kn + mn)$ 个元素。两者之比就是**算术强度**（每字节的运算次数）。与硬件的**屋脊点**（峰值算力 / 带宽，H100 的 BF16 约 295 FLOP/字节）比较：强度低于屋脊点，时间由读写决定（访存受限）；高于屋脊点，由计算决定（计算受限）。能达到的算力是 $\min(\text{峰值}, \text{强度} \times \text{带宽})$，这就是**屋顶线模型**。

![图：屋顶线模型（H100，bf16）](../assets/figures/roofline.svg){.aig-svg}

```python
import math
import random
import statistics

peak, bandwidth = 989e12, 3.35e12                                  # H100：BF16 稠密峰值与 HBM 带宽
print(f"H100 屋脊点：{peak / bandwidth:.0f} FLOP/字节")
print("场景                      m      算术强度    可达算力（TFLOPS）  受限于")
for name, m in [("decode，batch 1", 1), ("decode，batch 8", 8), ("decode，batch 64", 64),
                ("decode，batch 256", 256), ("prefill，4096 token", 4096)]:
    k = n = 4096                                                   # 一个 4096×4096 的 BF16 权重
    flops, bytes_ = 2 * m * n * k, 2 * (m * k + k * n + m * n)
    intensity = flops / bytes_
    attainable = min(peak, intensity * bandwidth)
    print(f"{name:20s} {m:6d} {intensity:10.1f} {attainable / 1e12:16.0f}      {'计算' if attainable == peak else '访存'}")
```

```text
H100 屋脊点：295 FLOP/字节
场景                      m      算术强度    可达算力（TFLOPS）  受限于
decode，batch 1            1        1.0                3      访存
decode，batch 8            8        8.0               27      访存
decode，batch 64          64       62.1              208      访存
decode，batch 256        256      227.6              762      访存
prefill，4096 token     4096     1365.3              989      计算
```

$k = n$ 很大、$m$ 很小时，强度约等于 $m$：decode 时 $m$ 就是 batch 大小，所以 batch 1 的 decode 只能发挥 H100 算力的 0.3%。这一张表就是"为什么要批处理""为什么 decode 要量化权重""为什么 prefill 与 decode 性质不同"的全部数学基础（详细的应用见[一个 token 的完整旅程](../synthesis/token-journey.md#decode-的时间花在哪里)）。

拖动 m，看同一个矩阵乘怎么从访存受限走到计算受限；换成 H20 这类算力低、带宽高的卡，屋脊点会左移很多：

<div class="aig-widget" data-widget="roofline"></div>

## Amdahl 定律：优化一部分，能快多少

如果某部分占总时间的比例为 $f$，把它加速 $s$ 倍，整体加速比是

$$
S = \frac{1}{(1 - f) + f / s}
$$

即使 $s \to \infty$，整体加速也不会超过 $1/(1-f)$。

```pycon
>>> amdahl = lambda f, s: 1 / ((1 - f) + f / s)
>>> round(amdahl(0.3, 2), 2), round(amdahl(0.3, 100), 2)          # 注意力占 30%：加速 2 倍 / 100 倍
(1.18, 1.42)
>>> round(amdahl(0.8, 2), 2)                                       # 如果它占 80%
1.67
```

所以优化之前先 profiling：确认要优化的部分占多大比例，再决定值不值得做。这也是[Profiling 一章](serving://perf/profiling/)反复强调"先量化瓶颈"的原因。

## Little 定律

一个稳定的系统中，**平均在系统中的请求数 = 平均到达率 × 平均停留时间**：

$$
L = \lambda W
$$

它不依赖于任何分布假设，非常好用：每秒 20 个请求、每个平均 8 秒，系统中平均有 160 个请求在处理，也就需要能同时容纳 160 个请求的 KV Cache；反过来，引擎的最大并发和平均延迟决定了它能承受的最大到达率。用一个单服务台排队的模拟验证：

```python
random.seed(0)

def single_server(arrival_rate, service_rate, n=200_000):
    """泊松到达、指数服务时间、先来先服务：返回每个请求的 (到达时间, 完成时间)。"""
    t, free_at, records = 0.0, 0.0, []
    for _ in range(n):
        t += random.expovariate(arrival_rate)
        start = max(t, free_at)
        free_at = start + random.expovariate(service_rate)
        records.append((t, free_at))
    return records

records = single_server(8.0, 10.0)
horizon = records[-1][0]
events = sorted([(a, 1) for a, _ in records] + [(d, -1) for _, d in records])
in_system, last, area = 0, 0.0, 0.0
for time, delta in events:                                         # 对"系统中的请求数"做时间平均
    if time > horizon:
        break
    area += in_system * (time - last)
    in_system += delta
    last = time
W = statistics.mean(d - a for a, d in records)
print(f"实测平均请求数 L = {area / horizon:.3f}；λ × W = 8 × {W:.3f} = {8 * W:.3f}")
```

```text
实测平均请求数 L = 4.013；λ × W = 8 × 0.501 = 4.005
```

## 排队论：为什么接近满载时延迟爆炸

![图：利用率与排队延迟](../assets/figures/queue-latency.svg){.aig-svg}

最简单的排队模型 M/M/1（泊松到达、指数服务时间、一个服务台）中，服务率为 $\mu$、到达率为 $\lambda$、利用率 $\rho = \lambda / \mu$，平均停留时间是

$$
W = \frac{1}{\mu - \lambda} = \frac{1}{\mu} \cdot \frac{1}{1 - \rho}
$$

当 $\rho \to 1$，$W \to \infty$。模拟验证，并看看尾部延迟：

```python
mu = 10.0                                                           # 每秒能处理 10 个请求
print("利用率   平均延迟（模拟）  平均延迟（公式）   P99 延迟")
for lam in (2, 5, 8, 9, 9.5):
    waits = sorted(d - a for a, d in single_server(lam, mu))
    print(f"{lam / mu:5.0%}   {statistics.mean(waits):10.3f} s   {1 / (mu - lam):10.3f} s   {waits[int(0.99 * len(waits))]:8.2f} s")
```

```text
利用率   平均延迟（模拟）  平均延迟（公式）   P99 延迟
  20%        0.125 s        0.125 s       0.58 s
  50%        0.199 s        0.200 s       0.91 s
  80%        0.486 s        0.500 s       2.14 s
  90%        0.988 s        1.000 s       4.08 s
  95%        1.716 s        2.000 s       6.46 s
```

利用率从 50% 到 90%，平均延迟涨了 5 倍；从 90% 到 95%，公式给出的平均延迟再翻一倍（模拟值偏低，是因为高负载下排队长度的波动很大、收敛很慢），P99 达到 6.5 秒。推理服务不是单服务台（它能批处理，服务时间随批大小变化），但定性规律完全相同：推理系统手册的[压测一章](serving://perf/benchmark/)用模拟器看到，请求速率超过某个点之后 TTFT 突然爆炸。这就是**不能把服务规划在满载附近**的原因：留出余量，本质上是把利用率控制在延迟曲线平坦的区间。

## 尾部延迟会被放大

一个请求如果要**并行**调用 $k$ 个服务（例如 Agent 并行调用多个工具、一个页面同时生成多段内容），它的延迟由最慢的那个决定。每个调用有 1% 的概率超过自己的 P99，那么至少一个超过的概率是 $1 - 0.99^k$：

```python
latency = lambda: random.lognormvariate(math.log(100), 0.5)         # 单次调用延迟（毫秒），中位数 100
single = sorted(latency() for _ in range(100_000))
p99 = single[int(0.99 * len(single))]
fanout = [max(latency() for _ in range(10)) for _ in range(20_000)]
slow = sum(x > p99 for x in fanout) / len(fanout)
print(f"单次调用：中位数 {single[50_000]:.0f} ms，P99 {p99:.0f} ms")
print(f"并行 10 次：中位数 {sorted(fanout)[10_000]:.0f} ms；超过单次 P99 的比例 {slow:.1%}（公式 1 - 0.99^10 = {1 - 0.99 ** 10:.1%}）")
```

```text
单次调用：中位数 100 ms，P99 318 ms
并行 10 次：中位数 211 ms；超过单次 P99 的比例 10.1%（公式 1 - 0.99^10 = 9.6%）
```

并行 10 次时，近 10% 的请求会遇到单次调用的 P99，中位数也翻了一倍。所以面向这类负载的服务更要控制尾部延迟（P99、P999），而不只是平均值。

## 性能测量的统计学

压测得到的 P99、平均 TPOT 都是**样本统计量**，有随机波动（大模型手册[概率与采样](probability.md#蒙特卡洛的误差)一章讨论过估计误差）。分位数的置信区间可以用 **bootstrap** 估计：从样本中有放回地重复抽样，每次算一个 P99，看这些 P99 的分布：

```python
def percentile99(xs):
    s = sorted(xs)
    return s[int(0.99 * len(s))]

for n in (200, 5000):
    sample = [latency() for _ in range(n)]
    boots = sorted(percentile99(random.choices(sample, k=n)) for _ in range(500))
    print(f"{n:5d} 个样本：P99 = {percentile99(sample):.0f} ms，95% 置信区间 [{boots[12]:.0f}, {boots[487]:.0f}] ms")
```

```text
  200 个样本：P99 = 233 ms，95% 置信区间 [204, 313] ms
 5000 个样本：P99 = 320 ms，95% 置信区间 [310, 341] ms
```

只有 200 个样本时，P99 的置信区间宽达一百多毫秒，而且这一次的估计值（233 ms）比 5000 个样本给出的 320 ms 低了四分之一；两个配置的 P99 相差 10% 完全可能只是噪声。5000 个样本时区间才收窄到三十毫秒左右。经验法则：估计 P99 至少需要几千个样本，比较两个方案时要多次重复压测、看置信区间是否重叠，并且固定其他变量（预热、前缀缓存状态、并发的其他负载）。

!!! interview "面试怎么答"
    性能题有四个工具：屋顶线（GEMM 的算术强度约等于较小的那一维，decode 时就是 batch 大小，H100 BF16 的屋脊点约 295 FLOP/字节）；Amdahl 定律（整体加速不超过 $1/(1-f)$）；Little 定律 $L = \lambda W$（并发 = 到达率 × 延迟）；排队论（等待时间随 $1/(1-\rho)$ 增长，利用率从 80% 升到 95% 等待约变成 4 倍）。再补两点：一个页面并行调用 10 次模型时，页面的 P99 约是单次调用的 P99.9；压测的 P99 要几千个样本才可信。

## 练习

**1. 批处理的屋顶线。** 在上面的表中，batch 为多少时，4096×4096 的 BF16 GEMM 刚好达到 H100 的屋脊点？这个数字为什么与"decode 的最佳 batch"不完全一样？

??? success "参考答案"
    强度约为 $\frac{2mnk}{2(mk + kn + mn)}$，$k = n = 4096$ 时，令 $\frac{4096m}{2m + 4096} = 295$，解得 $m \approx 345$（$m$ 不再远小于 $k$ 时，强度略低于 $m$）。但 decode 还要读 KV Cache，而注意力的强度不随 batch 增长（等于 GQA 分组数），所以随着 batch 增大，瓶颈会转移到读 KV；再加上 TPOT 的 SLO 限制，实际的最佳 batch 通常更小，需要压测确定（见[一个 token 的完整旅程](../synthesis/token-journey.md#decode-的时间花在哪里)）。

**2. 用 Little 定律估算显存。** 一个服务平均每秒 50 个请求，平均上下文 4000 token，平均 E2E 延迟 6 秒，每 token 的 KV 为 56 KB（Qwen2.5-7B）。平均需要多少 GB 的 KV Cache？

??? success "参考答案"
    系统中平均有 $L = 50 \times 6 = 300$ 个请求，每个约 $4000 \times 56\ \text{KB} \approx 229\ \text{MB}$，合计约 69 GB。这还只是平均值，要按峰值（到达率与上下文长度都更大时）留余量。这类估算在容量规划时非常实用。

## 小结

- [x] 算术强度 = FLOPs / 字节，与屋脊点比较判断瓶颈；GEMM 的强度约等于较小的那一维，decode 时就是 batch 大小。
- [x] Amdahl 定律：只有占比大的部分值得优化；整体加速不超过 $1/(1-f)$。
- [x] Little 定律 $L = \lambda W$：并发、到达率、延迟三者知二求一。
- [x] 排队延迟随利用率按 $1/(1-\rho)$ 增长，接近满载时平均与尾部延迟都会爆炸；并行调用会放大尾部延迟。
- [x] 压测数字有统计波动，用 bootstrap 估计分位数的置信区间，P99 需要几千个样本。
