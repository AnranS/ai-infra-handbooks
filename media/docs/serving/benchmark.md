# 评测与压测：延迟、吞吐、质量与可复现

<p class="lead">"我们的优化让 FLUX 快了 2.3 倍"——这句话在没有说清楚分辨率、步数、CFG、batch、精度、预热方式、统计口径和质量变化之前，什么也没说明。生成模型的基准比 LLM 更容易测错：第一次调用含编译，GPU 是异步的，调度器有状态，batch 组成会改结果，而"快了"往往是用画质换的。这一章给出一套可复现的测法：单次生成怎么计时、分阶段怎么拆、吞吐和延迟在负载下怎么变、质量用什么量、报告里必须写什么；每个坑都用一个可以跑的模拟演示。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 测一次生成的延迟，最常见的三个错误是什么？
    2. 为什么生成服务的压测要区分"开环"和"闭环"负载？两者测出的 P99 为什么不一样？
    3. PSNR 40 dB、30 dB、10 dB 分别意味着两张图什么关系？FID 量的是什么、对什么敏感？
    4. 一份生成模型的性能报告至少要写哪些条件？
    5. 怎么验证一个优化（量化、缓存、少步）"没有损失质量"？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 没预热（第一次调用含编译和 kernel 选择，几秒到几十秒）；没同步（GPU 异步执行，不 `synchronize` 计到的是发射时间）；只报均值（长尾被平均掉，而用户感受的是 P99）。再加一个：没固定条件（分辨率、步数、CFG、batch、精度），不同条件下的数字不可比。
    2. 闭环是固定数量的用户每人发完一个等结果再发下一个，到达率受服务速度反压，系统永远不会"过载"，吞吐到容量就封顶、延迟随用户数线性涨；开环是按固定到达率发请求，不管前面有没有处理完，超过容量时队列无限增长、P99 随测试时长发散。线上流量是开环的，所以闭环测出的"吞吐 X 时 P99 Y"不代表线上到达率为 X 时的体验。
    3. 40 dB 以上肉眼无法区分，是"等价"（FP8、注意力 kernel 替换级别的差异）；30 dB 左右能看出细微差别（缓存跳步、蒸馏）；10 dB 是两张不相关的图（换了种子）。FID 量两组图在特征空间的分布距离，不是单张图的差，对样本数（少了偏大）、提示词集合和特征网络敏感，不同条件下的 FID 不可比。
    4. 硬件（卡型、数量、互联）、软件版本（框架、CUDA、注意力后端）、模型与精度、分辨率 / 帧数、步数、CFG、batch、调度器、预热轮数与重复次数、统计口径（P50 / P99 / 均值）、分阶段拆分、显存峰值、质量指标与基线、固定种子是否可复现。
    5. 固定一组提示词和种子，优化前后各生成一遍：先算 PSNR / LPIPS 看逐图差异（等价性），再算一组指标（FID、CLIP score、VBench 这类）看分布差异，最后人工抽看最差的几张——指标正常但某类图退化是缓存和量化的典型问题。

## 测什么

| 指标 | 定义 | 单位 | 容易错在 |
| --- | --- | --- | --- |
| 单次延迟 | 从请求到拿到图 / 视频 | 秒 | 没预热、没同步、没分位数 |
| 分阶段延迟 | 文本编码 / 去噪（每步）/ VAE 解码 | 秒 | 只报总时间，不知道该优化哪里 |
| 吞吐 | 每张卡每秒生成的张数（视频用"每秒生成的视频秒数"） | 张/s、s/s | 不写分辨率和步数；用 batch 堆吞吐却把延迟翻了几倍 |
| 显存峰值 | `max_memory_allocated`（真用）和 `max_memory_reserved`（池里占着） | GB | 只看去噪，VAE 解码时才是峰值 |
| 质量 | 逐图：PSNR / SSIM / LPIPS；分布：FID、CLIP score；视频：VBench 等 | — | 用不同种子比逐图指标；样本太少的 FID |
| 可复现 | 同条件两次生成的 PSNR | dB | batch 组成、注意力后端、调度器状态串了 |

单次延迟要和[算账一章](../perf/accounting.md)的估算对照：实测比估算慢太多，说明 MFU 低（kernel、形状、小 batch），而不是"模型就这么慢"。

## 单次生成怎么计时

一个最小的基准工具：预热、重复、分位数、分阶段，时钟和同步函数都可以注入——这样下面能用一个模拟的 GPU 把常见的错误演示出来，而不用真的卡：

```python title="bench.py"
"""最小的生成基准：预热、重复、分位数、分阶段计时。clock / sync 可注入，便于用模拟时钟复现各种测错的方式。"""
import statistics
import time
from contextlib import contextmanager


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class StageTimer:
    """with timer.stage("denoise"): ...  按阶段累计时间；每次计时前后都同步"""

    def __init__(self, clock=time.perf_counter, sync=lambda: None):
        self.clock, self.sync, self.stages = clock, sync, {}

    @contextmanager
    def stage(self, name):
        self.sync()
        t0 = self.clock()
        yield
        self.sync()
        self.stages[name] = self.stages.get(name, 0.0) + self.clock() - t0


def run_benchmark(generate, warmup=2, runs=10, clock=time.perf_counter, sync=lambda: None):
    """generate() 做一次完整生成。返回预热轮的时间和正式轮的统计"""
    def timed():
        sync()
        t0 = clock()
        generate()
        sync()
        return clock() - t0

    warm = [timed() for _ in range(warmup)]
    times = [timed() for _ in range(runs)]
    return {"warmup": warm, "times": times, "mean": statistics.fmean(times), "min": min(times), "max": max(times),
            "p50": percentile(times, 0.5), "p90": percentile(times, 0.9), "p99": percentile(times, 0.99)}
```

模拟的 GPU：`launch` 只是把工作排进队列（像真的 CUDA 发射一样几乎不花时间），`sync` 才等它做完；第一次调用要先"编译" 12 秒；每次生成约 3 秒，带 2% 的抖动，十次里有一次因为别的原因（时钟降频、碎片整理）多 0.8 秒：

```python
import random
from bench import run_benchmark, StageTimer, percentile

class FakeGPU:
    def __init__(self, seed=0):
        self.now, self.busy_until, self.compiled = 0.0, 0.0, False
        self.rng = random.Random(seed)
    def clock(self):
        return self.now
    def launch(self, seconds):                      # 异步：CPU 侧只花 2 ms 发射，工作排在队列末尾
        self.now += 0.002
        self.busy_until = max(self.busy_until, self.now) + seconds
    def sync(self):                                 # 等 GPU 把队列里的活干完
        self.now = max(self.now, self.busy_until)
    def generate(self):
        if not self.compiled:
            self.launch(12.0)                       # 第一次：编译、kernel 自动调优、显存池扩张
            self.compiled = True
        self.launch(3.0 * self.rng.gauss(1.0, 0.02) + (0.8 if self.rng.random() < 0.1 else 0.0))

def report(tag, r):
    print(f"{tag:<12} mean {r['mean']:5.2f} s   p50 {r['p50']:5.2f}   p99 {r['p99']:5.2f}   max {r['max']:5.2f}")

gpu = FakeGPU()
r = run_benchmark(gpu.generate, warmup=0, runs=5, clock=gpu.clock)              # 错误一：忘了同步
print("忘了同步：   ", " ".join(f"{t * 1000:.0f} ms" for t in r["times"]), "  ← 测到的是发射时间")
gpu = FakeGPU()
report("不预热：", run_benchmark(gpu.generate, warmup=0, runs=10, clock=gpu.clock, sync=gpu.sync))   # 错误二
gpu = FakeGPU()
report("预热 2 轮：", run_benchmark(gpu.generate, warmup=2, runs=10, clock=gpu.clock, sync=gpu.sync))
gpu = FakeGPU()
report("预热，测 50 次：", run_benchmark(gpu.generate, warmup=2, runs=50, clock=gpu.clock, sync=gpu.sync))
```

```text title="输出"
忘了同步：    4 ms 2 ms 2 ms 2 ms 2 ms   ← 测到的是发射时间
不预热：         mean  4.20 s   p50  2.99   p99 13.99   max 15.06
预热 2 轮：      mean  3.01 s   p50  2.99   p99  3.14   max  3.15
预热，测 50 次：   mean  3.03 s   p50  2.99   p99  3.83   max  3.91
```

三个错误各自怎么骗人：忘了同步，测到的是毫秒级的发射时间，"快了一千倍"；不预热，均值被第一轮的编译拉高四成，P99 被那一轮的 15 秒决定；只测 10 次，P99 就是 10 次里最差的一次——这里 10 次恰好没撞上长尾（3.14 秒），50 次才把它测出来（3.83 秒），样本太少时长尾要么被漏掉、要么被一次抖动决定，长尾指标要几十上百次才稳定。

分阶段计时用的是同一套同步规则，每个阶段前后都要 `sync`，否则上一阶段排在队列里的工作会算到下一阶段头上：

```python
gpu = FakeGPU()
gpu.generate()                                              # 预热
timer = StageTimer(clock=gpu.clock, sync=gpu.sync)
with timer.stage("文本编码"):
    gpu.launch(0.02)
with timer.stage("去噪 30 步 × CFG"):
    for _ in range(30):
        gpu.launch(2 * 0.049)
with timer.stage("VAE 解码"):
    gpu.launch(0.16)
total = sum(timer.stages.values())
for name, t in timer.stages.items():
    print(f"{name:<14} {t:5.2f} s  {t / total:4.0%}")
```

```text title="输出"
文本编码            0.02 s    1%
去噪 30 步 × CFG   2.94 s   94%
VAE 解码          0.16 s    5%
```

这就是 SDXL 1024² 在 H100 上的大致分布（数字来自[算账](../perf/accounting.md)和 [VAE](../basics/vae-latent.md) 两章的估算）：去噪占九成多，所以少步、缓存、量化都在那上面做文章；VAE 解码只占 5%，但它决定显存峰值。

## 负载下的吞吐与延迟：开环和闭环

先用推理系统手册里的小模拟器直观看一下两种负载模式的差别：

<div class="aig-widget" data-widget="open-closed"></div>

单次延迟是空载的数字，服务要测的是"到达率为 λ 时的延迟分布"。压测工具有两种发请求的方式，测出来的东西不一样：

- **闭环**（closed loop）：固定 $N$ 个虚拟用户，每人发完一个、等到结果再发下一个。到达率被服务速度反压，系统永远不过载。
- **开环**（open loop）：按固定到达率发（泊松或固定间隔），不管前面的处理完没有。这是线上流量的样子。

用上面的服务时间分布（约 3 秒，十分之一有 0.8 秒的长尾），4 张卡、先到先服务，分别模拟两种负载：

```python
import random

def service_time(rng):
    return 3.0 * rng.gauss(1.0, 0.02) + (0.8 if rng.random() < 0.1 else 0.0)

def open_loop(rate, gpus=4, seconds=1800, seed=1):
    """按泊松到达发请求；free[i] 是第 i 张卡空闲的时刻，先到先服务"""
    rng, t, free, lat = random.Random(seed), 0.0, [0.0] * gpus, []
    while True:
        t += rng.expovariate(rate)
        if t > seconds:
            break
        i = min(range(gpus), key=free.__getitem__)
        free[i] = max(t, free[i]) + service_time(rng)
        lat.append(free[i] - t)
    return lat

def closed_loop(users, gpus=4, seconds=1800, seed=1):
    """users 个用户，收到结果立刻发下一个"""
    rng, free, next_send, lat = random.Random(seed), [0.0] * gpus, [0.0] * users, []
    while True:
        u = min(range(users), key=next_send.__getitem__)         # 下一个发请求的用户
        t = next_send[u]
        if t > seconds:
            break
        i = min(range(gpus), key=free.__getitem__)
        free[i] = max(t, free[i]) + service_time(rng)
        lat.append(free[i] - t)
        next_send[u] = free[i]
    return lat

def row(tag, lat, seconds=1800):
    print(f"{tag:<16} 吞吐 {len(lat) / seconds:5.2f} 请求/s   p50 {percentile(lat, 0.5):6.1f} s   p99 {percentile(lat, 0.99):6.1f} s")

print(f"容量：4 张卡 / 约 3.1 s = {4 / 3.08:.2f} 请求/s")
for users in (2, 4, 8, 16):
    row(f"闭环 {users} 用户", closed_loop(users))
for rate in (0.6, 1.0, 1.2, 1.4):
    row(f"开环 {rate} 请求/s", open_loop(rate))
print("开环 1.4 请求/s，测 2 倍时长：p99 =", f"{percentile(open_loop(1.4, seconds=3600), 0.99):.1f} s")
```

```text title="输出"
容量：4 张卡 / 约 3.1 s = 1.30 请求/s
闭环 2 用户          吞吐  0.65 请求/s   p50    3.0 s   p99    3.9 s
闭环 4 用户          吞吐  1.30 请求/s   p50    3.0 s   p99    3.9 s
闭环 8 用户          吞吐  1.30 请求/s   p50    6.0 s   p99    7.0 s
闭环 16 用户         吞吐  1.31 请求/s   p50   12.1 s   p99   13.6 s
开环 0.6 请求/s      吞吐  0.57 请求/s   p50    3.0 s   p99    4.5 s
开环 1.0 请求/s      吞吐  0.96 请求/s   p50    3.2 s   p99    9.3 s
开环 1.2 请求/s      吞吐  1.17 请求/s   p50    4.9 s   p99   16.9 s
开环 1.4 请求/s      吞吐  1.37 请求/s   p50   44.5 s   p99  113.6 s
开环 1.4 请求/s，测 2 倍时长：p99 = 248.7 s
```

闭环的曲线很"好看"：用户数超过卡数之后吞吐封顶在容量，延迟随用户数线性涨，P99 紧贴 P50——因为队列长度永远不会超过用户数。开环在容量以下就已经有排队（泊松到达会成簇：利用率 77% 时 P99 是空载的 3 倍），但有界；一过容量，队列无限增长，P99 和测试时长一起发散：同样"1.4 请求/s 下 P99 是多少"，测半小时和测一小时得到两个数。这就是为什么线上服务要用开环压测找到"膝点"（延迟开始飙升的到达率），并且要用能把请求"拒绝 / 降级"的策略保护它（见[生成服务的调度](scheduling.md)）。报告压测结果时，负载模式、到达率、测试时长缺一不可。

## 质量：逐图差异和分布差异

优化前后要回答两个不同的问题：**"同一个请求出的图还是不是那张图"**（逐图差异），和 **"整体画得还好不好"**（分布差异）。

逐图差异用 PSNR / SSIM / LPIPS，固定提示词和种子比较：

```python
import numpy as np

rng = np.random.default_rng(0)
def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))      # 像素范围 [0, 1]

img = rng.random((64, 64, 3))
for name, sigma in [("FP8 / 换注意力 kernel", 0.002), ("缓存跳步 / 4 步蒸馏", 0.02), ("换了种子", None)]:
    other = rng.random(img.shape) if sigma is None else np.clip(img + rng.normal(0, sigma, img.shape), 0, 1)
    print(f"{name:<18} PSNR {psnr(img, other):5.1f} dB")
```

```text title="输出"
FP8 / 换注意力 kernel  PSNR  54.0 dB
缓存跳步 / 4 步蒸馏       PSNR  34.0 dB
换了种子               PSNR   7.9 dB
```

经验门槛：40 dB 以上是"同一张图"，30 dB 上下是"同一张图但细节变了"，20 dB 以下已经是另一张图——所以用不同种子的两张图算 PSNR 没有意义，这是最常见的误用。LPIPS 更接近人眼（用网络特征算距离），0.1 以下难以分辨。

分布差异用 FID：把两组图各自过一个特征网络（Inception 或 CLIP），比较特征的均值和协方差——它量的是"这一批图整体像不像那一批"，对单张图不敏感，对样本数很敏感。用独立特征的简化版（协方差取对角）看它的行为：

```python
def fid_diag(x, y):
    """特征各维独立时的 Fréchet 距离：均值差的平方 + 方差项"""
    mu1, mu2, v1, v2 = x.mean(0), y.mean(0), x.var(0), y.var(0)
    return float(((mu1 - mu2) ** 2).sum() + (v1 + v2 - 2 * np.sqrt(v1 * v2)).sum())

real = rng.normal(0, 1, (5000, 64))
for name, other in [("同分布的另一批 5000 张", rng.normal(0, 1, (5000, 64))),
                    ("均值偏 0.3（风格漂了）", rng.normal(0.3, 1, (5000, 64))),
                    ("方差缩到一半（多样性下降）", rng.normal(0, 0.5, (5000, 64))),
                    ("同分布但只有 200 张", rng.normal(0, 1, (200, 64)))]:
    print(f"{name:<18} FID {fid_diag(real, other):6.2f}")
```

```text title="输出"
同分布的另一批 5000 张     FID   0.04
均值偏 0.3（风格漂了）      FID   5.79
方差缩到一半（多样性下降）      FID  16.16
同分布但只有 200 张       FID   0.44
```

同分布的两批 FID 接近 0；风格漂移和多样性下降都会被量出来——后者正是蒸馏和 CFG 过大的典型副作用，逐图指标看不出来；而同分布只因为样本少就被判出一个不小的距离：FID 有正偏差，**样本数不同的 FID 不可比**，论文里常用 5k / 10k / 30k 张，复现时要对齐。视频的质量评测（VBench 这类）把一致性、运动平滑度、主体保持等拆成多个子项，道理相同：固定提示词集合和样本数，比相对值。

验证一个优化"无损"的流程：固定 50～200 条提示词和种子 → 优化前后各生成一遍 → 逐图 PSNR / LPIPS 找出最差的几张人工看 → 整体 FID / CLIP score 和基线比 → 把阈值（比如 PSNR 中位数 > 35 dB、LPIPS 最差 < 0.2、FID 变化 < 1）写进 CI。

## 可复现

[扩散与流匹配一章](../basics/diffusion-inference.md)说过，固定种子不等于固定输出。可复现要锁住：

| 变量 | 做法 |
| --- | --- |
| 种子 | `torch.Generator(device).manual_seed(s)`，噪声在目标设备上生成（CPU 和 GPU 的生成器序列不同） |
| batch 组成 | 单独生成和在 batch 里生成的归约顺序不同；基准里固定 batch |
| 注意力后端 | SDPA 可能在 Flash / 高效 / 数学实现之间自动选择，显式指定 |
| 确定性算法 | `torch.use_deterministic_algorithms(True)`、`CUBLAS_WORKSPACE_CONFIG`，有速度代价，基准时开、线上通常不开 |
| 调度器状态 | 每个请求一个调度器实例（见[采样器与调度器](../basics/schedulers.md)） |
| 编译 / 缓存 | 预热后再测；特征缓存的阈值判断对输入敏感，复现时阈值和校准系数要一致 |
| 版本 | 框架、CUDA、注意力库、模型权重的哈希都记下来 |

一个实用的办法：基准脚本启动时打印并保存全部环境信息，跑完对同一个请求生成两次算 PSNR，低于 40 dB 就把这次结果标成"不可复现"。

## 报告里必须有什么

```
硬件      1 × H100 80 GB SXM，NVLink 不适用；驱动 / CUDA 版本
软件      diffusers x.y / torch x.y / FlashAttention x / SageAttention x；注意力后端
模型      FLUX.1-dev，去噪网络 FP8（按块缩放），T5 bf16，VAE bf16
条件      1024×1024，28 步，CFG 关闭（guidance-distilled），batch 1，Euler 调度器
方法      预热 3 次，测 50 次，固定 20 条提示词 × 种子 0–4；开环压测 30 分钟 × 4 个到达率
结果      延迟 p50 / p90 / p99（秒）；分阶段（文本编码 / 去噪每步 / VAE）；吞吐（张/s/卡）
显存      max_memory_allocated / reserved（GB），峰值出现在哪个阶段
质量      相对 bf16 基线：PSNR 中位数 / 最差，LPIPS 中位数 / 最差，FID（10k 张）
可复现    同请求两次的 PSNR；确定性开关是否打开
```

每一行都对应一种可能的"快了 2.3 倍"的歧义：不写精度就不知道是不是 FP8 带来的；不写步数就不知道是不是把 50 步测成了 28 步；不写质量就不知道代价。

!!! interview "怎么讲清楚"
    讲"怎么给扩散模型的优化做评测"，先说计时的三个坑：预热（首轮含编译）、同步（GPU 异步）、分位数（不只报均值、样本要够）；再说分阶段：去噪、文本编码、VAE 分开计，才知道该优化哪里、显存峰值在哪。然后把单次延迟和服务压测分开：服务要用开环负载找膝点，闭环测出的 P99 不代表线上。质量分两层：逐图（固定种子的 PSNR / LPIPS，40 dB 是"等价"）和分布（FID / CLIP score，固定样本数和提示词集合），验"无损"要两层都过、再人工看最差的几张。最后一句：报告里把硬件、软件、精度、分辨率、步数、CFG、batch、预热、次数、质量全写上，否则"快了 2.3 倍"没有意义。

## 练习

1. 把 `FakeGPU` 的抖动改成 10% 的请求多 3 秒（比如显存不够时触发了整理），比较 10 次和 100 次测出的 P99。要多少次 P99 才稳定？

??? success "参考答案"
    10 次里"至少一次撞上"的概率是 $1 - 0.9^{10} \approx 65\%$，所以 10 次的 P99 要么是 3 秒要么是 6 秒，取决于运气；100 次时 P99 几乎总落在长尾上但数值仍在 6 秒附近抖。经验：P99 要上百次，P99.9 要上千次；更稳的做法是报 P90 加上"长尾的比例和原因"。

2. 开环 1.2 请求/s 的 P99（约 17 秒）比闭环 16 用户的 P99（约 14 秒）还高，可它的吞吐更低。为什么？线上到达率为 1.2 请求/s 时该参考哪个数？容量利用率各是多少？

??? success "参考答案"
    闭环 16 用户利用率 100%，但排队的人最多 12 个，所以 P99 被封在约 4 轮服务时间；开环 1.2 请求/s 利用率约 92%，到达是随机成簇的，偶尔会堆出比 12 个更长的队列，P99 反而更高——利用率接近 1 时等待时间对到达的波动极其敏感。线上是开环的，该参考开环的 17 秒；而且只要到达率波动越过 1.3，队列就开始无限增长——所以线上要留余量（利用率 70%～80%）并有拒绝 / 降级策略。

3. 一个缓存加速把 FLUX 的中位 PSNR 保持在 38 dB，但最差的一张只有 22 dB。要不要上线？怎么查？

??? success "参考答案"
    不能直接上。中位数好只说明多数图没问题，22 dB 意味着某类请求（通常是构图在中间步才确定的、或者提示词变化剧烈的）被跳步破坏了。查法：把最差的 10 张按提示词 / 步数 / 跳过的步分布归类，看探针的累积误差曲线是不是在那些请求上异常；对策是降低阈值、在前几步禁止跳过、或者对这类请求关闭缓存（按请求特征路由，见[特征缓存](../perf/caching.md)）。

## 小结

- [x] 计时三坑：预热（首轮含编译）、同步（GPU 异步）、分位数与样本数；分阶段拆才知道该优化哪里，VAE 解码才是显存峰值。
- [x] 服务压测分开环和闭环：闭环被服务速度反压、永不过载、P99 有界；开环是线上的样子，过容量后 P99 随时长发散，要用它找膝点并留余量。
- [x] 质量分逐图（固定种子的 PSNR / LPIPS，40 dB 等价、30 dB 可见差异、换种子无意义）和分布（FID / CLIP score，对样本数敏感、固定条件比相对值）；验"无损"要两层都过并人工看最差的。
- [x] 可复现要锁种子设备、batch 组成、注意力后端、调度器实例、缓存阈值和版本；报告写全硬件、软件、精度、分辨率、步数、CFG、batch、预热、次数与质量。
