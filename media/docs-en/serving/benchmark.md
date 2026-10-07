# Benchmarking and load testing: latency, throughput, quality and reproducibility

<p class="lead">"Our optimisation made FLUX 2.3 times faster" — that sentence says nothing until the resolution, the step count, the guidance, the batch, the precision, the warm-up, the statistics and the change in quality are all stated. A generative model's benchmark is easier to get wrong than an LLM's: the first call includes compilation, the GPU is asynchronous, the scheduler is stateful, the batch composition changes the result, and "faster" is often paid for in image quality. This chapter gives a reproducible method: how to time one generation, how to break it down by stage, how the throughput and latency move under load, what to measure quality with, and what a report must state; every pitfall is demonstrated with a runnable simulation.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What are the three commonest mistakes in measuring one generation's latency?
    2. Why does load testing a generation service have to distinguish open-loop from closed-loop load? Why do the two give different P99s?
    3. What do 40 dB, 30 dB and 10 dB of PSNR say about the relationship between two images? What does FID measure, and what is it sensitive to?
    4. What conditions does a generative model's performance report have to state at a minimum?
    5. How do you verify that an optimisation (quantization, caching, fewer steps) "loses no quality"?

??? success "Answers for the self-test (answer first, then open this)"
    1. No warm-up (the first call includes compilation and kernel selection, seconds to tens of seconds); no synchronisation (the GPU executes asynchronously, and without a `synchronize` what is measured is the launch time); reporting only the mean (the long tail is averaged away while what the user feels is the P99). And one more: not fixing the conditions (the resolution, the steps, the guidance, the batch, the precision), which makes numbers from different conditions incomparable.
    2. Closed loop is a fixed number of users each waiting for one result before sending the next, so the arrival rate is back-pressured by the service's speed and the system never overloads: the throughput caps at the capacity and the latency rises linearly with the user count. Open loop sends at a fixed arrival rate regardless of whether the earlier ones are done, so beyond the capacity the queue grows without bound and the P99 diverges with the test's duration. Production traffic is open loop, so a closed-loop "P99 of Y at a throughput of X" does not describe the experience at an arrival rate of X in production.
    3. Above 40 dB the eye cannot tell them apart, which is "equivalent" (the level of difference from FP8 or swapping an attention kernel); around 30 dB a fine difference is visible (cache skipping, distillation); 10 dB is two unrelated images (a different seed). FID measures the distance between two sets of images' distributions in a feature space, not one image's difference; it is sensitive to the sample count (too few inflates it), to the prompt set and to the feature network, so FIDs under different conditions are incomparable.
    4. The hardware (the card, the count, the interconnect), the software versions (the framework, CUDA, the attention backend), the model and the precision, the resolution and frame count, the steps, the guidance, the batch, the scheduler, the warm-up rounds and the repetitions, the statistics (p50 / p99 / mean), the breakdown by stage, the memory peak, the quality metrics and the baseline, and whether a fixed seed reproduces.
    5. Fix a set of prompts and seeds and generate once before and once after: compute the PSNR and LPIPS first for the per-image difference (equivalence), then a set of metrics (FID, the CLIP score, something like VBench) for the distribution difference, and finally look at the worst few by hand — normal metrics with one kind of image degraded is the classic problem with caching and quantization.

## What to measure {#测什么}

| Metric | Definition | Unit | Easy to get wrong by |
| --- | --- | --- | --- |
| Single latency | from the request to having the image or video | seconds | no warm-up, no synchronisation, no percentiles |
| Per-stage latency | the text encoding / the denoising (per step) / the VAE decode | seconds | reporting only the total, so where to optimise is unknown |
| Throughput | images per second per card (for video, seconds of video generated per second) | images/s, s/s | not stating the resolution and steps; stacking throughput with a batch while multiplying the latency |
| Memory peak | `max_memory_allocated` (really used) and `max_memory_reserved` (held in the pool) | GB | watching only the denoising, when the peak is during the VAE decode |
| Quality | per image: PSNR / SSIM / LPIPS; distribution: FID, the CLIP score; video: VBench and the like | — | comparing per-image metrics across different seeds; an FID on too few samples |
| Reproducibility | the PSNR of two generations under the same conditions | dB | the batch composition, the attention backend, a scheduler's state crossing over |

The single latency has to be checked against [the accounting chapter](../perf/accounting.md)'s estimate: measuring far slower than the estimate means the utilization is low (kernels, shapes, a small batch), not that "the model is simply that slow".

## How to time one generation {#单次生成怎么计时}

A minimal benchmarking tool: warm-up, repetition, percentiles and stages, with the clock and the synchronisation function both injectable — which lets a simulated GPU below demonstrate the common mistakes without a real card:

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

The simulated GPU: `launch` only queues the work (taking almost no time, like a real CUDA launch) and `sync` is what waits for it; the first call has to "compile" for 12 seconds; each generation is about 3 seconds with 2% of jitter, and one in ten takes another 0.8 seconds for some other reason (a clock throttle, defragmentation):

```python
import random
from bench import run_benchmark, StageTimer, percentile

class FakeGPU:
    def __init__(self, seed=0):
        self.now, self.busy_until, self.compiled = 0.0, 0.0, False
        self.rng = random.Random(seed)
    def clock(self):
        return self.now
    def launch(self, seconds):                      # asynchronous: the CPU side spends only 2 ms launching and the work goes to the end of the queue
        self.now += 0.002
        self.busy_until = max(self.busy_until, self.now) + seconds
    def sync(self):                                 # wait for the GPU to finish the work in the queue
        self.now = max(self.now, self.busy_until)
    def generate(self):
        if not self.compiled:
            self.launch(12.0)                       # the first time: compilation, kernel autotuning, the memory pool growing
            self.compiled = True
        self.launch(3.0 * self.rng.gauss(1.0, 0.02) + (0.8 if self.rng.random() < 0.1 else 0.0))

def report(tag, r):
    print(f"{tag:<12} mean {r['mean']:5.2f} s   p50 {r['p50']:5.2f}   p99 {r['p99']:5.2f}   max {r['max']:5.2f}")

gpu = FakeGPU()
r = run_benchmark(gpu.generate, warmup=0, runs=5, clock=gpu.clock)              # mistake one: forgot to synchronise
print("忘了同步：   ", " ".join(f"{t * 1000:.0f} ms" for t in r["times"]), "  ← 测到的是发射时间")
gpu = FakeGPU()
report("不预热：", run_benchmark(gpu.generate, warmup=0, runs=10, clock=gpu.clock, sync=gpu.sync))   # mistake two
gpu = FakeGPU()
report("预热 2 轮：", run_benchmark(gpu.generate, warmup=2, runs=10, clock=gpu.clock, sync=gpu.sync))
gpu = FakeGPU()
report("预热，测 50 次：", run_benchmark(gpu.generate, warmup=2, runs=50, clock=gpu.clock, sync=gpu.sync))
```

```text title="output"
忘了同步：    4 ms 2 ms 2 ms 2 ms 2 ms   ← 测到的是发射时间
不预热：         mean  4.20 s   p50  2.99   p99 13.99   max 15.06
预热 2 轮：      mean  3.01 s   p50  2.99   p99  3.14   max  3.15
预热，测 50 次：   mean  3.03 s   p50  2.99   p99  3.83   max  3.91
```

How each of the three mistakes deceives: forgetting to synchronise measures the launch time in milliseconds, "a thousand times faster"; not warming up has the mean pulled forty percent higher by the first round and the P99 decided by that round's 15 seconds; measuring only 10 times makes the P99 the worst of the 10 — and here the 10 happen to miss the long tail (3.14 seconds) while 50 bring it out (3.83 seconds). With too few samples the long tail is either missed or decided by a single piece of jitter, and a tail metric takes dozens to hundreds of runs to stabilise.

Per-stage timing uses the same synchronisation rule, with a `sync` before and after each stage, or the previous stage's work still queued is charged to the next one:

```python
gpu = FakeGPU()
gpu.generate()                                              # warm up
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

```text title="output"
文本编码            0.02 s    1%
去噪 30 步 × CFG   2.94 s   94%
VAE 解码          0.16 s    5%
```

That is roughly SDXL 1024²'s distribution on an H100 (the numbers come from the estimates in [the accounting](../perf/accounting.md) and [the VAE](../basics/vae-latent.md) chapters): the denoising is over ninety percent, which is why fewer steps, caching and quantization all work on it; the VAE decode is only 5%, but it decides the memory peak.

## Throughput and latency under load: open loop and closed loop {#负载下的吞吐与延迟开环和闭环}

The inference-systems handbook's small simulator shows the two load modes' difference directly first:

<div class="aig-widget" data-widget="open-closed"></div>

The single latency is the number under no load; what a service has to measure is the latency distribution at an arrival rate of λ. A load-testing tool has two ways of sending requests, and they measure different things:

- **Closed loop**: a fixed $N$ virtual users, each sending one and waiting for the result before the next. The arrival rate is back-pressured by the service's speed and the system never overloads.
- **Open loop**: sending at a fixed arrival rate (Poisson or at a fixed interval) regardless of whether the earlier ones are done. This is what production traffic looks like.

With the service-time distribution above (about 3 seconds, a tenth with an 0.8-second tail), 4 cards and first come first served, simulate both modes:

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
        u = min(range(users), key=next_send.__getitem__)         # the next user to send a request
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

```text title="output"
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

The closed-loop curve looks very good: past the card count the throughput caps at the capacity, the latency rises linearly with the user count, and the P99 hugs the P50 — because the queue can never be longer than the user count. Open loop already queues below the capacity (Poisson arrivals come in clusters: at 77% utilization the P99 is 3 times the unloaded figure) but is bounded; past the capacity the queue grows without bound and the P99 diverges along with the test's duration: the same question "what is the P99 at 1.4 requests/s" gives two numbers for half an hour and for an hour. That is why a production service has to find the knee (the arrival rate at which the latency starts to climb) with an open-loop test, and has to protect it with a policy that can reject or degrade requests (see [Scheduling a generation service](scheduling.md)). When reporting a load test, the load mode, the arrival rate and the duration are all indispensable.

## Quality: the per-image difference and the distribution difference {#质量逐图差异和分布差异}

Before and after an optimisation, two different questions have to be answered: **"is the image from the same request still that image"** (the per-image difference), and **"does it still paint well overall"** (the distribution difference).

The per-image difference uses PSNR / SSIM / LPIPS, comparing at a fixed prompt and seed:

```python
import numpy as np

rng = np.random.default_rng(0)
def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))      # pixels in the range [0, 1]

img = rng.random((64, 64, 3))
for name, sigma in [("FP8 / 换注意力 kernel", 0.002), ("缓存跳步 / 4 步蒸馏", 0.02), ("换了种子", None)]:
    other = rng.random(img.shape) if sigma is None else np.clip(img + rng.normal(0, sigma, img.shape), 0, 1)
    print(f"{name:<18} PSNR {psnr(img, other):5.1f} dB")
```

```text title="output"
FP8 / 换注意力 kernel  PSNR  54.0 dB
缓存跳步 / 4 步蒸馏       PSNR  34.0 dB
换了种子               PSNR   7.9 dB
```

The rules of thumb: above 40 dB is "the same image", around 30 dB is "the same image with changed detail", and below 20 dB is already another image — so a PSNR between two images from different seeds is meaningless, which is the commonest misuse. LPIPS is closer to the eye (a distance over network features), and below 0.1 is hard to tell apart.

The distribution difference uses FID: put both sets of images through a feature network (Inception or CLIP) and compare the features' means and covariances — it measures how much this batch of images looks like that batch overall, is insensitive to any one image and very sensitive to the sample count. A simplified version with independent features (a diagonal covariance) shows its behaviour:

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

```text title="output"
同分布的另一批 5000 张     FID   0.04
均值偏 0.3（风格漂了）      FID   5.79
方差缩到一半（多样性下降）      FID  16.16
同分布但只有 200 张       FID   0.44
```

Two batches from the same distribution have an FID near 0; a style drift and a loss of diversity are both measured — and the latter is exactly the classic side effect of distillation and of too much guidance, which the per-image metrics cannot see. Meanwhile the same distribution with fewer samples is given a far from small distance: FID is positively biased, so **FIDs over different sample counts are incomparable**. Papers commonly use 5k, 10k or 30k images, and reproducing one means matching that. Video quality evaluation (VBench and the like) breaks consistency, motion smoothness, subject preservation and more into separate sub-scores, on the same principle: fix the prompt set and the sample count, and compare relative values.

The procedure for verifying an optimisation "lossless": fix 50 to 200 prompts and seeds → generate once before and once after → find the worst few by per-image PSNR and LPIPS and look at them → compare the overall FID and CLIP score with the baseline → put the thresholds (say a median PSNR above 35 dB, a worst LPIPS below 0.2, an FID change under 1) into CI.

## Reproducibility {#可复现}

As [the diffusion and flow matching chapter](../basics/diffusion-inference.md) said, a fixed seed does not mean a fixed output. Reproducibility means pinning down:

| Variable | How |
| --- | --- |
| The seed | `torch.Generator(device).manual_seed(s)`, with the noise generated on the target device (a CPU and a GPU generator produce different sequences) |
| The batch composition | generating alone and generating in a batch reduce in different orders; fix the batch in a benchmark |
| The attention backend | SDPA may choose automatically among the flash, efficient and math implementations, so specify it |
| Deterministic algorithms | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`, at a cost in speed; on for a benchmark, usually off in production |
| The scheduler's state | one scheduler instance per request (see [Samplers and schedulers](../basics/schedulers.md)) |
| Compilation and caching | measure after warming up; a feature cache's threshold decision is sensitive to the input, so the threshold and the calibration coefficients have to match to reproduce |
| Versions | record the hashes of the framework, CUDA, the attention library and the model weights |

A practical measure: have the benchmark script print and save the whole environment at startup, and afterwards generate the same request twice and compute the PSNR, marking the run "not reproducible" below 40 dB.

## What a report must have {#报告里必须有什么}

<!-- i18n:diagram 4bc84a6536 -->
```
hardware        1 x H100 80 GB SXM, NVLink not applicable; the driver and CUDA versions
software        diffusers x.y / torch x.y / FlashAttention x / SageAttention x; the attention backend
model           FLUX.1-dev, the denoising network in FP8 (scaled per block), T5 in bf16, the VAE in bf16
conditions      1024x1024, 28 steps, guidance off (guidance-distilled), a batch of 1, the Euler scheduler
method          3 warm-up runs and 50 measured, 20 fixed prompts x seeds 0-4; an open-loop test of 30 minutes at each of 4 arrival rates
results         the latency p50 / p90 / p99 (seconds); per stage (text encoding / denoising per step / VAE); the throughput (images/s/card)
memory          max_memory_allocated / reserved (GB), and which stage the peak is in
quality         against the bf16 baseline: the median and worst PSNR, the median and worst LPIPS, the FID (10k images)
reproducible    the PSNR of the same request twice; whether the determinism switches are on
```

Every line corresponds to one possible ambiguity in "2.3 times faster": without the precision there is no telling whether FP8 did it; without the step count there is no telling whether 50 steps were measured as 28; without the quality there is no telling what it cost.

!!! interview "How to answer in an interview"
    Asked how to evaluate an optimisation to a diffusion model, give timing's three pitfalls first: warm-up (the first round includes compilation), synchronisation (the GPU is asynchronous) and percentiles (not the mean alone, and enough samples). Then the stages: timing the denoising, the text encoding and the VAE separately is what tells you where to optimise and where the memory peak is. Then separate the single latency from the service load test: a service has to find the knee with an open-loop load, and a closed-loop P99 does not describe production. Quality has two layers: per image (PSNR and LPIPS at a fixed seed, with 40 dB meaning "equivalent") and the distribution (FID and the CLIP score at a fixed sample count and prompt set), and verifying "lossless" means passing both and then looking at the worst few by hand. One last sentence: put the hardware, the software, the precision, the resolution, the steps, the guidance, the batch, the warm-up, the repetitions and the quality in the report, or "2.3 times faster" means nothing.

## Exercises {#练习}

1. Change `FakeGPU`'s jitter to 10% of requests taking 3 seconds longer (say a defragmentation triggered by running short of memory), and compare the P99 from 10 runs with that from 100. How many runs does the P99 take to stabilise?

??? success "Answer"
    The probability of hitting it at least once in 10 is $1 - 0.9^{10} \approx 65\%$, so a P99 over 10 runs is either 3 seconds or 6 depending on luck; over 100 the P99 almost always lands on the tail but the value still moves around 6 seconds. The rule of thumb: a P99 takes hundreds of runs and a P99.9 thousands; a steadier practice is to report the P90 plus "the tail's proportion and its cause".

2. Open loop at 1.2 requests/s has a higher P99 (about 17 seconds) than closed loop with 16 users (about 14), yet its throughput is lower. Why? Which number should be used for production at an arrival rate of 1.2 requests/s? What is each one's capacity utilization?

??? success "Answer"
    Closed loop with 16 users runs at 100% utilization, but at most 12 are queued, so the P99 is capped at about 4 service times; open loop at 1.2 requests/s is at about 92% utilization and the arrivals cluster randomly, occasionally piling up a queue longer than 12, so the P99 is in fact higher — near a utilization of 1 the waiting time is extremely sensitive to variation in the arrivals. Production is open loop, so the open-loop 17 seconds is the number to use; and as soon as the arrival rate's variation crosses 1.3 the queue starts growing without bound — which is why production needs headroom (70% to 80% utilization) and a rejection or degradation policy.

3. A caching speedup keeps FLUX's median PSNR at 38 dB, but the worst image is only 22 dB. Should it go live? How do you investigate?

??? success "Answer"
    Not as it is. A good median only says most images are fine; 22 dB means some class of request (usually one whose composition is only settled in the middle steps, or whose prompt changes sharply) was broken by the skipping. How to investigate: classify the worst 10 by prompt, step count and the distribution of skipped steps, and see whether the probe's accumulated error curve is abnormal on those requests. The remedies are lowering the threshold, forbidding skips in the first few steps, or turning the cache off for that class of request (routing by request features, see [Feature caching](../perf/caching.md)).

## Summary {#小结}

- [x] Timing's three pitfalls: warm-up (the first round includes compilation), synchronisation (the GPU is asynchronous), and percentiles with enough samples; breaking it down by stage is what tells you where to optimise, and the VAE decode is where the memory peaks.
- [x] A service load test is either open or closed loop: closed loop is back-pressured by the service's speed, never overloads and has a bounded P99; open loop is what production looks like, with a P99 that diverges with the duration past the capacity, and it is what finds the knee and sizes the headroom.
- [x] Quality splits into per image (PSNR and LPIPS at a fixed seed; 40 dB is equivalent, 30 dB is a visible difference, and a different seed is meaningless) and the distribution (FID and the CLIP score, sensitive to the sample count, compared as relative values under fixed conditions); verifying "lossless" means passing both and looking at the worst by hand.
- [x] Reproducibility means pinning the seed's device, the batch composition, the attention backend, the scheduler instances, the cache thresholds and the versions; a report states the hardware, the software, the precision, the resolution, the steps, the guidance, the batch, the warm-up, the repetitions and the quality in full.
