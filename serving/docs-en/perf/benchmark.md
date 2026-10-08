# Load testing, SLOs and capacity planning

<p class="lead">"How many QPS can this service take?" is the most common practical question in inference roles. Answering it takes three things: clear metric definitions, the right load-testing method, and an understanding of "how latency changes with load". This chapter first lays out the metrics and load-testing tools, then writes a serving simulator based on the roofline model, plots latency against throughput, finds the maximum load that meets the SLO, and finally uses it for capacity planning.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How do TPOT and ITL differ? What does P99 TTFT in a load-test report mean?
    2. Why do load tests at a fixed concurrency and at a fixed request rate (Poisson arrivals) give different results? Which question does each answer?
    3. Why has goodput often already collapsed by the time throughput reaches its maximum?
    4. Given a target QPS and an SLO, how do you estimate how many GPUs are needed?

??? success "Answers (try first, then expand to compare)"
    1. TPOT is the average time per token after the first (averaged within a request); ITL is every gap between adjacent tokens (for spotting stalls, with attention on its tail). P99 TTFT: 99% of requests have a first-token latency at or below this value.
    2. Fixed concurrency is closed-loop: the next request is sent only when one finishes, so sending slows as the service slows, masking overload; a fixed request rate (Poisson arrivals) is open-loop, closer to real traffic, and the queue keeps growing under overload. The former suits measuring "how fast at most under full load", the latter capacity planning and SLO verification.
    3. Past the knee, throughput keeps rising a little (larger batches), but queueing has exploded by then, most requests' latency exceeds the SLO, and the requests that meet the SLO (goodput) collapse instead.
    4. First settle the SLO, then use an open-loop load test (or a simulator) to find the maximum request rate one GPU sustains while meeting the SLO, divide the target QPS by it, and add 20%–30% headroom for bursts. In this chapter's example one GPU handles about 22.6 req/s, so 100 req/s needs 5 GPUs, about 6 with headroom.

## Metrics {#指标}

| Metric | Definition | What it tells you |
| --- | --- | --- |
| TTFT | time from sending the request to receiving the first token | queueing + prefill; the first impression of interactivity |
| TPOT | (time of last token − time of first token) / (output tokens − 1), one value per request | average generation speed |
| ITL | the gap between adjacent tokens, many values per request | stalls: when a long prefill slips in, ITL spikes while averaged TPOT hides it |
| E2E latency | from sending the request to the last token | non-streaming scenarios, one call from an agent |
| Throughput | requests, output tokens or total tokens completed per second | cost |
| goodput | requests completed per second **that meet the SLO** | the throughput that actually counts |

Latency metrics must be read as **percentiles**: P50 reflects the typical case, P99 the worst 1%. A service's SLO is usually defined like "P99 TTFT < 1 s and P99 TPOT < 40 ms".

## How to load test {#怎样压测}

The two load modes answer two different questions:

- **Fixed concurrency** (closed loop): always keep N requests in flight, sending the next as soon as one finishes. It answers "what is the experience with N simultaneous users", but the slower the system, the fewer requests get sent, **masking overload** (known as coordinated omission);
- **Fixed arrival rate** (open loop): requests arrive as a Poisson process at λ per second on average, whether or not the system keeps up. It answers "can the SLO be met when λ requests arrive per second", exposes the latency explosion caused by queueing, and is the right method for capacity planning.

Use a small simulator to see how the two modes measure different things (same system, same load intensity):

<div class="aig-widget" data-widget="open-closed"></div>

Both engines ship load-testing tools:

```bash
# vLLM: random dataset, 1024 in, 256 out, Poisson arrivals at 10 req/s, measuring goodput under the SLO
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --num-prompts 2000 \
    --request-rate 10 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --goodput ttft:1000 tpot:40

# SGLang (since 0.5.20 the entry point is sglang.benchmark.serving, formerly sglang.bench_serving)
python -m sglang.benchmark.serving --backend sglang --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --num-prompts 2000 --request-rate 10
```

Common pitfalls:

- **No warm-up**: the first requests trigger compilation, CUDA Graph capture and memory allocation, so run a round before timing;
- **Prefix cache interference**: when load testing repeatedly with a real dataset, the second round hits the cache heavily; with random datasets, make sure requests differ;
- **Uncontrolled output length**: a model emitting EOS early makes results incomparable, so fix the output length with `--ignore-eos`;
- **The client becomes the bottleneck**: at high QPS a single load-testing process may not keep up, and TTFT is stretched by queueing in the client;
- **Looking only at averages**: average TPOT may be fine while P99 ITL is awful.

## A serving simulator {#一个服务模拟器}

Real load tests need GPUs. To understand "how latency changes with load", we can write a simulator: schedule by the rules of continuous batching and chunked prefill, and estimate each step's time with the roofline model (the larger of "time to compute" and "time to read the weights and KV"; see [the lower bound on latency](llm://inference/estimation/#延迟的下限) in the LLM handbook):

```python title="sim.py"
"""sim.py —— 推理服务的离散事件模拟器：连续批处理 + 分块 prefill，每一步的耗时用屋顶线模型估算。

它不运行模型，只模拟调度与时间，用来回答"在这个负载下 TTFT/TPOT 会是多少、一张卡能扛多少 QPS"。
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class Setup:
    params: float                  # parameters each token computes with (activated parameters for MoE)
    weight_bytes: float            # weight bytes read per step
    kv_bytes_per_token: float
    kv_capacity_tokens: int        # total KV tokens that fit in memory
    peak_flops: float = 989e12     # H100 BF16 dense peak
    bandwidth: float = 3.35e12     # H100 HBM3
    mfu: float = 0.5               # compute utilization prefill can reach
    step_overhead: float = 0.5e-3  # fixed cost per step (scheduling, kernel launches, with CUDA Graphs)
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256


@dataclass
class SimRequest:
    arrival: float
    input_len: int
    output_len: int
    computed: int = 0              # prompt tokens computed so far
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
    for _ in range(num_requests):                                  # Poisson arrivals: exponentially distributed gaps
        t += rng.expovariate(rate)
        reqs.append(SimRequest(t, input_len, output_len))
    now, i, waiting, running, kv_used = 0.0, 0, [], [], 0
    while i < len(reqs) or waiting or running:
        while i < len(reqs) and reqs[i].arrival <= now:
            waiting.append(reqs[i])
            i += 1
        if not waiting and not running:
            now = reqs[i].arrival                                  # idle: jump straight to the next arrival
            continue
        budget = s.max_num_batched_tokens
        batch = []                                                 # (request, tokens this step)
        for r in running:                                          # running requests: decode or continue chunked prefill
            n = 1 if r.computed == r.input_len else min(r.input_len - r.computed, budget)
            if n <= 0:
                break
            batch.append((r, n))
            budget -= n
        while waiting and budget > 0 and len(running) < s.max_num_seqs:
            r = waiting[0]
            if kv_used + r.input_len + r.output_len > s.kv_capacity_tokens:
                break                                              # not enough memory, reserving for the final length: do not admit
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
                    continue                                       # chunked prefill not finished yet, so no token
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

First check that it behaves sensibly at low load, then raise the request rate step by step. The model is Qwen2.5-7B (BF16, about 15 GB of weights, 56 KB of KV per token) on one H100 with about 55 GB of memory for KV:

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

A sanity check first: at low load TTFT is about 34 ms, exactly the compute time to prefill 1024 tokens (2 × 7.6G × 1024 / 495 TFLOPS ≈ 31 ms, plus each step's fixed cost); TPOT is about 5 ms, exactly the time to read the 15 GB of weights once. The simulator's basic behavior is right.

This table shows the single most important phenomenon in inference serving:

- At low load, latency barely changes with the rate while throughput grows linearly with it: batching lets more requests share the weight reads, almost for free;
- Past some point (here between 20 and 25 req/s), TTFT suddenly explodes: the GPU's processing capacity is maxed out, requests pile up in the queue, and queueing time grows without bound;
- **Throughput stays at its maximum after the collapse** (about 6200 tok/s), but goodput plunges toward zero: almost no request meets the SLO. Capacity planning by throughput alone would put the system right at the collapse point.

## Capacity planning {#容量规划}

While meeting the SLO, how many req/s can one GPU take at most? Use binary search:

```python
lo, hi = 1.0, 40.0
for _ in range(12):
    mid = (lo + hi) / 2
    ok = summarize(simulate(qwen7b, mid, 2000, 1024, 256), 1.0, 0.04)["slo_ok"] >= 0.99   # 99% of requests meet the SLO
    lo, hi = (mid, hi) if ok else (lo, mid)
target_qps = 100
print(f"单卡满足 SLO 的最大速率约 {lo:.1f} req/s；支撑 {target_qps} req/s 需要 {math.ceil(target_qps / lo)} 张卡"
      f"（再留 20%～30% 余量应对突发，约 {math.ceil(target_qps / lo * 1.25)} 张）")
```

```text
单卡满足 SLO 的最大速率约 22.6 req/s；支撑 100 req/s 需要 5 张卡（再留 20%～30% 余量应对突发，约 6 张）
```

The other half of capacity is memory: how many requests one GPU can hold at once for a given context (the same widget as in the LLM handbook):

<div class="aig-widget" data-widget="kv-calc"></div>

In practice this number comes from real load tests rather than a simulator, but the approach is exactly the same: **settle the SLO first, find the maximum rate that meets it with an open-loop load test, then compute the GPU count from the target traffic and a redundancy factor**. The simulator's value is answering "what if..." quickly: change the model, the GPU, the parameters or the load, see the trend in seconds, then confirm with real load tests.

For example, the effect of the token budget on ITL when prompts grow to 4096 (4 requests per second):

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

With a budget of 8192, a 4096-token prefill finishes in one step, and the decode requests in the same batch hit a 250 ms stall; average TPOT still looks fine, but the ITL tail is awful. With a budget of 512 ITL is smooth, at the cost of a slightly longer median TTFT (while P99 is actually shorter: queued requests are not held up by one very long step either). This matches the trend measured on the mini engine in the [scheduler chapter](../engine/scheduler.md#token-预算的取舍).

!!! source "Source code"
    - **vLLM**: the load-testing tools are in `vllm/benchmarks/` (`serve.py` for online load tests, `throughput.py` for offline throughput, `latency.py` for single-batch latency, `sweep/` for parameter sweeps), with the command-line entry `vllm bench serve|throughput|latency`. The definition of `--goodput` cites the DistServe paper directly. Server-side metrics are exposed in Prometheus format at `/metrics` (`vllm/v1/metrics/`), and during load tests you should also watch KV Cache usage, queued requests and preemption counts.
    - **SGLang**: `sglang/benchmark/serving.py` (online), `offline_throughput.py`, and `one_batch.py` (a single batch without starting the server); `--enable-metrics` turns on Prometheus metrics.

!!! interview "How to explain it"
    The structure for explaining "how do you assess an inference service's capacity?": **define the SLO** (P99 TTFT, P99 TPOT or ITL) → **build a realistic load** (input and output length distributions, prefix sharing ratio, arrival pattern) → **load test open-loop**, raising the request rate step by step and plotting the latency-throughput curve → find **the rate with maximum goodput that meets the SLO** → compute the GPU count from target traffic plus redundancy. Bonus points: explain why goodput has already collapsed when throughput peaks; mention the coordinated omission problem of closed-loop tests; mention watching server-side metrics at the same time to locate the bottleneck (queueing? KV full? preemption?).

## Exercises {#练习}

**1. Switch to INT4 weights.** Change the simulator's `weight_bytes` to about 4.5 GB (INT4 weights + scales), keeping prefill compute the same. How does the maximum rate one GPU sustains under the SLO change? Why is the change smaller than the speedup in decode?

??? success "Approach"
    The time to read weights in decode drops to about 1/3, and TPOT falls sharply at low load; but near saturation every step includes prefill (compute-bound, which INT4 weight-only quantization cannot speed up), and the amount of KV read is unchanged, so the maximum rate rises far less than 3×. You can verify by running the binary search with `replace(qwen7b, weight_bytes=4.5e9)`. To go further, you need schemes like FP8 (W8A8) that also speed up compute, or PD disaggregation.

**2. Bursty traffic.** Poisson arrivals assume independent requests, but real traffic is often bursty (for example, many users sending requests at once on the hour). What does burstiness do to TTFT? How do you simulate it in a load test?

??? success "Answer"
    At the same average rate, bursts push the instantaneous rate far above capacity, the queue piles up quickly, and P99 TTFT gets noticeably worse, while average throughput may not change at all. To simulate it, draw inter-arrival gaps from a distribution with larger variance: the `--burstiness` parameter of `vllm bench serve` is exactly the shape parameter of a Gamma distribution, where 1 is a Poisson process and less than 1 is burstier. Capacity planning should leave headroom for bursts, or protect the SLO with rate limiting and queue timeouts.

## Summary {#小结}

- [x] TTFT, TPOT, ITL, E2E, throughput and goodput each emphasize something different; read latency at P99, and stalls in the ITL tail.
- [x] Fixed-concurrency load tests mask overload; capacity planning needs fixed-arrival-rate (open-loop) tests.
- [x] The latency-throughput curve has a knee: before it latency is flat and throughput grows linearly, after it queueing explodes; when throughput peaks, goodput has often already collapsed.
- [x] Capacity planning: settle the SLO → find the maximum rate that meets it → compute the GPU count from target traffic plus redundancy; a simulator quickly evaluates all kinds of "what ifs".
